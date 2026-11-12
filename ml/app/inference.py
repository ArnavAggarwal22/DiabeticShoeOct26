"""Prediction: raw logs -> features -> active model -> per-foot risk, drivers and safety flags."""
from __future__ import annotations

import math
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import storage
from .features import FEATURE_VERSION, HIGH_PRESSURE_KPA, session_features
from .parsing import parse_sensor_csv
from .rules import rule_scores
from .schema import LABEL_SEVERITY, LABELS, OVERALL_STATUS, ZONES

_lock = threading.Lock()
_cache: dict[str, dict] = {}

ZONE_NAMES = {"toe": "toe (hallux)", "met": "metatarsal heads", "heel": "heel", "mid": "midfoot"}


def feature_label(name: str) -> str:
    """Human-readable feature description for explanations / importance charts."""
    fixed = {
        "p_peak_p95": "Peak plantar pressure (95th pct)",
        "p_peak_p95_per_kg": "Peak pressure per kg body weight",
        "p_forefoot_heel_ratio": "Forefoot-to-heel load ratio",
        "p_total_mean_per_kg": "Total plantar load per kg",
        "p_sustained_frac": "Time under sustained high load",
        "p_peak_zone_pti_per_min": "Pressure-time integral at peak zone",
        "gait_cycles_per_min": "Gait cadence (steps/min)",
        "gait_stance_frac": "Stance (loaded) time fraction",
        "t_mean": "Mean skin temperature", "t_max": "Max skin temperature", "t_std": "Temperature variability",
        "t_range": "Temperature range", "t_slope_per_min": "Temperature trend",
        "rh_mean": "Mean in-shoe humidity", "rh_max": "Max in-shoe humidity",
        "rh_frac_over_75": "Time with humidity > 75%", "rh_slope_per_min": "Humidity trend",
        "rec_duration_s": "Recording length",
        "bi_dT_mean": "Temperature vs. other foot (ΔT)", "bi_abs_dT_mean": "|ΔT| between feet",
        "bi_dT_max_rolling": "Max sustained ΔT vs. other foot", "bi_dT_frac_over_2_2": "Time with ΔT > 2.2 °C",
        "bi_p_peak_asym": "Peak pressure asymmetry", "bi_rh_diff": "Humidity vs. other foot",
        "pt_age": "Age", "pt_sex_male": "Sex", "pt_weight_kg": "Body weight", "pt_bmi": "BMI",
        "pt_diabetes_years": "Diabetes duration", "pt_hba1c": "HbA1c", "pt_neuropathy": "Peripheral neuropathy",
        "pt_pad": "Peripheral artery disease", "pt_deformity": "Foot deformity", "pt_prior_ulcer": "Prior ulcer",
        "pt_prior_amputation": "Prior amputation",
        "rule_prs": "Pressure risk index (rule)", "rule_trs": "Thermal risk index (rule)",
        "rule_mrs": "Moisture risk index (rule)", "rule_tbi": "Tissue breakdown index (rule)",
        "rule_dfu": "Legacy rule-based DFU score", "side_left": "Foot side",
    }
    if name in fixed:
        return fixed[name]
    for z, zn in ZONE_NAMES.items():
        if name.startswith(f"bi_p_{z}"):
            return f"{zn.capitalize()} pressure asymmetry"
        if name.startswith(f"p_{z}_"):
            stat = name[len(f"p_{z}_"):]
            return {
                "mean": f"Mean pressure - {zn}", "p95": f"Peak pressure - {zn}", "max": f"Max pressure - {zn}",
                "std": f"Pressure variability - {zn}", "cv": f"Pressure CV - {zn}",
                "mean_per_kg": f"Pressure per kg - {zn}", "frac_over_200": f"Time > 200 kPa - {zn}",
                "share": f"Load share - {zn}",
            }.get(stat, name)
    return name


class NoModelError(RuntimeError):
    pass


def load_active() -> dict:
    meta = storage.active_model()
    if meta is None:
        raise NoModelError("No active model. Train one from the admin page.")
    with _lock:
        art = _cache.get(meta["version"])
        if art is None:
            path = Path(meta["path"])
            if not path.exists():
                raise NoModelError(f"Model file for {meta['version']} is missing.")
            art = joblib.load(path)
            _cache.clear()
            _cache[meta["version"]] = art
    return {**art, "registry": meta}


def invalidate_cache() -> None:
    with _lock:
        _cache.clear()


def _risk(proba: np.ndarray, classes: list[str]) -> np.ndarray:
    sev = np.array([LABEL_SEVERITY[c] for c in classes])
    return proba @ sev


def _num(v, nd=2):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return None
    return round(float(v), nd)


def _isnum(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def _drivers(model, x: pd.DataFrame, classes: list[str], ref: dict, base_risk: float, top: int = 5) -> list[dict]:
    """Local explanation against a typical low-risk foot (training low-risk medians).

    Two-point Shapley approximation, averaged per feature:
      * remove: risk drop when this feature alone is reset to typical (from the patient's foot)
      * add:    risk rise when this feature alone is set to the patient's value (on a typical foot)
    The "add" side keeps explanations meaningful when the model is saturated near 100.
    """
    typical = {c: ref["low_median"].get(c) for c in x.columns}
    cols = [c for c in x.columns if c != "side_left" and _isnum(x.iloc[0][c]) and _isnum(typical[c])]
    if not cols:
        return []
    baseline = x.copy()
    for c in cols:
        baseline.iat[0, baseline.columns.get_loc(c)] = typical[c]

    removed = pd.concat([x] * len(cols), ignore_index=True)
    added = pd.concat([baseline] * len(cols), ignore_index=True)
    for i, c in enumerate(cols):
        j = x.columns.get_loc(c)
        removed.iat[i, j] = typical[c]
        added.iat[i, j] = x.iat[0, j]
    probe = pd.concat([baseline, removed, added], ignore_index=True)
    risks = _risk(_proba_full(model, probe, classes), classes)
    base_low, r_removed, r_added = risks[0], risks[1:1 + len(cols)], risks[1 + len(cols):]

    out = []
    for i, c in enumerate(cols):
        impact = 0.5 * ((base_risk - r_removed[i]) + (r_added[i] - base_low))
        if impact > 0.5:
            out.append({"feature": c, "label": feature_label(c), "value": _num(x.iloc[0][c]),
                        "typical": _num(ref["low_median"].get(c)), "impact": round(float(impact), 1)})
    out.sort(key=lambda d: -d["impact"])
    return out[:top]


def _proba_full(model, X: pd.DataFrame, classes: list[str]) -> np.ndarray:
    p = model.predict_proba(X)
    out = np.zeros((len(X), len(classes)))
    for j, c in enumerate(model.classes_):
        out[:, classes.index(c)] = p[:, j]
    return out


def _ood_score(x: pd.Series, ref: dict) -> int:
    n = 0
    for c, v in x.items():
        sd, mu = ref["std"].get(c), ref["mean"].get(c)
        if v is None or sd is None or mu is None or not sd or (isinstance(v, float) and math.isnan(v)):
            continue
        if abs(v - mu) / sd > 6:
            n += 1
    return n


def predict(meta: dict, left_csv: str, right_csv: str) -> dict:
    art = load_active()
    if art["feature_version"] != FEATURE_VERSION:
        raise NoModelError("Active model was trained with an older feature set - retrain it from the admin page.")
    left, right = parse_sensor_csv(left_csv), parse_sensor_csv(right_csv)
    feats = session_features(meta, left, right)
    model, classes, ref = art["model"], art["classes"], art["reference"]
    abs_dt = feats["abs_delta_t"]
    weight = meta.get("weight_kg") or 70.0

    feet = {}
    flags = []
    warnings = []
    for side, log in (("left", left), ("right", right)):
        x = pd.DataFrame([feats[side]]).reindex(columns=art["features"]).astype(float)
        proba_c = _proba_full(model, x, classes)[0]
        proba = {c: 0.0 for c in LABELS}
        proba.update({c: float(p) for c, p in zip(classes, proba_c)})
        risk = float(_risk(proba_c[None, :], classes)[0])
        level = max(proba, key=proba.get)
        f = feats[side]
        rules = rule_scores(log.df, weight, abs_dt)
        feet[side] = {
            "riskLevel": level.upper(),
            "dfuScore": int(round(risk)),
            "probabilities": {k: round(v, 4) for k, v in proba.items()},
            "confidence": round(max(proba.values()), 4),
            "drivers": _drivers(model, x, classes, ref, risk),
            "pressures": {z: int(round(f[f"p_{z}_mean"])) for z in ZONES},
            "peakPressures": {z: int(round(f[f"p_{z}_p95"])) for z in ZONES},
            "temp": f"{f['t_mean']:.1f}",
            "humidity": int(round(f["rh_mean"])),
            "cadence": _num(f["gait_cycles_per_min"], 1),
            "scores": {k: f"{rules[k]:.1f}" for k in ("PRS", "TRS", "MRS", "TBI")},
            "quality": log.quality,
        }
        warnings += [f"{side.capitalize()} log: {w}" for w in log.quality.get("warnings", [])]
        if _ood_score(x.iloc[0], ref) >= 6:
            warnings.append(f"{side.capitalize()} foot recording is unlike the model's training data - "
                            "treat this prediction with extra caution.")
        zone = max(("toe", "met"), key=lambda z: f[f"p_{z}_p95"])
        if f[f"p_{zone}_p95"] > HIGH_PRESSURE_KPA:
            flags.append({"code": "HIGH_PEAK_PRESSURE", "side": side, "severity": "moderate",
                          "message": f"{side.capitalize()} foot: forefoot peak pressure {f[f'p_{zone}_p95']:.0f} kPa at the "
                                     f"{ZONE_NAMES[zone]} exceeds the 200 kPa in-shoe offloading target."})
        if f["rh_frac_over_75"] > 0.5:
            flags.append({"code": "HIGH_HUMIDITY", "side": side, "severity": "low",
                          "message": f"{side.capitalize()} foot: in-shoe humidity above 75% for "
                                     f"{f['rh_frac_over_75'] * 100:.0f}% of the recording (maceration risk)."})

    if abs_dt > 2.2:
        warmer = "left" if feats["left"]["t_mean"] > feats["right"]["t_mean"] else "right"
        flags.insert(0, {"code": "DELTA_T", "side": warmer, "severity": "high",
                         "message": f"Left/right temperature difference of {abs_dt:.1f} °C (> 2.2 °C) with the {warmer} "
                                    "foot warmer - an established early sign of inflammation. If it persists on two "
                                    "consecutive days, reduce activity and consult a clinician."})

    worst = max((feet["left"]["riskLevel"].lower(), feet["right"]["riskLevel"].lower()), key=LABELS.index)
    reg = art["registry"]
    return {
        "overallRisk": worst,
        "overallStatus": OVERALL_STATUS[worst],
        "abs_deltaT": f"{abs_dt:.2f}",
        "left": feet["left"],
        "right": feet["right"],
        "clinicalFlags": flags,
        "warnings": warnings,
        "model": {
            "version": reg["version"],
            "algorithm": reg["algorithm"],
            "dataSource": reg["data_source"],
            "isBootstrap": reg["data_source"] == "synthetic-bootstrap",
            "trainedOn": reg["n_samples"],
            "createdAt": reg["created_at"],
            "metrics": reg["metrics"],
        },
    }
