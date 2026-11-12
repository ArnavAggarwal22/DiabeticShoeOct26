"""Feature engineering: raw bilateral sensor logs + patient profile -> one feature row per foot.

Each session (left log + right log + profile) produces two training rows, one per foot. Every
row also carries *contralateral* features (this foot vs. the other foot), because asymmetry,
especially the left/right temperature difference, is the best-documented early DFU signal.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .parsing import ParsedLog
from .rules import rule_scores
from .schema import ZONES

FEATURE_VERSION = 1
HIGH_PRESSURE_KPA = 200.0  # IWGDF offloading target: peak plantar pressure < 200 kPa


def _f(x) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def _slope_per_min(t: np.ndarray, y: np.ndarray) -> float:
    if len(t) < 3 or np.ptp(t) <= 0:
        return 0.0
    return float(np.polyfit(t, y, 1)[0] * 60.0)


def _loading_cycles(total: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """Return (loading cycles per minute, stance fraction) from total plantar load."""
    lo, hi = np.percentile(total, 10), np.percentile(total, 90)
    if hi - lo < 0.15 * max(hi, 1e-6):           # essentially static (standing / sitting)
        return 0.0, float((total > 0.1 * max(hi, 1e-6)).mean())
    up, down = lo + 0.6 * (hi - lo), lo + 0.4 * (hi - lo)
    loaded, cycles = total[0] > up, 0
    for v in total[1:]:
        if not loaded and v > up:
            loaded, cycles = True, cycles + 1
        elif loaded and v < down:
            loaded = False
    minutes = max(np.ptp(t) / 60.0, 1e-6)
    return cycles / minutes, float((total > lo + 0.5 * (hi - lo)).mean())


def foot_features(df: pd.DataFrame, weight: float) -> dict[str, float]:
    w = weight if weight and weight > 0 else 70.0
    t = df["t"].to_numpy()
    feats: dict[str, float] = {}

    means = {}
    for z in ZONES:
        x = df[z].to_numpy()
        m = float(x.mean())
        means[z] = m
        feats[f"p_{z}_mean"] = m
        feats[f"p_{z}_p95"] = float(np.percentile(x, 95))
        feats[f"p_{z}_max"] = float(x.max())
        feats[f"p_{z}_std"] = float(x.std())
        feats[f"p_{z}_cv"] = float(x.std() / m) if m > 1e-6 else 0.0
        feats[f"p_{z}_mean_per_kg"] = m / w
        feats[f"p_{z}_frac_over_200"] = float((x > HIGH_PRESSURE_KPA).mean())

    total_mean = sum(means.values())
    for z in ZONES:
        feats[f"p_{z}_share"] = means[z] / total_mean if total_mean > 1e-6 else 0.25

    p95s = [feats[f"p_{z}_p95"] for z in ZONES]
    feats["p_peak_p95"] = max(p95s)
    feats["p_peak_p95_per_kg"] = max(p95s) / w
    feats["p_forefoot_heel_ratio"] = (means["toe"] + means["met"]) / max(means["heel"], 1.0)
    feats["p_total_mean_per_kg"] = total_mean / w
    feats["p_sustained_frac"] = float((df[ZONES].mean(axis=1) > 2.5 * w).mean())
    # Pressure-time integral (kPa*s per minute of recording) over the peak zone.
    peak_zone = ZONES[int(np.argmax(p95s))]
    feats["p_peak_zone_pti_per_min"] = float(np.trapezoid(df[peak_zone].to_numpy(), t) / max(np.ptp(t) / 60, 1e-6)) \
        if len(t) > 1 else means[peak_zone] * 60

    cadence, stance = _loading_cycles(df[ZONES].sum(axis=1).to_numpy(), t)
    feats["gait_cycles_per_min"] = cadence
    feats["gait_stance_frac"] = stance

    temp = df["temp"].to_numpy()
    feats["t_mean"] = float(temp.mean())
    feats["t_max"] = float(temp.max())
    feats["t_std"] = float(temp.std())
    feats["t_range"] = float(np.ptp(temp))
    feats["t_slope_per_min"] = _slope_per_min(t, temp)

    rh = df["rh"].to_numpy()
    feats["rh_mean"] = float(rh.mean())
    feats["rh_max"] = float(rh.max())
    feats["rh_frac_over_75"] = float((rh > 75).mean())
    feats["rh_slope_per_min"] = _slope_per_min(t, rh)

    feats["rec_duration_s"] = float(np.ptp(t))
    return feats


def _resample(df: pd.DataFrame, grid: np.ndarray, col: str) -> np.ndarray:
    return np.interp(grid, df["t"].to_numpy(), df[col].to_numpy())


def bilateral_features(this: pd.DataFrame, other: pd.DataFrame, f_this: dict, f_other: dict) -> dict[str, float]:
    feats: dict[str, float] = {}
    feats["bi_dT_mean"] = f_this["t_mean"] - f_other["t_mean"]          # >0 : this foot is warmer
    feats["bi_abs_dT_mean"] = abs(feats["bi_dT_mean"])

    span = min(np.ptp(this["t"]), np.ptp(other["t"]))
    if span >= 2:
        grid = np.arange(0, span, 1.0)
        diff = pd.Series(_resample(this, grid, "temp") - _resample(other, grid, "temp"))
        roll = diff.rolling(min(10, len(diff)), min_periods=1).mean()
        feats["bi_dT_max_rolling"] = float(roll.max())
        feats["bi_dT_frac_over_2_2"] = float((roll > 2.2).mean())
    else:
        feats["bi_dT_max_rolling"] = feats["bi_dT_mean"]
        feats["bi_dT_frac_over_2_2"] = float(feats["bi_dT_mean"] > 2.2)

    for z in ZONES:
        a, b = f_this[f"p_{z}_mean"], f_other[f"p_{z}_mean"]
        feats[f"bi_p_{z}_asym"] = (a - b) / (a + b) if (a + b) > 1e-6 else 0.0
    a, b = f_this["p_peak_p95"], f_other["p_peak_p95"]
    feats["bi_p_peak_asym"] = (a - b) / (a + b) if (a + b) > 1e-6 else 0.0
    feats["bi_rh_diff"] = f_this["rh_mean"] - f_other["rh_mean"]
    return feats


def patient_features(meta: dict) -> dict[str, float]:
    sex = str(meta.get("sex", "")).strip().lower()
    w, h = _f(meta.get("weight_kg")), _f(meta.get("height_cm"))
    return {
        "pt_age": _f(meta.get("age")),
        "pt_sex_male": 1.0 if sex in {"m", "male"} else 0.0 if sex in {"f", "female"} else float("nan"),
        "pt_weight_kg": w,
        "pt_bmi": w / (h / 100) ** 2 if math.isfinite(w) and math.isfinite(h) and h > 0 else float("nan"),
        "pt_diabetes_years": _f(meta.get("diabetes_years")),
        "pt_hba1c": _f(meta.get("hba1c")),
        "pt_neuropathy": _f(meta.get("neuropathy")),
        "pt_pad": _f(meta.get("pad")),
        "pt_deformity": _f(meta.get("deformity")),
        "pt_prior_ulcer": _f(meta.get("prior_ulcer")),
        "pt_prior_amputation": _f(meta.get("prior_amputation")),
    }


def session_features(meta: dict, left: ParsedLog, right: ParsedLog) -> dict[str, dict]:
    """Return {"left": features, "right": features, "summary": display values}."""
    weight = _f(meta.get("weight_kg"))
    weight = weight if math.isfinite(weight) else 70.0
    fl, fr = foot_features(left.df, weight), foot_features(right.df, weight)
    abs_dt = abs(fl["t_mean"] - fr["t_mean"])
    pt = patient_features(meta)

    out = {}
    for side, df, f_this, other_df, f_other in (
        ("left", left.df, fl, right.df, fr),
        ("right", right.df, fr, left.df, fl),
    ):
        rules = rule_scores(df, weight, abs_dt)
        row = {**f_this, **bilateral_features(df, other_df, f_this, f_other), **pt}
        row.update({f"rule_{k.lower()}": v for k, v in rules.items()})
        row["side_left"] = 1.0 if side == "left" else 0.0
        out[side] = {k: _f(v) for k, v in row.items()}
    out["abs_delta_t"] = abs_dt
    return out


def feature_names() -> list[str]:
    """Canonical, ordered feature list (derived from a tiny dummy session)."""
    t = np.arange(30) / 10.0
    df = pd.DataFrame({"t": t, "toe": 100.0 + t, "met": 200.0, "heel": 200.0, "mid": 60.0, "temp": 30.0, "rh": 50.0})
    log = ParsedLog(df=df, quality={})
    return list(session_features({"weight_kg": 70}, log, log)["left"].keys())
