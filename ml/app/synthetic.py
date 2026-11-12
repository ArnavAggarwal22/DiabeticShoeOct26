"""Synthetic bootstrap data.

Until real, clinically-labelled recordings are uploaded on /admin, the service needs *some*
model. We simulate plausible bilateral insole recordings (walking or standing, with
weight-scaled plantar loads, inflammatory hot-spots, sweat build-up) for a population of
virtual patients and label each foot with the legacy rule engine + IWGDF clinical risk points.

The resulting "bootstrap" model therefore reproduces expert rules, not real outcomes. It is
clearly flagged as such in the API and UI and is replaced as soon as a model is trained on
real data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .features import session_features
from .parsing import ParsedLog
from .rules import clinical_risk_points, rule_scores

# Typical healthy in-shoe load per zone, kPa per kg body weight (quiet standing level).
# Walking peaks are ~1.45x this, giving ~130-200 kPa forefoot peaks for a 70 kg adult.
HEALTHY_LOAD = {"toe": 1.3, "met": 1.9, "heel": 2.0, "mid": 0.6}
WALK_PEAK = 1.45


def _gait_shape(phase: np.ndarray, zone: str) -> np.ndarray:
    """Normalised loading profile over a gait cycle (stance = first 60%)."""
    stance = phase < 0.6
    s = np.clip(phase / 0.6, 0, 1)
    centre = {"heel": 0.15, "mid": 0.45, "met": 0.75, "toe": 0.9}[zone]
    width = {"heel": 0.16, "mid": 0.22, "met": 0.16, "toe": 0.12}[zone]
    return np.where(stance, np.exp(-0.5 * ((s - centre) / width) ** 2), 0.0)


def _simulate_foot(rng: np.random.Generator, n: int, fs: float, weight: float, walking: bool,
                   severity: float, hot_zone: str, base_temp: float, base_rh: float, meta: dict) -> pd.DataFrame:
    t = np.arange(n) / fs
    data = {"t": t}
    cadence = rng.uniform(0.85, 1.25)  # seconds per stride
    phase = ((t + rng.uniform(0, cadence)) % cadence) / cadence
    for zone, k in HEALTHY_LOAD.items():
        mean = k * weight * rng.uniform(0.75, 1.25)
        if meta.get("neuropathy") and zone in ("met", "toe"):
            mean *= rng.uniform(1.0, 1.35)
        if meta.get("deformity") and zone in ("met", "toe"):
            mean *= rng.uniform(1.05, 1.4)
        if zone == hot_zone:
            mean *= 1 + severity * rng.uniform(0.8, 2.4)
        if walking:
            sig = mean * WALK_PEAK * _gait_shape(phase, zone)
        else:
            sig = mean * (1 + 0.04 * np.sin(2 * np.pi * t / rng.uniform(4, 9)))
        sig = sig + rng.normal(0, 0.04 * mean + 2, n)
        data[zone] = np.clip(sig, 0, None)

    inflam = severity * rng.uniform(0.5, 3.8)
    drift = rng.normal(0, 0.15) * t / 60
    data["temp"] = base_temp + inflam + drift + rng.normal(0, 0.12, n)
    sweat = severity * rng.uniform(0, 35)
    data["rh"] = np.clip(base_rh + sweat + rng.normal(0, 1.5, n) + 2 * t / max(t[-1], 1), 0, 100)
    return pd.DataFrame(data)


def _patient(rng: np.random.Generator) -> tuple[dict, float]:
    r = float(rng.beta(1.3, 2.4))  # latent risk
    sex = "Male" if rng.random() < 0.55 else "Female"
    height = rng.normal(172 if sex == "Male" else 160, 7)
    meta = {
        "age": int(np.clip(rng.normal(58 + 10 * r, 11), 25, 90)),
        "sex": sex,
        "weight_kg": round(float(np.clip(rng.normal(74 + 12 * r, 13), 40, 150)), 1),
        "height_cm": round(float(height), 1) if rng.random() < 0.7 else None,
        "diabetes_years": round(float(np.clip(rng.normal(6 + 14 * r, 5), 0, 50)), 1) if rng.random() < 0.8 else None,
        "hba1c": round(float(np.clip(rng.normal(7 + 2.5 * r, 1.1), 5, 14)), 1) if rng.random() < 0.7 else None,
        "neuropathy": bool(rng.random() < 0.1 + 0.7 * r),
        "pad": bool(rng.random() < 0.05 + 0.35 * r),
        "deformity": bool(rng.random() < 0.1 + 0.45 * r),
        "prior_ulcer": bool(rng.random() < 0.02 + 0.45 * r ** 2),
        "prior_amputation": bool(rng.random() < 0.01 + 0.12 * r ** 3),
    }
    return meta, r


def _label(score: float) -> str:
    if score >= 30:
        return "high"
    if score >= 17:
        return "moderate"
    return "low"


def generate_sessions(n_patients: int, seed: int = 42) -> list[dict]:
    """Return session dicts: {patient_id, meta, left_df, right_df, left_label, right_label}."""
    rng = np.random.default_rng(seed)
    sessions = []
    for i in range(n_patients):
        meta, r = _patient(rng)
        walking = rng.random() < 0.65
        fs = 10.0
        n = int(rng.integers(250, 700))
        base_temp = rng.normal(30.2, 1.0)
        base_rh = rng.normal(44, 7)
        affected = "left" if rng.random() < 0.5 else "right"
        sev = {affected: float(np.clip(r * rng.uniform(0.5, 1.4), 0, 1.3)),
               ("right" if affected == "left" else "left"): float(r * rng.uniform(0, 0.35))}
        hot = rng.choice(["met", "toe", "heel", "met"])
        dfs = {side: _simulate_foot(rng, n, fs, meta["weight_kg"], walking, sev[side], hot,
                                    base_temp + rng.normal(0, 0.25), base_rh, meta) for side in ("left", "right")}
        abs_dt = abs(dfs["left"]["temp"].mean() - dfs["right"]["temp"].mean())
        pts = clinical_risk_points(meta)
        labels = {}
        for side in ("left", "right"):
            dfu = rule_scores(dfs[side], meta["weight_kg"], abs_dt)["DFU"]
            warmer = dfs[side]["temp"].mean() >= dfs["right" if side == "left" else "left"]["temp"].mean()
            score = dfu + pts + (8 if (abs_dt > 2.2 and warmer) else 0) + rng.normal(0, 4)
            labels[side] = _label(score)
        sessions.append({"patient_id": f"SYN-{i:04d}", "meta": meta, "left_df": dfs["left"], "right_df": dfs["right"],
                         "left_label": labels["left"], "right_label": labels["right"]})
    return sessions


def synthetic_training_rows(n_patients: int, seed: int = 42) -> tuple[list[dict], list[str], list[str]]:
    """Feature rows, labels and group ids (patient) for the bootstrap dataset."""
    X, y, groups = [], [], []
    for s in generate_sessions(n_patients, seed):
        feats = session_features(s["meta"], ParsedLog(s["left_df"]), ParsedLog(s["right_df"]))
        for side in ("left", "right"):
            X.append(feats[side])
            y.append(s[f"{side}_label"])
            groups.append(s["patient_id"])
    return X, y, groups


def to_sensor_csv(df: pd.DataFrame, start_s: float = 20 * 3600 + 45 * 60) -> str:
    """Render a simulated foot back into the firmware CSV format (used for demo files)."""
    k = config.PRESSURE_SCALE
    lines = ["Time,P1_Toe,P2_Met,P3_Heel,P4_Mid,Temp,Hum"]
    for row in df.itertuples(index=False):
        s = start_s + row.t
        hh, mm, ss = int(s // 3600) % 24, int(s % 3600 // 60), s % 60
        lines.append(f"{hh:02d}:{mm:02d}:{ss:06.3f},{row.toe / k:.2f},{row.met / k:.2f},"
                     f"{row.heel / k:.2f},{row.mid / k:.2f},{row.temp:.1f},{row.rh:.1f}")
    return "\n".join(lines)


DEMO_SCENARIOS = {
    "healthy": {"severity": 0.0, "other": 0.0, "delta_t": 0.2, "meta": {"age": 46, "sex": "Male", "weight_kg": 64, "height_cm": 171,
                                                       "diabetes_years": 3, "hba1c": 6.8, "neuropathy": False,
                                                       "pad": False, "deformity": False, "prior_ulcer": False,
                                                       "prior_amputation": False}},
    "moderate": {"severity": 0.55, "other": 0.1, "delta_t": 1.4, "meta": {"age": 61, "sex": "Female", "weight_kg": 78,
                                                          "height_cm": 160, "diabetes_years": 12, "hba1c": 8.4,
                                                          "neuropathy": True, "pad": False, "deformity": True,
                                                          "prior_ulcer": False, "prior_amputation": False}},
    "critical": {"severity": 1.15, "other": 0.05, "delta_t": 3.1, "meta": {"age": 67, "sex": "Male", "weight_kg": 88,
                                                           "height_cm": 174, "diabetes_years": 19, "hba1c": 9.6,
                                                           "neuropathy": True, "pad": True, "deformity": True,
                                                           "prior_ulcer": True, "prior_amputation": False}},
}


def demo_session(scenario: str, seed: int = 7) -> dict:
    """Deterministic demo recording (affected foot = left) in firmware CSV format."""
    spec = DEMO_SCENARIOS[scenario]
    rng = np.random.default_rng(seed)
    meta = dict(spec["meta"])
    out = {"meta": meta}
    dfs = {side: _simulate_foot(rng, 600, 10.0, meta["weight_kg"], True, sev, "met", 30.3, 43, meta)
           for side, sev in (("left", spec["severity"]), ("right", spec["other"]))}
    # Pin the left/right temperature difference so each demo tells a consistent story.
    dfs["left"]["temp"] += spec["delta_t"] - (dfs["left"]["temp"].mean() - dfs["right"]["temp"].mean())
    for side, df in dfs.items():
        out[f"{side}_csv"] = to_sensor_csv(df)
    return out
