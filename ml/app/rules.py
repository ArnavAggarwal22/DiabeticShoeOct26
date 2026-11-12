"""Legacy rule-based scores (ported from the original Next.js /api/analyze route).

They are no longer the decision-maker. They are used in two ways:
  1. As *knowledge-guided features* fed to the ML model (PRS / TRS / MRS / TBI).
  2. As the labelling teacher for the synthetic bootstrap dataset, so the app has a working
     model before any real clinical data has been collected.

Difference from the original: EWMA scores use the mean of the EWMA trajectory over the whole
recording instead of only its final value, so the result is not dominated by the last second.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter

LAMBDA = 0.2
MU_T, SIGMA_T = 31.0, 1.5
MU_RH, SIGMA_RH = 60.0, 15.0


def _ewma_mean(z: np.ndarray) -> float:
    """Mean of the EWMA trajectory s_i = λ·z_i + (1-λ)·s_{i-1}, s_{-1} = 0 (vectorised IIR filter)."""
    if len(z) == 0:
        return 0.0
    return float(lfilter([LAMBDA], [1.0, -(1.0 - LAMBDA)], np.asarray(z, dtype=float)).mean())


def rule_scores(df: pd.DataFrame, weight: float, abs_delta_t: float) -> dict[str, float]:
    w = weight if weight and weight > 0 else 70.0
    mu_p = {"toe": w * 1.5, "met": w * 3.5, "heel": w * 3.5, "mid": w * 1.0}
    sigma_p = w * 1.0

    z = lambda x, mu, sd: np.maximum(0.0, (x - mu) / sd)  # noqa: E731
    ewma_p = sum(_ewma_mean(z(df[k].to_numpy(), mu_p[k], sigma_p)) for k in mu_p)
    ewma_t = _ewma_mean(z(df["temp"].to_numpy(), MU_T, SIGMA_T))
    ewma_rh = _ewma_mean(z(df["rh"].to_numpy(), MU_RH, SIGMA_RH))

    mean_p = df[["toe", "met", "heel", "mid"]].mean(axis=1).to_numpy()
    duty_p = float((mean_p > w * 2.5).mean())
    duty_rh = float((df["rh"].to_numpy() > 75).mean())
    z_dt = max(0.0, (abs_delta_t - 1.0) / 1.0)

    prs = min(100.0, ewma_p * 12 + duty_p * 40)
    trs = min(100.0, z_dt * 40 + ewma_t * 20)
    mrs = min(100.0, ewma_rh * 25 + duty_rh * 50)
    tbi = min(100.0, prs * 0.4 + trs * 0.3 + mrs * 0.3)
    dfu = 0.4 * prs + 0.3 * trs + 0.2 * mrs + 0.1 * tbi
    return {"PRS": prs, "TRS": trs, "MRS": mrs, "TBI": tbi, "DFU": dfu}


def clinical_risk_points(meta: dict) -> float:
    """IWGDF-inspired points for clinical risk factors (used only for synthetic labelling)."""
    pts = 0.0
    if meta.get("neuropathy"):
        pts += 12
    if meta.get("neuropathy") and meta.get("deformity"):
        pts += 6
    if meta.get("pad"):
        pts += 10
    if meta.get("prior_ulcer"):
        pts += 18
    if meta.get("prior_amputation"):
        pts += 18
    hba1c = meta.get("hba1c")
    if hba1c is not None and not np.isnan(hba1c) and hba1c > 9:
        pts += 5
    return pts
