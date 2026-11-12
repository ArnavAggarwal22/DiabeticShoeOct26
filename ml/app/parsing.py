"""Robust parsing + quality checks for the insole's Bluetooth CSV logs.

Expected log (one file per foot), as produced by the shoe firmware:

    Time,P1_Toe,P2_Met,P3_Heel,P4_Mid,Temp,Hum
    20:45:18.100, 0.95, 1.31, 1.42, 0.41, 30.4, 42.1

Delimiters may be commas, semicolons, tabs or spaces. Headers are optional and matched by
name when present; otherwise columns are positional. A missing time column is tolerated.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config

_SPLIT = re.compile(r"[,;\t ]+")
_CLOCK = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2}(?:\.\d+)?)$")

_HEADER_ALIASES = {
    "time": ("time", "timestamp", "ts", "t_ms", "ms", "sec"),
    "toe": ("p1", "toe", "hallux"),
    "met": ("p2", "met", "metatarsal", "forefoot", "mth"),
    "heel": ("p3", "heel", "calcaneus", "rearfoot"),
    "mid": ("p4", "mid", "midfoot", "arch"),
    "temp": ("temp", "temperature", "tmp", "celsius"),
    "rh": ("hum", "humidity", "rh", "moisture"),
}


class SensorParseError(ValueError):
    pass


@dataclass
class ParsedLog:
    df: pd.DataFrame                    # columns: t (s), toe, met, heel, mid (kPa), temp (°C), rh (%)
    quality: dict = field(default_factory=dict)


def _parse_time(token: str) -> float | None:
    m = _CLOCK.match(token)
    if m:
        h, mi, s = m.groups()
        return int(h) * 3600 + int(mi) * 60 + float(s)
    if "T" in token or "-" in token[1:]:
        try:
            ts = pd.Timestamp(token)
            return ts.hour * 3600 + ts.minute * 60 + ts.second + ts.microsecond / 1e6
        except (ValueError, TypeError):
            return None
    try:
        return float(token)
    except ValueError:
        return None


def _map_header(tokens: list[str]) -> dict[str, int] | None:
    mapping: dict[str, int] = {}
    for idx, tok in enumerate(tokens):
        low = tok.lower()
        for col, aliases in _HEADER_ALIASES.items():
            if col in mapping:
                continue
            if any(low == a or low.startswith(a + "_") or low.endswith("_" + a) or a in low.split("_") for a in aliases):
                mapping[col] = idx
                break
    needed = {"toe", "met", "heel", "mid", "temp", "rh"}
    return mapping if needed.issubset(mapping) else None


def _looks_like_header(tokens: list[str]) -> bool:
    return any(re.search(r"[a-zA-Z]", t) and _parse_time(t) is None for t in tokens)


def parse_sensor_csv(text: str) -> ParsedLog:
    if not text or not text.strip():
        raise SensorParseError("File is empty.")

    lines = [ln.strip() for ln in text.replace("\r", "\n").split("\n")]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    if not lines:
        raise SensorParseError("File has no data rows.")

    mapping: dict[str, int] | None = None
    first = [t for t in _SPLIT.split(lines[0]) if t]
    if _looks_like_header(first):
        mapping = _map_header(first)
        lines = lines[1:]

    rows: list[list[float]] = []
    skipped = 0
    has_time = True
    for ln in lines:
        tokens = [t for t in _SPLIT.split(ln) if t]
        if mapping is not None:
            try:
                vals = [float(tokens[mapping[c]]) for c in ("toe", "met", "heel", "mid", "temp", "rh")]
            except (IndexError, ValueError):
                skipped += 1
                continue
            t = _parse_time(tokens[mapping["time"]]) if "time" in mapping and mapping["time"] < len(tokens) else None
            if "time" not in mapping:
                has_time = False
        elif len(tokens) >= 7:
            try:
                vals = [float(x) for x in tokens[1:7]]
            except ValueError:
                skipped += 1
                continue
            t = _parse_time(tokens[0])
        elif len(tokens) == 6:
            try:
                vals = [float(x) for x in tokens]
            except ValueError:
                skipped += 1
                continue
            t, has_time = None, False
        else:
            skipped += 1
            continue
        rows.append([np.nan if t is None else t, *vals])

    if len(rows) < 10:
        raise SensorParseError(
            f"Only {len(rows)} valid rows found (need at least 10). Expected columns: "
            "Time, P1_Toe, P2_Met, P3_Heel, P4_Mid, Temp, Hum."
        )

    df = pd.DataFrame(rows, columns=["t", "toe", "met", "heel", "mid", "temp", "rh"])
    df[["toe", "met", "heel", "mid"]] *= config.PRESSURE_SCALE

    # --- time axis -------------------------------------------------------------------------
    t = df["t"].to_numpy(dtype=float)
    time_ok = has_time and np.isfinite(t).mean() > 0.9
    if time_ok:
        t = pd.Series(t).interpolate(limit_direction="both").to_numpy()
        t = np.where(np.diff(t, prepend=t[0]) < -43000, t + 86400, t)  # midnight rollover
        t = np.maximum.accumulate(t)
        # Values in milliseconds (e.g. 0, 100, 200 ...) -> seconds.
        dt_med = np.median(np.diff(t)) if len(t) > 1 else 0
        if dt_med >= 20:
            t = t / 1000.0
        t = t - t[0]
        dt = np.diff(t)
        if len(dt) == 0 or np.median(dt) <= 0:
            time_ok = False
    if not time_ok:
        t = np.arange(len(df)) / config.DEFAULT_SAMPLING_HZ
    df["t"] = t

    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    hz = (len(df) - 1) / duration if duration > 0 else config.DEFAULT_SAMPLING_HZ

    # --- quality report --------------------------------------------------------------------
    warnings: list[str] = []
    if skipped:
        warnings.append(f"{skipped} malformed row(s) skipped.")
    if not time_ok:
        warnings.append(f"No usable time column; assumed {config.DEFAULT_SAMPLING_HZ:g} Hz sampling.")
    if duration < 10:
        warnings.append(f"Recording is short ({duration:.1f}s). 60s+ of walking/standing gives more reliable features.")

    for zone in ("toe", "met", "heel", "mid"):
        col = df[zone]
        if (col < 0).any():
            warnings.append(f"{zone}: {(col < 0).sum()} negative pressure value(s) clipped to 0.")
        if col.std() < 1e-6 and len(col) > 20:
            warnings.append(f"{zone}: sensor is flat-lined (constant value) - check the sensor.")
    df[["toe", "met", "heel", "mid"]] = df[["toe", "met", "heel", "mid"]].clip(lower=0)

    bad_temp = ((df["temp"] < 15) | (df["temp"] > 45)).sum()
    if bad_temp:
        warnings.append(f"temperature: {bad_temp} value(s) outside 15-45 °C (implausible skin temperature).")
    bad_rh = ((df["rh"] < 0) | (df["rh"] > 100)).sum()
    if bad_rh:
        warnings.append(f"humidity: {bad_rh} value(s) outside 0-100 % clipped.")
    df["rh"] = df["rh"].clip(0, 100)

    quality = {
        "rows": int(len(df)),
        "skipped_rows": int(skipped),
        "duration_s": round(duration, 2),
        "sampling_hz": round(float(hz), 2),
        "has_time_column": bool(time_ok),
        "warnings": warnings,
    }
    return ParsedLog(df=df, quality=quality)
