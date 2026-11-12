"""SQLite persistence: labelled sessions (raw logs + features), model registry, prediction log."""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from . import config

_lock = threading.RLock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    recorded_on TEXT,
    meta TEXT NOT NULL,
    left_label TEXT,
    right_label TEXT,
    notes TEXT,
    left_csv TEXT NOT NULL,
    right_csv TEXT NOT NULL,
    left_name TEXT,
    right_name TEXT,
    quality TEXT,
    features TEXT,
    feature_version INTEGER,
    include INTEGER NOT NULL DEFAULT 1,
    source TEXT NOT NULL DEFAULT 'admin'
);
CREATE INDEX IF NOT EXISTS idx_sessions_patient ON sessions(patient_id);

CREATE TABLE IF NOT EXISTS models (
    version TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    data_source TEXT NOT NULL,
    n_samples INTEGER NOT NULL,
    metrics TEXT NOT NULL,
    report TEXT NOT NULL,
    path TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    model_version TEXT,
    meta TEXT NOT NULL,
    left_csv TEXT NOT NULL,
    right_csv TEXT NOT NULL,
    result TEXT NOT NULL,
    promoted_session_id TEXT
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect():
    with _lock:
        conn = sqlite3.connect(config.DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def init_db() -> None:
    with connect() as c:
        c.executescript(SCHEMA)


def _row(r: sqlite3.Row | None, json_cols=("meta", "quality", "features", "metrics", "report", "result")) -> dict | None:
    if r is None:
        return None
    d = dict(r)
    for k in json_cols:
        if k in d and d[k] is not None:
            d[k] = json.loads(d[k])
    if "include" in d:
        d["include"] = bool(d["include"])
    if "is_active" in d:
        d["is_active"] = bool(d["is_active"])
    return d


# --- sessions ------------------------------------------------------------------------------

def insert_session(*, patient_id, meta, left_label, right_label, left_csv, right_csv, quality, features,
                   feature_version, notes=None, recorded_on=None, left_name=None, right_name=None,
                   source="admin") -> str:
    sid = uuid.uuid4().hex[:12]
    with connect() as c:
        c.execute(
            """INSERT INTO sessions (id, patient_id, created_at, recorded_on, meta, left_label, right_label, notes,
               left_csv, right_csv, left_name, right_name, quality, features, feature_version, source)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (sid, patient_id, now(), recorded_on, json.dumps(meta), left_label, right_label, notes, left_csv,
             right_csv, left_name, right_name, json.dumps(quality), json.dumps(features), feature_version, source),
        )
    return sid


LIST_COLS = ("id, patient_id, created_at, recorded_on, meta, left_label, right_label, notes, left_name, right_name, "
             "quality, include, source, feature_version")


def list_sessions() -> list[dict]:
    with connect() as c:
        rows = c.execute(f"SELECT {LIST_COLS} FROM sessions ORDER BY created_at DESC").fetchall()
    return [_row(r) for r in rows]


def get_session(sid: str, raw: bool = False) -> dict | None:
    cols = "*" if raw else LIST_COLS + ", features"
    with connect() as c:
        return _row(c.execute(f"SELECT {cols} FROM sessions WHERE id = ?", (sid,)).fetchone())


def all_sessions_for_training() -> list[dict]:
    with connect() as c:
        rows = c.execute("SELECT * FROM sessions WHERE include = 1").fetchall()
    return [_row(r) for r in rows]


def update_session(sid: str, fields: dict) -> None:
    allowed = {"patient_id", "meta", "left_label", "right_label", "notes", "include", "recorded_on",
               "features", "feature_version", "quality"}
    sets, vals = [], []
    for k, v in fields.items():
        if k not in allowed:
            continue
        if k in ("meta", "features", "quality"):
            v = json.dumps(v)
        if k == "include":
            v = 1 if v else 0
        sets.append(f"{k} = ?")
        vals.append(v)
    if not sets:
        return
    with connect() as c:
        c.execute(f"UPDATE sessions SET {', '.join(sets)} WHERE id = ?", (*vals, sid))


def delete_session(sid: str) -> bool:
    with connect() as c:
        return c.execute("DELETE FROM sessions WHERE id = ?", (sid,)).rowcount > 0


# --- models --------------------------------------------------------------------------------

def insert_model(*, version, algorithm, data_source, n_samples, metrics, report, path, activate) -> None:
    with connect() as c:
        if activate:
            c.execute("UPDATE models SET is_active = 0")
        c.execute(
            "INSERT INTO models (version, created_at, algorithm, data_source, n_samples, metrics, report, path, is_active)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (version, now(), algorithm, data_source, n_samples, json.dumps(metrics), json.dumps(report), path,
             1 if activate else 0),
        )


def list_models() -> list[dict]:
    with connect() as c:
        rows = c.execute("SELECT version, created_at, algorithm, data_source, n_samples, metrics, is_active "
                         "FROM models ORDER BY created_at DESC, version DESC").fetchall()
    return [_row(r) for r in rows]


def get_model(version: str) -> dict | None:
    with connect() as c:
        return _row(c.execute("SELECT * FROM models WHERE version = ?", (version,)).fetchone())


def active_model() -> dict | None:
    with connect() as c:
        return _row(c.execute("SELECT * FROM models WHERE is_active = 1").fetchone())


def activate_model(version: str) -> bool:
    with connect() as c:
        if c.execute("SELECT 1 FROM models WHERE version = ?", (version,)).fetchone() is None:
            return False
        c.execute("UPDATE models SET is_active = 0")
        c.execute("UPDATE models SET is_active = 1 WHERE version = ?", (version,))
    return True


def delete_model(version: str) -> dict | None:
    m = get_model(version)
    if m:
        with connect() as c:
            c.execute("DELETE FROM models WHERE version = ?", (version,))
    return m


# --- prediction log ------------------------------------------------------------------------

def log_prediction(*, model_version, meta, left_csv, right_csv, result) -> str:
    pid = uuid.uuid4().hex[:12]
    with connect() as c:
        c.execute("INSERT INTO predictions (id, created_at, model_version, meta, left_csv, right_csv, result)"
                  " VALUES (?,?,?,?,?,?,?)",
                  (pid, now(), model_version, json.dumps(meta), left_csv, right_csv, json.dumps(result)))
    return pid


def list_predictions(limit: int = 200) -> list[dict]:
    with connect() as c:
        rows = c.execute("SELECT id, created_at, model_version, meta, result, promoted_session_id FROM predictions "
                         "ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [_row(r) for r in rows]


def get_prediction(pid: str) -> dict | None:
    with connect() as c:
        return _row(c.execute("SELECT * FROM predictions WHERE id = ?", (pid,)).fetchone())


def mark_prediction_promoted(pid: str, session_id: str) -> None:
    with connect() as c:
        c.execute("UPDATE predictions SET promoted_session_id = ? WHERE id = ?", (session_id, pid))


def delete_prediction(pid: str) -> bool:
    with connect() as c:
        return c.execute("DELETE FROM predictions WHERE id = ?", (pid,)).rowcount > 0
