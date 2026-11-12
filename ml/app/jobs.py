"""Background training jobs.

Training takes ~1-2 minutes, which is longer than hosting platforms allow for a start-up
health check or a proxied HTTP request. So training always runs in a background thread:
callers start a job, then poll its status.
"""
from __future__ import annotations

import logging
import threading
import traceback
import uuid
from datetime import datetime, timezone

from . import inference, training

log = logging.getLogger("smartsole")

_lock = threading.Lock()
_state: dict = {"status": "idle"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def status() -> dict:
    with _lock:
        return dict(_state)


def is_running() -> bool:
    return status().get("status") == "running"


def _run(job_id: str, kind: str, params: dict) -> None:
    try:
        report = training.train(**params)
        inference.invalidate_cache()
        result = {"status": "succeeded", "report": report}
        log.info("Training job %s (%s) finished: model %s, CV macro-F1 %.3f", job_id, kind, report["version"],
                 report["candidates"][0]["cv_metrics"]["macro_f1"])
    except training.TrainingError as e:
        result = {"status": "failed", "error": str(e)}
        log.warning("Training job %s (%s) failed: %s", job_id, kind, e)
    except Exception as e:  # noqa: BLE001 - surface any crash to the admin page instead of losing it
        result = {"status": "failed", "error": f"Unexpected error: {e}"}
        log.error("Training job %s (%s) crashed:\n%s", job_id, kind, traceback.format_exc())
    with _lock:
        if _state.get("job_id") == job_id:
            _state.update(result, finished_at=_now())


def start(kind: str, **params) -> dict:
    """Start a training job. Raises training.TrainingError if one is already running."""
    with _lock:
        if _state.get("status") == "running":
            raise training.TrainingError("A training run is already in progress.")
        job_id = uuid.uuid4().hex[:8]
        _state.clear()
        _state.update(job_id=job_id, kind=kind, status="running", params=params, started_at=_now())
        snapshot = dict(_state)
    threading.Thread(target=_run, args=(job_id, kind, params), name=f"train-{job_id}", daemon=True).start()
    return snapshot


def wait(timeout: float = 600) -> dict:
    """Block until the current job finishes (used by tests and CLI tools)."""
    import time

    deadline = time.monotonic() + timeout
    while is_running() and time.monotonic() < deadline:
        time.sleep(0.25)
    return status()
