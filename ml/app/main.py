"""SmartSole ML service (FastAPI).

Public:
  GET  /health                      liveness + active model summary
  POST /predict                     bilateral risk prediction from raw insole logs
  GET  /demo/{scenario}             demo recordings (healthy | moderate | critical)

Admin (header X-Admin-Token: $ADMIN_TOKEN):
  GET    /admin/schema              clinical fields, labels, guides (drives the admin UI)
  GET    /admin/overview            dataset readiness, active model, model list
  POST   /admin/sessions/preview    parse + feature-extract without saving
  GET    /admin/sessions            list labelled sessions
  POST   /admin/sessions            add one labelled session (profile + 2 raw logs + labels)
  POST   /admin/sessions/bulk       add many sessions from a manifest CSV + log files
  GET    /admin/sessions/{id}       session detail incl. features
  PATCH  /admin/sessions/{id}       edit labels / profile / include flag
  DELETE /admin/sessions/{id}
  POST   /admin/features/recompute  re-run feature extraction on all stored raw logs
  GET    /admin/dataset/export      ML-ready feature matrix as CSV
  GET    /admin/templates/{name}    manifest.csv | sensor_log.csv
  POST   /admin/train               start a background job: cross-validate candidates, register best
  GET    /admin/train/status        poll the training job (running | succeeded | failed | idle)
  GET    /admin/models              model registry
  GET    /admin/models/{version}    full training report
  POST   /admin/models/{version}/activate
  DELETE /admin/models/{version}
  GET    /admin/predictions         prediction log
  POST   /admin/predictions/{id}/promote   label a past prediction and add it to the dataset
  DELETE /admin/predictions/{id}
"""
from __future__ import annotations

import io
import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, ValidationError

from . import config, inference, jobs, starter, storage, training
from .features import FEATURE_VERSION, feature_names, session_features
from .parsing import SensorParseError, parse_sensor_csv
from .schema import (CLINICAL_FIELDS, CLINICAL_KEYS, LABEL_GUIDE, LABELS, REQUIRED_KEYS, SENSOR_HEADER)
from .synthetic import DEMO_SCENARIOS, demo_session, generate_sessions, to_sensor_csv

log = logging.getLogger("smartsole")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def ensure_model() -> None:
    """Make sure a usable model is active on boot.

    1. Keep the current active model if it loads and matches the feature set.
    2. Otherwise activate the pre-trained starter bundled in the repo (instant).
    3. Only if that is missing/incompatible, train a starter model in the background; the
       server keeps accepting requests meanwhile.
    """
    try:
        art = inference.load_active()
        if art["feature_version"] == FEATURE_VERSION:
            return
        log.warning("Active model uses an old feature set.")
    except inference.NoModelError as e:
        log.info("%s", e)
    if starter.register_bundled():
        inference.invalidate_cache()
        return
    log.info("Training the synthetic starter model in the background (a few minutes)...")
    jobs.start("bootstrap", source="synthetic", activate=True)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    storage.init_db()
    if not config.ADMIN_TOKEN:
        log.warning("ADMIN_TOKEN is not set - all /admin endpoints are disabled.")
    ensure_model()
    yield


app = FastAPI(title="SmartSole DFU ML Service", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])


# --- helpers -------------------------------------------------------------------------------

_TRUE = {"1", "true", "yes", "y", "t"}
_FALSE = {"0", "false", "no", "n", "f"}


def normalise_meta(raw: dict[str, Any], strict: bool = True) -> dict:
    meta: dict[str, Any] = {}
    errors = []
    for f in CLINICAL_FIELDS:
        k = f["key"]
        v = raw.get(k)
        if isinstance(v, str):
            v = v.strip()
        if v in (None, "") or (isinstance(v, float) and pd.isna(v)):
            meta[k] = None
            if f["required"] and strict:
                errors.append(f"'{f['label']}' is required.")
            continue
        try:
            if f["kind"] == "number":
                v = float(v)
                if not (f.get("min", -1e9) <= v <= f.get("max", 1e9)):
                    errors.append(f"'{f['label']}' = {v:g} is outside {f['min']}-{f['max']} {f.get('unit', '')}.")
            elif f["kind"] == "bool":
                if isinstance(v, bool):
                    pass
                elif str(v).lower() in _TRUE:
                    v = True
                elif str(v).lower() in _FALSE:
                    v = False
                else:
                    raise ValueError
            elif f["kind"] == "sex":
                s = str(v).lower()
                v = "Male" if s in {"m", "male"} else "Female" if s in {"f", "female"} else None
                if v is None:
                    raise ValueError
        except (TypeError, ValueError):
            errors.append(f"'{f['label']}' has an invalid value: {raw.get(k)!r}.")
            v = None
        meta[k] = v
    if errors:
        raise HTTPException(422, detail=" ".join(errors))
    return meta


def normalise_label(v: Any) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip().lower()
    if s in ("", "none", "unlabelled", "unknown"):
        return None
    alias = {"0": "low", "1": "moderate", "2": "high", "medium": "moderate", "ulcer": "high", "critical": "high",
             "healthy": "low", "warning": "moderate"}
    s = alias.get(s, s)
    if s not in LABELS:
        raise HTTPException(422, detail=f"Invalid label {v!r}. Use one of: {', '.join(LABELS)}.")
    return s


def _parse_pair(left_csv: str, right_csv: str):
    try:
        return parse_sensor_csv(left_csv), parse_sensor_csv(right_csv)
    except SensorParseError as e:
        raise HTTPException(422, detail=str(e)) from e


def _features_payload(meta, left, right) -> dict:
    feats = session_features(meta, left, right)
    return {"left": training._clean(feats["left"]), "right": training._clean(feats["right"]),
            "abs_delta_t": feats["abs_delta_t"]}


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    if not config.ADMIN_TOKEN:
        raise HTTPException(503, detail="Admin is disabled: set the ADMIN_TOKEN environment variable on the ML service.")
    if not x_admin_token or not secrets.compare_digest(x_admin_token, config.ADMIN_TOKEN):
        raise HTTPException(401, detail="Invalid admin token.")


# --- public --------------------------------------------------------------------------------

class PredictRequest(BaseModel):
    meta: dict[str, Any]
    left_csv: str = Field(min_length=1)
    right_csv: str = Field(min_length=1)


@app.get("/")
def root():
    return {"service": "SmartSole DFU ML Service", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    m = storage.active_model()
    job = jobs.status()
    return {"status": "ok",
            "training": job.get("status") == "running" and job.get("kind") or None,
            "model": None if m is None else
            {"version": m["version"], "algorithm": m["algorithm"], "data_source": m["data_source"],
             "metrics": m["metrics"], "n_samples": m["n_samples"]}}


@app.post("/predict")
def predict(req: PredictRequest):
    meta = normalise_meta(req.meta)
    try:
        result = inference.predict(meta, req.left_csv, req.right_csv)
    except SensorParseError as e:
        raise HTTPException(422, detail=str(e)) from e
    except inference.NoModelError as e:
        if jobs.is_running() and jobs.status().get("kind") == "bootstrap":
            raise HTTPException(503, detail="The ML service has just started and is still training its starter "
                                            "model (about 2 minutes). Please try again shortly.") from e
        raise HTTPException(503, detail=str(e)) from e
    if config.LOG_PREDICTIONS:
        log_result = {k: result[k] for k in ("overallRisk", "overallStatus", "abs_deltaT")}
        log_result["left"] = {k: result["left"][k] for k in ("riskLevel", "dfuScore", "probabilities")}
        log_result["right"] = {k: result["right"][k] for k in ("riskLevel", "dfuScore", "probabilities")}
        result["predictionId"] = storage.log_prediction(model_version=result["model"]["version"], meta=meta,
                                                        left_csv=req.left_csv, right_csv=req.right_csv,
                                                        result=log_result)
    return result


@app.get("/demo/{scenario}")
def demo(scenario: str):
    if scenario not in DEMO_SCENARIOS:
        raise HTTPException(404, detail=f"Unknown scenario. Use: {', '.join(DEMO_SCENARIOS)}.")
    return demo_session(scenario)


# --- admin: schema / overview --------------------------------------------------------------

@app.get("/admin/schema", dependencies=[Depends(require_admin)])
def admin_schema():
    return {
        "clinical_fields": CLINICAL_FIELDS,
        "labels": LABELS,
        "label_guide": LABEL_GUIDE,
        "sensor_header": SENSOR_HEADER,
        "pressure_scale": config.PRESSURE_SCALE,
        "feature_version": FEATURE_VERSION,
        "features": [{"name": f, "label": inference.feature_label(f)} for f in feature_names()],
        "algorithms": training.ALGORITHMS,
        "minimums": {"per_class": config.MIN_SAMPLES_PER_CLASS, "patients": config.MIN_PATIENTS,
                     "recommended": config.RECOMMENDED_SAMPLES},
        "manifest_columns": ["patient_id", "recorded_on", "left_file", "right_file", "left_label", "right_label",
                             "notes", *CLINICAL_KEYS],
    }


@app.get("/admin/overview", dependencies=[Depends(require_admin)])
def admin_overview():
    active = storage.active_model()
    if active:
        active = {k: v for k, v in active.items() if k != "path"}
    return {"readiness": training.dataset_readiness(), "active_model": active, "models": storage.list_models(),
            "training_job": jobs.status(), "storage": {"ephemeral": config.EPHEMERAL_STORAGE}}


# --- admin: sessions -----------------------------------------------------------------------

class SessionIn(BaseModel):
    patient_id: str = Field(min_length=1, max_length=80)
    meta: dict[str, Any]
    left_csv: str
    right_csv: str
    left_label: str | None = None
    right_label: str | None = None
    notes: str | None = None
    recorded_on: str | None = None
    left_name: str | None = None
    right_name: str | None = None


class PreviewIn(BaseModel):
    meta: dict[str, Any] = {}
    left_csv: str
    right_csv: str


@app.post("/admin/sessions/preview", dependencies=[Depends(require_admin)])
def preview_session(body: PreviewIn):
    meta = normalise_meta(body.meta, strict=False)
    left, right = _parse_pair(body.left_csv, body.right_csv)
    feats = _features_payload(meta, left, right)
    return {"quality": {"left": left.quality, "right": right.quality}, "features": feats,
            "feature_labels": {f: inference.feature_label(f) for f in feats["left"]}}


def _store_session(body: SessionIn, source: str = "admin") -> dict:
    meta = normalise_meta(body.meta)
    ll, rl = normalise_label(body.left_label), normalise_label(body.right_label)
    left, right = _parse_pair(body.left_csv, body.right_csv)
    feats = _features_payload(meta, left, right)
    sid = storage.insert_session(
        patient_id=body.patient_id.strip(), meta=meta, left_label=ll, right_label=rl, left_csv=body.left_csv,
        right_csv=body.right_csv, quality={"left": left.quality, "right": right.quality}, features=feats,
        feature_version=FEATURE_VERSION, notes=body.notes, recorded_on=body.recorded_on,
        left_name=body.left_name, right_name=body.right_name, source=source)
    return {"id": sid, "quality": {"left": left.quality, "right": right.quality}}


@app.get("/admin/sessions", dependencies=[Depends(require_admin)])
def list_sessions():
    return storage.list_sessions()


@app.post("/admin/sessions", dependencies=[Depends(require_admin)])
def add_session(body: SessionIn):
    return _store_session(body)


class BulkIn(BaseModel):
    manifest_csv: str
    files: dict[str, str]


@app.post("/admin/sessions/bulk", dependencies=[Depends(require_admin)])
def bulk_add(body: BulkIn):
    try:
        manifest = pd.read_csv(io.StringIO(body.manifest_csv), dtype=str, keep_default_na=False)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, detail=f"Could not read manifest CSV: {e}") from e
    manifest.columns = [c.strip().lower() for c in manifest.columns]
    missing = {"patient_id", "left_file", "right_file", *REQUIRED_KEYS} - set(manifest.columns)
    if missing:
        raise HTTPException(422, detail=f"Manifest is missing column(s): {', '.join(sorted(missing))}.")
    files = {Path(k).name.lower(): v for k, v in body.files.items()}

    results = []
    for i, row in manifest.iterrows():
        line = i + 2
        try:
            lf, rf = Path(row["left_file"]).name.lower(), Path(row["right_file"]).name.lower()
            if lf not in files or rf not in files:
                raise HTTPException(422, detail=f"log file not uploaded: {', '.join(f for f in (lf, rf) if f not in files)}")
            res = _store_session(SessionIn(
                patient_id=row["patient_id"] or f"ROW-{line}",
                meta={k: row.get(k) for k in CLINICAL_KEYS},
                left_csv=files[lf], right_csv=files[rf], left_label=row.get("left_label"),
                right_label=row.get("right_label"), notes=row.get("notes") or None,
                recorded_on=row.get("recorded_on") or None, left_name=row["left_file"], right_name=row["right_file"],
            ), source="bulk")
            results.append({"line": line, "patient_id": row["patient_id"], "ok": True, "id": res["id"],
                            "warnings": res["quality"]["left"]["warnings"] + res["quality"]["right"]["warnings"]})
        except HTTPException as e:
            results.append({"line": line, "patient_id": row.get("patient_id"), "ok": False, "error": e.detail})
        except ValidationError as e:
            results.append({"line": line, "patient_id": row.get("patient_id"), "ok": False,
                            "error": "; ".join(f"{'.'.join(map(str, er['loc']))}: {er['msg']}" for er in e.errors())})
    return {"imported": sum(r["ok"] for r in results), "failed": sum(not r["ok"] for r in results), "rows": results}


@app.get("/admin/sessions/{sid}", dependencies=[Depends(require_admin)])
def get_session(sid: str):
    s = storage.get_session(sid)
    if not s:
        raise HTTPException(404, detail="Session not found.")
    s["feature_labels"] = {f: inference.feature_label(f) for f in (s.get("features") or {}).get("left", {})}
    return s


class SessionPatch(BaseModel):
    patient_id: str | None = None
    meta: dict[str, Any] | None = None
    left_label: str | None = None
    right_label: str | None = None
    notes: str | None = None
    include: bool | None = None
    recorded_on: str | None = None


@app.patch("/admin/sessions/{sid}", dependencies=[Depends(require_admin)])
def patch_session(sid: str, body: SessionPatch):
    s = storage.get_session(sid, raw=True)
    if not s:
        raise HTTPException(404, detail="Session not found.")
    fields = body.model_dump(exclude_unset=True)
    if "left_label" in fields:
        fields["left_label"] = normalise_label(fields["left_label"])
    if "right_label" in fields:
        fields["right_label"] = normalise_label(fields["right_label"])
    if "meta" in fields:
        fields["meta"] = normalise_meta(fields["meta"])
    storage.update_session(sid, fields)
    if "meta" in fields:  # profile changes alter weight-normalised / clinical features
        training.refresh_session_features(storage.get_session(sid, raw=True))
    return storage.get_session(sid)


@app.delete("/admin/sessions/{sid}", dependencies=[Depends(require_admin)])
def delete_session(sid: str):
    if not storage.delete_session(sid):
        raise HTTPException(404, detail="Session not found.")
    return {"deleted": sid}


@app.post("/admin/features/recompute", dependencies=[Depends(require_admin)])
def recompute_features():
    done, failed = 0, []
    for s in storage.all_sessions_for_training() + [
            storage.get_session(x["id"], raw=True) for x in storage.list_sessions() if not x["include"]]:
        try:
            training.refresh_session_features(s)
            done += 1
        except SensorParseError as e:
            failed.append({"id": s["id"], "error": str(e)})
    return {"recomputed": done, "failed": failed, "feature_version": FEATURE_VERSION}


@app.get("/admin/dataset/export", dependencies=[Depends(require_admin)], response_class=PlainTextResponse)
def export_dataset():
    rows = []
    for s in storage.all_sessions_for_training():
        if not s.get("features") or s.get("feature_version") != FEATURE_VERSION:
            training.refresh_session_features(s)
        for side in ("left", "right"):
            rows.append({"session_id": s["id"], "patient_id": s["patient_id"], "side": side,
                         "label": s.get(f"{side}_label") or "", **s["features"][side]})
    cols = ["session_id", "patient_id", "side", "label", *feature_names()]
    csv = pd.DataFrame(rows, columns=cols).to_csv(index=False)
    return PlainTextResponse(csv, media_type="text/csv",
                             headers={"Content-Disposition": 'attachment; filename="smartsole_features.csv"'})


@app.get("/admin/templates/{name}", dependencies=[Depends(require_admin)], response_class=PlainTextResponse)
def template(name: str):
    if name == "manifest.csv":
        cols = admin_schema()["manifest_columns"]
        example = {"patient_id": "P001", "recorded_on": "2026-11-01", "left_file": "P001_left.csv",
                   "right_file": "P001_right.csv", "left_label": "moderate", "right_label": "low",
                   "notes": "callus under 2nd MTH (left)", "age": "62", "sex": "Male", "weight_kg": "81",
                   "height_cm": "175", "diabetes_years": "11", "hba1c": "8.1", "neuropathy": "yes", "pad": "no",
                   "deformity": "yes", "prior_ulcer": "no", "prior_amputation": "no"}
        text = pd.DataFrame([example], columns=cols).to_csv(index=False)
    elif name == "sensor_log.csv":
        s = generate_sessions(1, seed=3)[0]
        text = to_sensor_csv(s["left_df"])
    else:
        raise HTTPException(404, detail="Unknown template.")
    return PlainTextResponse(text, media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{name}"'})


# --- admin: training & models --------------------------------------------------------------

class TrainIn(BaseModel):
    source: str = "real"
    algorithms: list[str] | None = None
    activate: bool = True


@app.post("/admin/train", dependencies=[Depends(require_admin)])
def train(body: TrainIn):
    if body.algorithms:
        bad = [a for a in body.algorithms if a not in training.ALGORITHMS]
        if bad:
            raise HTTPException(422, detail=f"Unknown algorithm(s): {', '.join(bad)}.")
    if body.source not in {"real", "synthetic", "real+synthetic"}:
        raise HTTPException(422, detail="source must be 'real', 'synthetic' or 'real+synthetic'.")
    if body.source == "real":  # fail fast instead of inside the background job
        ready = training.dataset_readiness()
        if not ready["trainable"]:
            failed = [c["text"] for c in ready["checks"] if not c["ok"] and not c.get("optional")]
            raise HTTPException(422, detail="Dataset not ready: " + " ".join(failed))
    try:
        job = jobs.start("admin", source=body.source, algorithms=body.algorithms, activate=body.activate)
    except training.TrainingError as e:
        raise HTTPException(409, detail=str(e)) from e
    return job


@app.get("/admin/train/status", dependencies=[Depends(require_admin)])
def train_status():
    """Poll this after POST /admin/train: status is running | succeeded | failed | idle."""
    return jobs.status()


@app.get("/admin/models", dependencies=[Depends(require_admin)])
def list_models():
    return storage.list_models()


@app.get("/admin/models/{version}", dependencies=[Depends(require_admin)])
def get_model(version: str):
    m = storage.get_model(version)
    if not m:
        raise HTTPException(404, detail="Model not found.")
    m.pop("path", None)
    for imp in m["report"].get("feature_importance", []):
        imp["label"] = inference.feature_label(imp["feature"])
    return m


@app.post("/admin/models/{version}/activate", dependencies=[Depends(require_admin)])
def activate(version: str):
    if not storage.activate_model(version):
        raise HTTPException(404, detail="Model not found.")
    inference.invalidate_cache()
    return {"active": version}


@app.delete("/admin/models/{version}", dependencies=[Depends(require_admin)])
def delete_model(version: str):
    m = storage.get_model(version)
    if not m:
        raise HTTPException(404, detail="Model not found.")
    if m["is_active"]:
        raise HTTPException(409, detail="Cannot delete the active model - activate another one first.")
    storage.delete_model(version)
    Path(m["path"]).unlink(missing_ok=True)
    return {"deleted": version}


# --- admin: prediction log -----------------------------------------------------------------

@app.get("/admin/predictions", dependencies=[Depends(require_admin)])
def list_predictions():
    return storage.list_predictions()


class PromoteIn(BaseModel):
    patient_id: str = Field(min_length=1, max_length=80)
    left_label: str | None = None
    right_label: str | None = None
    notes: str | None = None


@app.post("/admin/predictions/{pid}/promote", dependencies=[Depends(require_admin)])
def promote(pid: str, body: PromoteIn):
    p = storage.get_prediction(pid)
    if not p:
        raise HTTPException(404, detail="Prediction not found.")
    if p["promoted_session_id"]:
        raise HTTPException(409, detail="Already added to the dataset.")
    res = _store_session(SessionIn(patient_id=body.patient_id, meta=p["meta"], left_csv=p["left_csv"],
                                   right_csv=p["right_csv"], left_label=body.left_label,
                                   right_label=body.right_label, notes=body.notes,
                                   recorded_on=p["created_at"][:10]), source="prediction")
    storage.mark_prediction_promoted(pid, res["id"])
    return res


@app.delete("/admin/predictions/{pid}", dependencies=[Depends(require_admin)])
def delete_prediction(pid: str):
    if not storage.delete_prediction(pid):
        raise HTTPException(404, detail="Prediction not found.")
    return {"deleted": pid}
