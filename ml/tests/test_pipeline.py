"""End-to-end tests for parsing, features, training and the HTTP API.

Run from the ml/ folder:  .venv/Scripts/python -m pytest -q
"""
from __future__ import annotations

import atexit
import os
import shutil
import tempfile

import numpy as np
import pytest

_tmp = tempfile.mkdtemp(prefix="smartsole-test-")
atexit.register(shutil.rmtree, _tmp, ignore_errors=True)
os.environ["SMARTSOLE_DATA_DIR"] = _tmp
os.environ["ADMIN_TOKEN"] = "test-token"
os.environ["BOOTSTRAP_PATIENTS"] = "120"
os.environ["MIN_SAMPLES_PER_CLASS"] = "3"

from fastapi.testclient import TestClient  # noqa: E402

from app import config, jobs, main, starter  # noqa: E402
from app.features import feature_names, session_features  # noqa: E402
from app.parsing import SensorParseError, parse_sensor_csv  # noqa: E402
from app.synthetic import demo_session, generate_sessions, to_sensor_csv  # noqa: E402

H = {"X-Admin-Token": "test-token"}

LEGACY_LOG = "\n".join(
    ["Time,P1_Toe,P2_Met,P3_Heel,P4_Mid,Temp,Hum"]
    + [f"20:45:{18 + i // 10:02d}.{(i % 10) * 100:03d}, 0.95, 1.31, 1.42, 0.41, 30.4, 42.1" for i in range(50)]
)


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:
        # The bundled pre-trained starter is activated on boot: no training job runs.
        assert c.get("/health").status_code == 200
        assert jobs.status()["status"] == "idle"
        yield c


# --- parsing ---------------------------------------------------------------------------------

def test_parse_legacy_format():
    log = parse_sensor_csv(LEGACY_LOG)
    assert log.quality["rows"] == 50
    assert log.quality["has_time_column"]
    assert abs(log.quality["sampling_hz"] - 10) < 0.5
    assert np.isclose(log.df["toe"].iloc[0], 95.0)


def test_parse_headerless_semicolon_no_time():
    text = "\n".join("1.0;1.2;1.3;0.4;31.0;50" for _ in range(30))
    log = parse_sensor_csv(text)
    assert log.quality["rows"] == 30 and not log.quality["has_time_column"]


def test_parse_reordered_header():
    rows = ["Hum,Temp,P4_Mid,P3_Heel,P2_Met,P1_Toe,Time"] + [f"40,30.0,0.4,1.4,1.3,0.9,{i * 100}" for i in range(20)]
    log = parse_sensor_csv("\n".join(rows))
    assert np.isclose(log.df["toe"].iloc[0], 90.0) and np.isclose(log.df["rh"].iloc[0], 40.0)
    assert abs(log.quality["sampling_hz"] - 10) < 0.5  # millisecond timestamps detected


def test_parse_rejects_garbage():
    with pytest.raises(SensorParseError):
        parse_sensor_csv("hello\nworld")


# --- features --------------------------------------------------------------------------------

def test_feature_vector_is_stable_and_finite():
    s = generate_sessions(1, seed=1)[0]
    left = parse_sensor_csv(to_sensor_csv(s["left_df"]))
    right = parse_sensor_csv(to_sensor_csv(s["right_df"]))
    feats = session_features(s["meta"], left, right)
    assert list(feats["left"].keys()) == feature_names()
    sensor_feats = [k for k in feature_names() if not k.startswith("pt_")]
    assert all(np.isfinite(feats["left"][k]) for k in sensor_feats)
    assert np.isclose(feats["left"]["bi_dT_mean"], -feats["right"]["bi_dT_mean"])


# --- API -------------------------------------------------------------------------------------

def test_health_has_bootstrap_model(client):
    r = client.get("/health").json()
    assert r["model"]["data_source"] == "synthetic-bootstrap" and r["training"] is None
    assert r["model"]["version"] == starter.load_bundle()["version"]  # the shipped starter, not retrained


def test_starter_bundle_fallbacks(tmp_path, monkeypatch):
    import joblib

    monkeypatch.setattr(config, "STARTER_MODEL_PATH", tmp_path / "missing.joblib")
    assert starter.load_bundle() is None and starter.register_bundled() is None  # -> background training

    stale = tmp_path / "stale.joblib"
    joblib.dump({"feature_version": -1}, stale)
    monkeypatch.setattr(config, "STARTER_MODEL_PATH", stale)
    assert starter.load_bundle() is None  # incompatible feature set -> train instead


def test_predict_demo_scenarios_are_ordered(client):
    scores = {}
    for sc in ("healthy", "moderate", "critical"):
        d = demo_session(sc)
        r = client.post("/predict", json={"meta": d["meta"], "left_csv": d["left_csv"], "right_csv": d["right_csv"]})
        assert r.status_code == 200, r.text
        body = r.json()
        assert set(body["left"]["probabilities"]) == {"low", "moderate", "high"}
        scores[sc] = body["left"]["dfuScore"]
    assert scores["healthy"] < scores["moderate"] < scores["critical"]
    assert body["overallStatus"] == "CRITICAL ALERT"
    assert any(f["code"] == "DELTA_T" for f in body["clinicalFlags"])


def test_predict_validation(client):
    r = client.post("/predict", json={"meta": {"age": 50, "sex": "Male"}, "left_csv": LEGACY_LOG, "right_csv": LEGACY_LOG})
    assert r.status_code == 422 and "Weight" in r.json()["detail"]


def test_admin_requires_token(client):
    assert client.get("/admin/overview").status_code == 401
    assert client.get("/admin/overview", headers={"X-Admin-Token": "nope"}).status_code == 401


def test_admin_full_workflow(client):
    # 0. readiness gate refuses to train on an empty dataset
    r = client.post("/admin/train", headers=H, json={"source": "real"})
    assert r.status_code == 422 and "not ready" in r.json()["detail"]

    # 1. add labelled sessions for 12 patients
    sessions = generate_sessions(12, seed=99)
    ids = []
    for i, s in enumerate(sessions):
        labels = ("high", "low") if i % 3 == 0 else ("moderate", "low") if i % 3 == 1 else ("low", "low")
        if i % 3 == 0:
            labels = ("high", "moderate")
        r = client.post("/admin/sessions", headers=H, json={
            "patient_id": f"P{i:03d}", "meta": s["meta"], "left_csv": to_sensor_csv(s["left_df"]),
            "right_csv": to_sensor_csv(s["right_df"]), "left_label": labels[0], "right_label": labels[1]})
        assert r.status_code == 200, r.text
        ids.append(r.json()["id"])

    ov = client.get("/admin/overview", headers=H).json()
    assert ov["readiness"]["labelled_feet"] == 24
    assert ov["readiness"]["patients"] == 12

    # 2. edit + exclude
    r = client.patch(f"/admin/sessions/{ids[0]}", headers=H, json={"left_label": "moderate"})
    assert r.json()["left_label"] == "moderate"
    client.patch(f"/admin/sessions/{ids[0]}", headers=H, json={"left_label": "high"})

    # 3. export
    csv = client.get("/admin/dataset/export", headers=H)
    assert csv.status_code == 200 and csv.text.count("\n") == 25

    # 4. bulk import
    s = generate_sessions(1, seed=5)[0]
    manifest = "patient_id,left_file,right_file,left_label,right_label,age,sex,weight_kg,neuropathy\n" \
               "B001,b_left.csv,b_right.csv,low,low,55,F,62,no\n" \
               "B002,missing.csv,b_right.csv,low,low,55,F,62,no\n"
    r = client.post("/admin/sessions/bulk", headers=H, json={
        "manifest_csv": manifest,
        "files": {"b_left.csv": to_sensor_csv(s["left_df"]), "b_right.csv": to_sensor_csv(s["right_df"])}}).json()
    assert r["imported"] == 1 and r["failed"] == 1

    # 5. train on real data
    job = client.post("/admin/train", headers=H, json={"source": "real"})
    assert job.status_code == 200, job.text
    assert job.json()["status"] == "running"
    assert client.post("/admin/train", headers=H, json={"source": "real"}).status_code == 409  # one at a time
    jobs.wait()
    status = client.get("/admin/train/status", headers=H).json()
    assert status["status"] == "succeeded", status
    rep = status["report"]
    assert rep["data_source"] == "real" and rep["activated"]
    assert len(rep["candidates"]) == 3
    assert all(c["cv_metrics"]["roc_auc"] is not None for c in rep["candidates"])
    assert client.get("/health").json()["model"]["version"] == rep["version"]

    # 6. predictions use the new model and are logged; promote one into the dataset
    d = demo_session("moderate")
    pred = client.post("/predict", json={"meta": d["meta"], "left_csv": d["left_csv"], "right_csv": d["right_csv"]}).json()
    assert pred["model"]["version"] == rep["version"] and not pred["model"]["isBootstrap"]
    pr = client.post(f"/admin/predictions/{pred['predictionId']}/promote", headers=H,
                     json={"patient_id": "PX", "left_label": "moderate", "right_label": "low"})
    assert pr.status_code == 200
    assert client.post(f"/admin/predictions/{pred['predictionId']}/promote", headers=H,
                       json={"patient_id": "PX"}).status_code == 409

    # 7. model registry: roll back to bootstrap, cannot delete active
    models = client.get("/admin/models", headers=H).json()
    boot = next(m for m in models if m["data_source"] == "synthetic-bootstrap")
    assert client.post(f"/admin/models/{boot['version']}/activate", headers=H).status_code == 200
    assert client.delete(f"/admin/models/{boot['version']}", headers=H).status_code == 409
    report = client.get(f"/admin/models/{rep['version']}", headers=H).json()
    assert report["report"]["feature_importance"][0]["label"]
