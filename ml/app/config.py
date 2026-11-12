"""Runtime configuration for the SmartSole ML service (all overridable by env vars)."""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Minimal .env support (ml/.env), without overriding variables already set in the environment.
_env_file = BASE_DIR / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
DATA_DIR = Path(os.getenv("SMARTSOLE_DATA_DIR", BASE_DIR / "data"))
MODELS_DIR = Path(os.getenv("SMARTSOLE_MODELS_DIR", DATA_DIR / "models"))
DB_PATH = Path(os.getenv("SMARTSOLE_DB_PATH", DATA_DIR / "smartsole.db"))

# Token required on every /admin/* call (sent as the X-Admin-Token header).
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

# Raw sensor value -> kPa. The insole firmware logs pressure in units where 1.00 == 100 kPa,
# which is what the original dashboard assumed (parts[i] * 100).
PRESSURE_SCALE = float(os.getenv("PRESSURE_SCALE", "100"))

# Used when the CSV has no parsable time column.
DEFAULT_SAMPLING_HZ = float(os.getenv("DEFAULT_SAMPLING_HZ", "10"))

# Store every /predict call so it can later be labelled with the real outcome and promoted
# into the training set from the admin page.
LOG_PREDICTIONS = os.getenv("LOG_PREDICTIONS", "true").lower() in {"1", "true", "yes"}

# Seed for the synthetic bootstrap dataset so the starter model is reproducible.
BOOTSTRAP_SEED = int(os.getenv("BOOTSTRAP_SEED", "42"))
BOOTSTRAP_PATIENTS = int(os.getenv("BOOTSTRAP_PATIENTS", "400"))

# Minimums before the admin page reports the real dataset as "ready to train".
MIN_SAMPLES_PER_CLASS = int(os.getenv("MIN_SAMPLES_PER_CLASS", "10"))
MIN_PATIENTS = int(os.getenv("MIN_PATIENTS", "8"))
RECOMMENDED_SAMPLES = int(os.getenv("RECOMMENDED_SAMPLES", "150"))

# Parallel workers for model training (-1 = all cores). Set to 1 on tiny hosts such as
# Render's free tier (0.1 CPU), where extra threads only add memory and contention.
N_JOBS = int(os.getenv("ML_N_JOBS", "-1"))

# Set on hosts without a persistent disk (e.g. Render free): the admin page then warns that
# uploaded data and trained models are lost whenever the service restarts.
EPHEMERAL_STORAGE = os.getenv("EPHEMERAL_STORAGE", "false").lower() in {"1", "true", "yes"}

# Pre-trained starter model shipped in the repo; registered on boot instead of training.
STARTER_MODEL_PATH = Path(os.getenv("STARTER_MODEL_PATH", BASE_DIR / "models" / "starter.joblib"))

CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]

for _d in (DATA_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
