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

CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]

for _d in (DATA_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
