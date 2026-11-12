# SmartSole — ML-based Diabetic Foot Ulcer risk screening

Smart-insole dashboard (Next.js) backed by a Python machine-learning service. Each insole logs
four plantar pressure zones (toe, metatarsal heads, midfoot, heel), skin temperature and in-shoe
humidity. Upload a left and a right recording with the patient profile, and the model predicts
a **low / moderate / high** DFU risk for each foot. It also lists the main risk drivers and
evidence-based clinical alerts.

```
┌──────────────────────────┐      ┌──────────────────────────────────────────────┐
│ Next.js (app/)           │      │ Python ML service (ml/)  FastAPI + sklearn   │
│  /        patient flow   │─────▶│  parsing → feature engineering → model       │
│  /admin   data & models  │ API  │  SQLite: sessions, model registry, pred. log │
│  /api/*   thin proxies   │      │  joblib model artefacts (versioned)          │
└──────────────────────────┘      └──────────────────────────────────────────────┘
```

## Quick start (local)

Requirements: Node 20+ and Python 3.11+.

```bash
npm install
npm run ml:setup          # creates ml/.venv and installs the Python deps
cp ml/.env.example ml/.env   # then set ADMIN_TOKEN (needed for /admin)
cp .env.example .env.local
```

Run the two processes in separate terminals:

```bash
npm run ml      # ML service on http://127.0.0.1:8000 (first start trains the starter model)
npm run dev     # web app on http://localhost:3000
```

- Dashboard: http://localhost:3000. Use the **demo recording** buttons to try it without hardware.
- Admin: http://localhost:3000/admin. Unlock it with the `ADMIN_TOKEN` from `ml/.env`.
- Tests: `npm run ml:test`.

## The ML backend

### Pipeline
1. **Parsing** (`ml/app/parsing.py`) reads the firmware CSV (`Time,P1_Toe,P2_Met,P3_Heel,P4_Mid,Temp,Hum`).
   It accepts any delimiter, an optional or reordered header, clock, second or millisecond timestamps,
   and a missing time column. It also writes a quality report covering malformed rows, flat-lined
   sensors, implausible temperatures and short recordings.
2. **Feature engineering** (`ml/app/features.py`) produces about 80 features per foot:
   - Per zone: mean, peak (p95), max, variability, load share, pressure per kg, time above 200 kPa.
   - Gait: cadence, stance fraction, forefoot/heel ratio, pressure-time integral.
   - Temperature and humidity: level, spread and trend.
   - **Contralateral**: ΔT against the other foot (mean, sustained rolling maximum, time above 2.2 °C)
     and pressure and humidity asymmetry.
   - Clinical: age, sex, weight, BMI, diabetes duration, HbA1c, neuropathy, PAD, deformity,
     prior ulcer, prior amputation. Missing values are allowed.
   - The original rule-based indices (PRS, TRS, MRS, TBI) are kept as knowledge-guided features.
3. **Models** (`ml/app/training.py`): Histogram Gradient Boosting, Random Forest and L2 Logistic
   Regression. All three are evaluated with **StratifiedGroupKFold grouped by patient**, so neither
   the other foot nor a repeat visit of the same patient can leak across folds. The run reports
   macro-F1, balanced accuracy, ROC-AUC, at-risk sensitivity and specificity, the confusion matrix
   and permutation feature importance. The best model by macro-F1 is saved as a new **version**.
4. **Inference** (`ml/app/inference.py`) returns, for each foot:
   - Class probabilities and a 0–100 risk score.
   - The top risk drivers (a counterfactual against a typical low-risk foot).
   - A warning when the input is out of distribution.
   - Model-independent clinical alerts: ΔT above 2.2 °C, peak pressure above 200 kPa, humidity above 75 %.

**Why this model family:** the data is tabular, the labelled set will be small at first, and
some clinical fields will be missing. Gradient-boosted trees and random forests are the most
data-efficient and robust choice for data like this. Deep sequence models (LSTM or CNN on the
raw signal) only make sense once there are thousands of labelled recordings.

### Starter (bootstrap) model
Before any real data has been uploaded, the service trains a **starter model** on simulated
recordings. Each simulated foot is labelled by the legacy rules plus IWGDF-style clinical risk
points. The UI marks this model as *"Starter model · trained on simulated data"*. Replace it by
training on real labelled data in `/admin`.

### Admin page (`/admin`)
| Tab | Purpose |
|---|---|
| Overview | Dataset size, class balance, training-readiness checks, active model metrics |
| Add data | One session at a time: profile, both logs and a label per foot, with a feature preview. Bulk import takes a manifest CSV plus the log files. Templates can be downloaded. |
| Dataset | Edit labels inline, exclude recordings, view the extracted features, recompute features, export the ML-ready CSV |
| Train model | Choose the data source (real, real + synthetic, or synthetic) and the algorithms. Cross-validates and registers the result. |
| Models | Version registry with full reports. Activate (roll back to) or delete a version. |
| Prediction log | Every dashboard assessment is stored. Label the real outcome later and add it to the dataset. |
| Data guide | Recording protocol, label definitions, file format, feature list |

Labels are per foot:
- **low**: no lesion.
- **moderate**: pre-ulcerative signs, or IWGDF risk category 2–3.
- **high**: active ulcer, or an ulcer developed during follow-up.

### ML service configuration (`ml/.env`)
| Variable | Default | Meaning |
|---|---|---|
| `ADMIN_TOKEN` | *(empty → admin disabled)* | Token required by every `/admin` call |
| `PRESSURE_SCALE` | `100` | Raw pressure unit → kPa |
| `LOG_PREDICTIONS` | `true` | Store predictions for later labelling |
| `SMARTSOLE_DATA_DIR` | `ml/data` | SQLite DB and model files |
| `MIN_SAMPLES_PER_CLASS` / `MIN_PATIENTS` / `RECOMMENDED_SAMPLES` | `10` / `8` / `150` | Readiness thresholds |

The web app needs only `ML_API_URL` (default `http://127.0.0.1:8000`).

## Deployment
The ML service needs a host with a **persistent disk** for its SQLite database and model files,
so it cannot run on Vercel. The recommended setup is **Railway for the ML service** and
**Vercel for the web app**.

### ML service on Railway
`railway.json` (repo root) tells Railway to build `ml/Dockerfile`, so leave the service's
Root Directory empty.
1. New Project → Deploy from GitHub repo → this repository.
2. Add a **Volume** to the service mounted at `/data` (database + trained models).
3. **Variables:** `ADMIN_TOKEN` = a long random secret. Optional: `PRESSURE_SCALE`, `LOG_PREDICTIONS`.
4. **Settings → Networking → Generate Domain** (the app listens on Railway's `$PORT`).

On first boot the service answers immediately and trains the starter model in the background
(about 2 minutes). `/health` shows `"training": "bootstrap"` until it is ready.

### Web app on Vercel
1. Add New Project → import this repository (framework: Next.js, root directory: repo root).
2. Environment variable `ML_API_URL` = the Railway domain, e.g. `https://xxxx.up.railway.app`.
3. Deploy, then open `/admin` and unlock it with the Railway `ADMIN_TOKEN`.

Training runs as a background job (the admin page polls `/admin/train/status`), so it is not
limited by Vercel's request timeout.

### Single server (Docker Compose)
```bash
cp ml/.env.example ml/.env   # set ADMIN_TOKEN
docker compose up -d --build # ml on :8000, web on :3000
```
Put a reverse proxy with HTTPS (e.g. Caddy) in front for public access.

> SmartSole is a screening aid, not a diagnostic device. Predictions must be interpreted by a
> qualified clinician alongside a physical foot examination.
