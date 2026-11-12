"""Build the pre-trained starter model that ships with the repo.

    npm run ml:build-starter        (or: cd ml && .venv/Scripts/python build_starter.py)

Trains on the reproducible synthetic dataset in a throw-away data directory, then writes
ml/models/starter.joblib (model + reference stats + training report). Commit that file.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
_tmp = tempfile.mkdtemp(prefix="smartsole-starter-")
os.environ["SMARTSOLE_DATA_DIR"] = _tmp  # isolate from the real database before importing the app
sys.path.insert(0, str(HERE))

import joblib  # noqa: E402
import sklearn  # noqa: E402

from app import config, storage, training  # noqa: E402


def main() -> None:
    try:
        storage.init_db()
        print(f"Training starter model on {config.BOOTSTRAP_PATIENTS} synthetic patients...")
        report = training.train(source="synthetic", activate=True)
        artifact = joblib.load(config.MODELS_DIR / f"{report['version']}.joblib")
        best = report["candidates"][0]["cv_metrics"]
        artifact.update(
            report=report,
            metrics={k: best.get(k) for k in ("macro_f1", "balanced_accuracy", "accuracy", "roc_auc",
                                               "at_risk_sensitivity", "at_risk_specificity")},
            sklearn_version=sklearn.__version__,
        )
        out = HERE / "models" / "starter.joblib"
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, out, compress=3)
        print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB): {report['version']} "
              f"{report['selected']} macro-F1 {best['macro_f1']:.3f}, scikit-learn {sklearn.__version__}")
    finally:
        shutil.rmtree(_tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
