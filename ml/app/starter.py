"""Pre-trained starter model shipped in the repository (ml/models/starter.joblib).

On boot, if there is no usable active model, the bundled starter is registered and activated
instantly instead of training one (which takes minutes on small hosts such as Render's free
tier). Rebuild the bundle with `npm run ml:build-starter` whenever the feature set or the
synthetic generator changes.
"""
from __future__ import annotations

import logging
import shutil
import warnings

import joblib
import sklearn

from . import config, storage
from .features import FEATURE_VERSION

log = logging.getLogger("smartsole")


def load_bundle() -> dict | None:
    """Return the bundled artifact if present and compatible with this code, else None."""
    path = config.STARTER_MODEL_PATH
    if not path.exists():
        log.info("No bundled starter model at %s.", path)
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=UserWarning)  # surfaces sklearn version mismatches
            bundle = joblib.load(path)
    except Exception as e:  # noqa: BLE001 - any load problem means "train instead"
        log.warning("Bundled starter model could not be loaded (%s); will train instead.", e)
        return None
    if bundle.get("feature_version") != FEATURE_VERSION:
        log.warning("Bundled starter model uses feature set v%s (code is v%s); will train instead.",
                    bundle.get("feature_version"), FEATURE_VERSION)
        return None
    if bundle.get("sklearn_version") and bundle["sklearn_version"] != sklearn.__version__:
        log.warning("Bundled starter model was built with scikit-learn %s but %s is installed; will train instead.",
                    bundle["sklearn_version"], sklearn.__version__)
        return None
    return bundle


def register_bundled() -> str | None:
    """Copy the bundled starter into the models dir, register it and make it active."""
    bundle = load_bundle()
    if bundle is None:
        return None
    version = bundle["version"]
    target = config.MODELS_DIR / f"{version}.joblib"
    if not target.exists():
        shutil.copyfile(config.STARTER_MODEL_PATH, target)
    if storage.get_model(version) is None:
        storage.insert_model(version=version, algorithm=bundle["algorithm"], data_source=bundle["data_source"],
                             n_samples=bundle["report"]["n_samples"], metrics=bundle["metrics"],
                             report=bundle["report"], path=str(target), activate=True)
    else:
        storage.activate_model(version)
    log.info("Activated bundled starter model %s (no training needed).", version)
    return version
