"""Model training, patient-grouped cross-validated model selection, and the model registry.

Why this model family
---------------------
The data is *tabular* (engineered gait / thermal / moisture / clinical features per foot),
labelled by clinicians, and - at least initially - small (tens to a few hundred feet). On data
like that, tree ensembles (gradient boosting, random forest) are the strongest and most
robust choice, with a regularised logistic regression as an interpretable baseline. Deep
sequence models (LSTM / 1D-CNN on the raw time series) need thousands of labelled recordings
and are not appropriate yet.

Every training run evaluates all candidates with StratifiedGroupKFold *grouped by patient*
(both feet and repeat visits of one patient never sit on both sides of a split, which would
otherwise leak and inflate scores), picks the best by macro-F1, refits it on all data and
registers it as a new versioned model.
"""
from __future__ import annotations

import math
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config, storage
from .features import FEATURE_VERSION, feature_names, session_features
from .inference import feature_label
from .parsing import parse_sensor_csv
from .schema import LABELS, REQUIRED_KEYS
from .synthetic import synthetic_training_rows

_train_lock = threading.Lock()

ALGORITHMS = {
    "gradient_boosting": "Histogram Gradient Boosting",
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression (L2)",
}


class TrainingError(ValueError):
    pass


def _make(algo: str, n: int):
    leaf = int(max(3, min(20, n // 25)))
    if algo == "gradient_boosting":
        return HistGradientBoostingClassifier(learning_rate=0.06, max_iter=250, max_leaf_nodes=15,
                                              min_samples_leaf=leaf, l2_regularization=1.0,
                                              class_weight="balanced", random_state=0)
    if algo == "random_forest":
        return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=max(1, leaf // 3),
                                                        class_weight="balanced_subsample", n_jobs=-1,
                                                        random_state=0))])
    if algo == "logistic_regression":
        return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("scale", StandardScaler()),
                         ("clf", LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000))])
    raise TrainingError(f"Unknown algorithm '{algo}'.")


def _proba_full(model, X: pd.DataFrame, classes: list[str]) -> np.ndarray:
    """predict_proba aligned to `classes` (a fold model may not have seen every class)."""
    p = model.predict_proba(X)
    out = np.zeros((len(X), len(classes)))
    for j, c in enumerate(model.classes_):
        out[:, classes.index(c)] = p[:, j]
    return out


def _metrics(y_true: list[str], proba: np.ndarray, classes: list[str]) -> dict:
    y_pred = [classes[i] for i in proba.argmax(1)]
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
        "per_class": {c: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]),
                          "support": int(sup[i])} for i, c in enumerate(classes)},
        "confusion_matrix": {"labels": classes,
                             "matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist()},
    }
    try:
        if len(classes) == 2:
            m["roc_auc"] = roc_auc_score([c == classes[1] for c in y_true], proba[:, 1])
        else:
            order = sorted(range(len(classes)), key=lambda i: classes[i])  # sklearn needs sorted labels
            m["roc_auc"] = roc_auc_score(y_true, proba[:, order], multi_class="ovr", average="macro",
                                         labels=[classes[i] for i in order])
    except ValueError:
        m["roc_auc"] = None
    # Clinically most relevant view: "at risk" (moderate|high) vs low.
    if "low" in classes and len(classes) > 1:
        t = np.array([c != "low" for c in y_true])
        p = np.array([c != "low" for c in y_pred])
        m["at_risk_sensitivity"] = float((t & p).sum() / t.sum()) if t.sum() else None
        m["at_risk_specificity"] = float((~t & ~p).sum() / (~t).sum()) if (~t).sum() else None
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}


def _reference_stats(X: pd.DataFrame, y: list[str]) -> dict:
    low = X[np.array(y) == "low"] if "low" in y else X
    return {
        "median": X.median(numeric_only=True).to_dict(),
        "low_median": low.median(numeric_only=True).to_dict(),
        "mean": X.mean(numeric_only=True).to_dict(),
        "std": X.std(numeric_only=True).fillna(0).to_dict(),
    }


def _clean(d: dict) -> dict:
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in d.items()}


# --- dataset assembly ----------------------------------------------------------------------

def refresh_session_features(session: dict) -> dict:
    """Recompute features from the raw logs (after edits or a feature-engineering upgrade)."""
    left, right = parse_sensor_csv(session["left_csv"]), parse_sensor_csv(session["right_csv"])
    feats = session_features(session["meta"], left, right)
    feats = {"left": _clean(feats["left"]), "right": _clean(feats["right"]), "abs_delta_t": feats["abs_delta_t"]}
    storage.update_session(session["id"], {"features": feats, "feature_version": FEATURE_VERSION,
                                           "quality": {"left": left.quality, "right": right.quality}})
    session["features"] = feats
    return feats


def real_training_rows() -> tuple[list[dict], list[str], list[str]]:
    X, y, groups = [], [], []
    for s in storage.all_sessions_for_training():
        feats = s.get("features")
        if not feats or s.get("feature_version") != FEATURE_VERSION:
            feats = refresh_session_features(s)
        for side in ("left", "right"):
            label = s.get(f"{side}_label")
            if label in LABELS:
                X.append(feats[side])
                y.append(label)
                groups.append(str(s["patient_id"]))
    return X, y, groups


def dataset_readiness() -> dict:
    sessions = storage.list_sessions()
    included = [s for s in sessions if s["include"]]
    counts: Counter = Counter()
    patients: set = set()
    unlabelled = 0
    warn_sessions = 0
    missing_required = 0
    for s in included:
        for side in ("left", "right"):
            lab = s.get(f"{side}_label")
            if lab in LABELS:
                counts[lab] += 1
                patients.add(s["patient_id"])
            else:
                unlabelled += 1
        q = s.get("quality") or {}
        if any((q.get(side) or {}).get("warnings") for side in ("left", "right")):
            warn_sessions += 1
        if any(s["meta"].get(k) in (None, "") for k in REQUIRED_KEYS):
            missing_required += 1

    n = sum(counts.values())
    present = [c for c in LABELS if counts[c] > 0]
    checks = [
        {"id": "classes", "ok": len(present) >= 2,
         "text": f"At least 2 risk classes labelled (have: {', '.join(present) or 'none'})."},
        {"id": "per_class", "ok": all(counts[c] >= config.MIN_SAMPLES_PER_CLASS for c in present) and len(present) >= 2,
         "text": f"≥ {config.MIN_SAMPLES_PER_CLASS} labelled feet in every class present."},
        {"id": "patients", "ok": len(patients) >= config.MIN_PATIENTS,
         "text": f"≥ {config.MIN_PATIENTS} distinct patients (have {len(patients)}) - needed for honest cross-validation."},
        {"id": "all_classes", "ok": len(present) == len(LABELS),
         "text": "All three classes (low / moderate / high) represented.", "optional": True},
        {"id": "volume", "ok": n >= config.RECOMMENDED_SAMPLES,
         "text": f"≥ {config.RECOMMENDED_SAMPLES} labelled feet recommended for a stable model (have {n}).",
         "optional": True},
    ]
    trainable = all(c["ok"] for c in checks if not c.get("optional"))
    return {
        "sessions": len(sessions),
        "included_sessions": len(included),
        "labelled_feet": n,
        "unlabelled_feet": unlabelled,
        "class_counts": {c: counts[c] for c in LABELS},
        "patients": len(patients),
        "sessions_with_quality_warnings": warn_sessions,
        "sessions_missing_required_fields": missing_required,
        "checks": checks,
        "trainable": trainable,
    }


# --- training ------------------------------------------------------------------------------

def train(source: str = "real", algorithms: list[str] | None = None, activate: bool = True,
          synthetic_patients: int | None = None) -> dict:
    if source not in {"real", "synthetic", "real+synthetic"}:
        raise TrainingError("source must be 'real', 'synthetic' or 'real+synthetic'.")
    if not _train_lock.acquire(blocking=False):
        raise TrainingError("A training run is already in progress.")
    try:
        return _train(source, algorithms or list(ALGORITHMS), activate, synthetic_patients)
    finally:
        _train_lock.release()


def _train(source: str, algorithms: list[str], activate: bool, synthetic_patients: int | None) -> dict:
    started = datetime.now(timezone.utc)
    X_rows, y, groups = [], [], []
    n_real = 0
    if "real" in source:
        X_rows, y, groups = real_training_rows()
        n_real = len(y)
        if source == "real":
            ready = dataset_readiness()
            if not ready["trainable"]:
                failed = [c["text"] for c in ready["checks"] if not c["ok"] and not c.get("optional")]
                raise TrainingError("Dataset not ready: " + " ".join(failed))
    if "synthetic" in source:
        sx, sy, sg = synthetic_training_rows(synthetic_patients or config.BOOTSTRAP_PATIENTS, config.BOOTSTRAP_SEED)
        X_rows += sx
        y += sy
        groups += sg

    features = feature_names()
    X = pd.DataFrame(X_rows).reindex(columns=features).astype(float)
    classes = [c for c in LABELS if c in set(y)]
    if len(classes) < 2:
        raise TrainingError("Need at least two different risk labels to train.")
    counts = Counter(y)
    n_groups = len(set(groups))
    k = int(min(5, n_groups, min(counts.values())))
    if k < 2:
        raise TrainingError("Not enough patients / samples per class for cross-validation (need ≥ 2 of each).")

    warnings = []
    if len(y) < config.RECOMMENDED_SAMPLES:
        warnings.append(f"Only {len(y)} labelled feet - metrics will be noisy. Aim for {config.RECOMMENDED_SAMPLES}+.")
    if len(classes) < len(LABELS):
        warnings.append("Not every risk class is present; the model can only predict: " + ", ".join(classes) + ".")

    y_arr = np.array(y)
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=0)
    splits = list(cv.split(X, y_arr, groups))

    candidates = []
    for algo in algorithms:
        oof = np.zeros((len(y), len(classes)))
        for tr, te in splits:
            model = _make(algo, len(tr))
            model.fit(X.iloc[tr], y_arr[tr])
            oof[te] = _proba_full(model, X.iloc[te], classes)
        # Evaluate on real samples only when synthetic data was mixed in (that is what matters).
        mask = np.array([not str(g).startswith("SYN-") for g in groups]) if source == "real+synthetic" else \
            np.ones(len(y), bool)
        m = _metrics(list(y_arr[mask]), oof[mask], classes)
        candidates.append({"algorithm": algo, "name": ALGORITHMS[algo], "cv_metrics": m})

    candidates.sort(key=lambda c: (c["cv_metrics"]["macro_f1"], c["cv_metrics"]["balanced_accuracy"]), reverse=True)
    best = candidates[0]["algorithm"]

    final = _make(best, len(y))
    final.fit(X, y_arr)

    n_imp = min(len(y), 800)
    idx = np.random.default_rng(0).choice(len(y), n_imp, replace=False)
    imp = permutation_importance(final, X.iloc[idx], y_arr[idx], scoring="f1_macro", n_repeats=3,
                                 random_state=0, n_jobs=1)
    importances = sorted(({"feature": f, "label": feature_label(f), "importance": round(float(v), 5)}
                          for f, v in zip(features, imp.importances_mean)), key=lambda d: -d["importance"])

    data_source = {"real": "real", "synthetic": "synthetic-bootstrap", "real+synthetic": "real+synthetic"}[source]
    version = f"v{started.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
    path = config.MODELS_DIR / f"{version}.joblib"
    artifact = {
        "version": version,
        "model": final,
        "algorithm": best,
        "features": features,
        "feature_version": FEATURE_VERSION,
        "classes": classes,
        "reference": _reference_stats(X, y),
        "data_source": data_source,
        "created_at": started.isoformat(timespec="seconds"),
    }
    joblib.dump(artifact, path)

    best_metrics = candidates[0]["cv_metrics"]
    report = {
        "version": version,
        "data_source": data_source,
        "n_samples": len(y),
        "n_real_samples": n_real,
        "n_patients": n_groups,
        "class_counts": {c: counts.get(c, 0) for c in LABELS},
        "cv_folds": k,
        "cv_strategy": "StratifiedGroupKFold (grouped by patient)",
        "candidates": candidates,
        "selected": best,
        "feature_importance": importances[:25],
        "n_features": len(features),
        "warnings": warnings,
        "duration_s": round((datetime.now(timezone.utc) - started).total_seconds(), 2),
    }
    summary = {k2: best_metrics.get(k2) for k2 in ("macro_f1", "balanced_accuracy", "accuracy", "roc_auc",
                                                    "at_risk_sensitivity", "at_risk_specificity")}
    storage.insert_model(version=version, algorithm=best, data_source=data_source, n_samples=len(y),
                         metrics=summary, report=report, path=str(path), activate=activate)
    report["activated"] = activate
    return report
