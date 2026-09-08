"""Probability-of-default model ladder, from a trivial baseline to gradient boosting.

The ladder is deliberate. A credit-risk model has to justify its complexity: the
logistic regression is the reference point a scorecard team would actually deploy, and
every model above it has to earn its place on discrimination *and* calibration, not on
accuracy.

Hyper-parameters are chosen on the validation fold with a small, explicit grid. The
test fold is not used anywhere in this module.
"""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import config as C
from .data_prep import FEATURE_SETS, FeatureSet, load_splits

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except ImportError:  # pragma: no cover
    HAS_LGBM = False

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False


# --------------------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------------------
def build_preprocessor(fs: FeatureSet, scale: bool = True) -> ColumnTransformer:
    """Scale numeric inputs and one-hot encode categoricals.

    Scaling matters for the penalised logistic regression (so the L2 penalty treats
    features comparably) and is harmless for the tree ensembles, so the same transformer
    definition is reused with ``scale`` toggled off for trees to keep their splits on the
    original, interpretable scale.
    """
    numeric = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", numeric, fs.numeric),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                fs.categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# --------------------------------------------------------------------------------------
# Candidate specifications
# --------------------------------------------------------------------------------------
def candidate_specs() -> dict[str, dict]:
    """Model name -> {estimator factory, validation grid, scaling, notes}."""
    specs: dict[str, dict] = {
        "Baseline (prior)": {
            "factory": lambda **kw: DummyClassifier(strategy="prior", **kw),
            "grid": [{}],
            "scale": True,
            "notes": (
                "Predicts the training-set default rate for every account. It has no "
                "discriminatory power by construction and exists to show what any real "
                "model must beat."
            ),
        },
        "Logistic regression": {
            "factory": lambda **kw: LogisticRegression(
                max_iter=3000, solver="lbfgs", random_state=C.RANDOM_STATE, **kw
            ),
            "grid": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
            "scale": True,
            "notes": (
                "The scorecard reference model: monotone in each input, coefficients "
                "readable as log-odds, and the easiest model to defend in a credit "
                "committee or a model-validation review."
            ),
        },
        "Decision tree": {
            "factory": lambda **kw: DecisionTreeClassifier(random_state=C.RANDOM_STATE, **kw),
            "grid": [
                {"max_depth": d, "min_samples_leaf": leaf}
                for d in (3, 4, 5, 6)
                for leaf in (50, 200)
            ],
            "scale": False,
            "notes": (
                "A single shallow tree. Included because its splits read as policy rules, "
                "which makes it a useful bridge between the statistical model and a "
                "credit policy discussion."
            ),
        },
        "Random forest": {
            "factory": lambda **kw: RandomForestClassifier(
                random_state=C.RANDOM_STATE, n_jobs=-1, **kw
            ),
            "grid": [
                {"n_estimators": 400, "max_depth": d, "min_samples_leaf": leaf}
                for d in (8, 12)
                for leaf in (20, 50)
            ],
            "scale": False,
            "notes": (
                "Bagged trees. Reduces the variance of the single tree and gives a "
                "reference for how much of the gain is ensembling rather than boosting."
            ),
        },
    }

    if HAS_LGBM:
        specs["Gradient boosting (LightGBM)"] = {
            "factory": lambda **kw: LGBMClassifier(
                random_state=C.RANDOM_STATE, n_jobs=-1, verbose=-1, **kw
            ),
            "grid": [
                {
                    "n_estimators": n,
                    "learning_rate": lr,
                    "num_leaves": leaves,
                    "min_child_samples": 50,
                    "subsample": 0.9,
                    "subsample_freq": 1,
                    "colsample_bytree": 0.8,
                    "reg_lambda": 1.0,
                }
                for n in (300, 600)
                for lr in (0.03, 0.05)
                for leaves in (15, 31)
            ],
            "scale": False,
            "notes": (
                "Gradient-boosted trees. Captures interactions the scorecard cannot, at "
                "the cost of direct interpretability, which is why SHAP is used later."
            ),
        }

    if HAS_XGB:
        specs["Gradient boosting (XGBoost)"] = {
            "factory": lambda **kw: XGBClassifier(
                random_state=C.RANDOM_STATE,
                n_jobs=-1,
                eval_metric="logloss",
                tree_method="hist",
                **kw,
            ),
            "grid": [
                {
                    "n_estimators": n,
                    "learning_rate": lr,
                    "max_depth": d,
                    "min_child_weight": 5,
                    "subsample": 0.9,
                    "colsample_bytree": 0.8,
                    "reg_lambda": 1.0,
                }
                for n in (300, 600)
                for lr in (0.03, 0.05)
                for d in (3, 5)
            ],
            "scale": False,
            "notes": (
                "A second boosting implementation, kept as a cross-check that the "
                "measured gain is a property of the method rather than of one library."
            ),
        }

    return specs


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
def fit_one(name: str, spec: dict, fs: FeatureSet, train: pd.DataFrame, val: pd.DataFrame):
    """Fit every grid point on train, keep the one with the best validation ROC-AUC."""
    best = {"auc": -np.inf, "pipe": None, "params": None}
    for params in spec["grid"]:
        pipe = Pipeline(
            [
                ("prep", build_preprocessor(fs, scale=spec["scale"])),
                ("clf", spec["factory"](**params)),
            ]
        )
        pipe.fit(train[fs.all], train[C.TARGET])
        p = pipe.predict_proba(val[fs.all])[:, 1]
        auc = roc_auc_score(val[C.TARGET], p)
        if auc > best["auc"]:
            best = {"auc": float(auc), "pipe": pipe, "params": params}
    return best


class FlooredProbabilityModel:
    """Wraps a fitted classifier and applies a floor and cap to the predicted PD.

    Isotonic calibration is a step function, and on a fold this size it happily assigns a
    probability of exactly 0 or exactly 1 to accounts in its outermost steps. Neither is a
    defensible credit risk estimate: a PD of zero asserts that no loss is possible, which
    would zero out that account's expected loss entirely, and a PD of one asserts certainty
    from a handful of observations.

    Supervisory frameworks handle this with an explicit PD floor - Basel sets a 0.03%
    floor for retail exposures - and the same device is used here, with a symmetric cap so
    that no account is treated as a certain default either. The floor is applied to the
    default probability and the complement is adjusted so the two still sum to one.
    """

    def __init__(self, estimator, floor: float = 0.0003, cap: float = 0.9997):
        self.estimator = estimator
        self.floor = float(floor)
        self.cap = float(cap)

    def predict_proba(self, X):
        proba = np.asarray(self.estimator.predict_proba(X), dtype=float).copy()
        p1 = np.clip(proba[:, 1], self.floor, self.cap)
        proba[:, 1] = p1
        proba[:, 0] = 1.0 - p1
        return proba

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def classes_(self):
        return getattr(self.estimator, "classes_", np.array([0, 1]))


def select_calibration_method(
    pipe, val: pd.DataFrame, fs: FeatureSet, methods: tuple[str, ...] = ("isotonic", "sigmoid")
) -> tuple[str, dict[str, float]]:
    """Choose between isotonic and Platt scaling by cross-validation inside the validation fold.

    Fitting a calibrator on a fold and then judging it on the same fold would flatter the
    more flexible method - isotonic can bend to fit almost anything. So the comparison is
    made across five folds *within* the validation set: fit on four, score the fifth. The
    test fold is not involved at any point.
    """
    scores: dict[str, float] = {}
    for method in methods:
        fold_scores = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)
        for fit_idx, score_idx in skf.split(val[fs.all], val[C.TARGET]):
            fit_part, score_part = val.iloc[fit_idx], val.iloc[score_idx]
            cal = CalibratedClassifierCV(FrozenEstimator(pipe), method=method)
            cal.fit(fit_part[fs.all], fit_part[C.TARGET])
            p = cal.predict_proba(score_part[fs.all])[:, 1]
            fold_scores.append(brier_score_loss(score_part[C.TARGET], p))
        scores[method] = float(np.mean(fold_scores))
    return min(scores, key=scores.get), scores


def calibrate(pipe, val: pd.DataFrame, fs: FeatureSet, method: str = "isotonic"):
    """Post-hoc probability calibration on the held-out validation fold.

    Discrimination (does the model rank accounts correctly?) and calibration (is a
    predicted 8% actually an 8% default rate?) are different properties. Expected loss
    is a product of *levels*, not ranks, so an uncalibrated score cannot be used in
    EL = PD x LGD x EAD. The model is frozen first so that calibration uses the
    validation fold only and never refits on it, and the result is wrapped in a PD floor
    and cap so that no account is assigned an impossible probability.
    """
    calibrated = CalibratedClassifierCV(FrozenEstimator(pipe), method=method)
    calibrated.fit(val[fs.all], val[C.TARGET])
    return FlooredProbabilityModel(calibrated, floor=C.PD_FLOOR, cap=C.PD_CAP)


def train_all(feature_set: str = "behavioural", calibrate_method: str | None = None) -> dict:
    """Train the full ladder on one feature set and persist the fitted objects.

    ``calibrate_method`` defaults to None, meaning the method is chosen per model by
    cross-validation inside the validation fold rather than asserted up front.
    """
    fs = FEATURE_SETS[feature_set]
    splits = load_splits()
    train, val = splits["train"], splits["val"]

    results: dict[str, dict] = {}
    calibration_choices: dict[str, dict] = {}
    for name, spec in candidate_specs().items():
        t0 = time.perf_counter()
        best = fit_one(name, spec, fs, train, val)
        fit_seconds = time.perf_counter() - t0

        cal, chosen, cal_scores = None, None, None
        if name != "Baseline (prior)":
            if calibrate_method is None:
                chosen, cal_scores = select_calibration_method(best["pipe"], val, fs)
            else:
                chosen, cal_scores = calibrate_method, {}
            cal = calibrate(best["pipe"], val, fs, method=chosen)
            calibration_choices[name] = {"method": chosen, "cv_brier_by_method": cal_scores}

        tag = f"{feature_set}__{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
        joblib.dump(best["pipe"], C.MODELS / f"{tag}.joblib")
        if cal is not None:
            joblib.dump(cal, C.MODELS / f"{tag}__calibrated.joblib")

        results[name] = {
            "params": best["params"],
            "val_auc_uncalibrated": best["auc"],
            "fit_seconds": round(fit_seconds, 2),
            "notes": spec["notes"],
            "artifact": f"{tag}.joblib",
            "calibrated_artifact": f"{tag}__calibrated.joblib" if cal is not None else None,
            "n_grid_points": len(spec["grid"]),
            "calibration_method": chosen,
            "calibration_cv_brier": cal_scores,
        }
        print(f"  {name:<32} val AUC {best['auc']:.4f}  "
              f"calibration {chosen or '-':<9} ({fit_seconds:.1f}s)")

    manifest = {
        "feature_set": feature_set,
        "n_features": len(fs.all),
        "calibration_selection": "cross-validated within the validation fold"
        if calibrate_method is None else calibrate_method,
        "calibration_choices": calibration_choices,
        "pd_floor": C.PD_FLOOR,
        "pd_cap": C.PD_CAP,
        "random_state": C.RANDOM_STATE,
        "models": results,
    }
    (C.MODELS / f"manifest_{feature_set}.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_model(feature_set: str, name: str, calibrated: bool = True):
    tag = f"{feature_set}__{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
    suffix = "__calibrated" if calibrated else ""
    path = C.MODELS / f"{tag}{suffix}.joblib"
    if not path.exists() and calibrated:
        path = C.MODELS / f"{tag}.joblib"
    return joblib.load(path)


if __name__ == "__main__":
    # Re-import through the package before training. Run as ``python -m src.models`` this
    # file is ``__main__``, so any object pickled from here would record its class as
    # ``__main__.FlooredProbabilityModel`` and could not be loaded by ``src.run_pipeline``
    # or the dashboards. Calling the packaged copy keeps the saved artefacts portable.
    from src.models import train_all as packaged_train_all

    for fs_name in ("behavioural", "origination"):
        print(f"\nTraining ladder on the {fs_name} feature set")
        packaged_train_all(fs_name)
