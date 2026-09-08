"""Model explainability: scorecard coefficients, permutation importance and SHAP.

A PD model that cannot be explained will not pass a model-validation review, and it will
not survive a conversation with a credit officer either. Three complementary views are
produced:

* **Logistic-regression odds ratios** - the direction and size of each effect, in the
  language a scorecard team already uses.
* **Permutation importance** - measured on held-out data, so it reflects what the model
  actually relies on rather than what it split on during training.
* **SHAP** - additive per-account attributions for the gradient-boosting model, which
  give both a global ranking and a per-account reason code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from . import config as C

try:
    import shap

    HAS_SHAP = True
except ImportError:  # pragma: no cover
    HAS_SHAP = False


def feature_names_from_pipeline(pipe) -> list[str]:
    """Recover post-transformation feature names from a fitted pipeline."""
    return list(pipe.named_steps["prep"].get_feature_names_out())


def logistic_odds_ratios(pipe, top_n: int | None = None) -> pd.DataFrame:
    """Coefficients of the fitted logistic regression as odds ratios.

    Inputs are standardised, so each coefficient is the change in log-odds of default per
    one-standard-deviation increase in that feature (or, for the one-hot columns, versus
    the dropped reference level). ``odds_ratio`` above 1 means the feature increases the
    odds of default.
    """
    names = feature_names_from_pipeline(pipe)
    coefs = pipe.named_steps["clf"].coef_.ravel()
    out = pd.DataFrame(
        {
            "feature": names,
            "coefficient_log_odds": coefs,
            "odds_ratio": np.exp(coefs),
            "abs_coefficient": np.abs(coefs),
        }
    ).sort_values("abs_coefficient", ascending=False)
    out["direction"] = np.where(out["coefficient_log_odds"] > 0, "increases risk", "reduces risk")
    out = out.drop(columns="abs_coefficient").reset_index(drop=True)
    return out.head(top_n) if top_n else out


def permutation_importances(
    pipe, frame: pd.DataFrame, features: list[str], n_repeats: int = 10
) -> pd.DataFrame:
    """Drop in held-out ROC-AUC when each input is randomly permuted."""
    result = permutation_importance(
        pipe,
        frame[features],
        frame[C.TARGET],
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=C.RANDOM_STATE,
        n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "feature": features,
                "auc_drop_mean": result.importances_mean,
                "auc_drop_std": result.importances_std,
            }
        )
        .sort_values("auc_drop_mean", ascending=False)
        .reset_index(drop=True)
    )


def shap_values_for_tree_model(pipe, frame: pd.DataFrame, features: list[str], sample: int = 2000):
    """SHAP values for a tree model wrapped in a preprocessing pipeline.

    The features are transformed first and the explainer is pointed at the bare
    classifier, which is what keeps the attributions on the model's own input space.
    """
    if not HAS_SHAP:
        raise ImportError("shap is not installed")
    rng = np.random.default_rng(C.RANDOM_STATE)
    idx = rng.choice(len(frame), size=min(sample, len(frame)), replace=False)
    sub = frame.iloc[idx]

    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]
    X = prep.transform(sub[features])
    names = list(prep.get_feature_names_out())
    X_df = pd.DataFrame(X, columns=names, index=sub.index)

    explainer = shap.TreeExplainer(clf)
    values = explainer.shap_values(X_df)
    if isinstance(values, list):  # some versions return one array per class
        values = values[1]
    if values.ndim == 3:
        values = values[:, :, 1]
    return values, X_df, sub, explainer


def shap_global_importance(values: np.ndarray, X_df: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute SHAP value per feature, plus the average signed direction."""
    return (
        pd.DataFrame(
            {
                "feature": X_df.columns,
                "mean_abs_shap": np.abs(values).mean(axis=0),
                "mean_shap": values.mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def reason_codes(
    values: np.ndarray, X_df: pd.DataFrame, row: int, top_n: int = 5
) -> pd.DataFrame:
    """The top drivers behind one account's score - an adverse-action-style explanation."""
    contrib = pd.DataFrame(
        {
            "feature": X_df.columns,
            "feature_value": X_df.iloc[row].to_numpy(),
            "shap_value": values[row],
        }
    )
    contrib["abs"] = contrib["shap_value"].abs()
    out = contrib.sort_values("abs", ascending=False).head(top_n).drop(columns="abs")
    out["effect"] = np.where(out["shap_value"] > 0, "raises PD", "lowers PD")
    return out.reset_index(drop=True)


def stability_check(pipe, splits: dict, features: list[str]) -> pd.DataFrame:
    """ROC-AUC on train, validation and test - the simplest over-fitting diagnostic."""
    rows = []
    for name, frame in splits.items():
        p = pipe.predict_proba(frame[features])[:, 1]
        rows.append(
            {
                "split": name,
                "accounts": int(len(frame)),
                "roc_auc": float(roc_auc_score(frame[C.TARGET], p)),
            }
        )
    return pd.DataFrame(rows)


def population_stability_index(
    expected: np.ndarray, actual: np.ndarray, n_bins: int = 10
) -> float:
    """PSI between two score distributions.

    The standard monitoring trigger in a retail credit shop: below 0.10 is stable,
    0.10-0.25 warrants investigation, above 0.25 usually means the model is being applied
    to a population it was not built on.
    """
    cuts = np.quantile(expected, np.linspace(0, 1, n_bins + 1))
    cuts[0], cuts[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=cuts)[0] / len(expected)
    a = np.histogram(actual, bins=cuts)[0] / len(actual)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))
