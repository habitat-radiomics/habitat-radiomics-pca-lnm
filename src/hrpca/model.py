"""The frozen combined model and the small fitting helpers shared by the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["CombinedModel", "fit_logistic", "predict_logistic", "COLUMN_ALIASES",
           "resolve_column"]

# column names used by the different exports for the same model input
COLUMN_ALIASES = {
    "Clinic": ["Clinic", "pc"],
    "HabitatH2": ["HabitatH2", "H2", "p2"],
    "HabitatH4": ["HabitatH4", "H4", "p4"],
}


def resolve_column(df, feature: str) -> str:
    """Return the first alias of *feature* that exists in *df*."""
    for cand in COLUMN_ALIASES.get(feature, [feature]):
        if cand in df.columns:
            return cand
    raise KeyError(f"none of {COLUMN_ALIASES.get(feature, [feature])} found in the table")

# Frozen before prospective enrolment began (see Methods 2.1).
_DEFAULT_FEATURES = ("Clinic", "HabitatH2", "HabitatH4")
_DEFAULT_COEF = (4.062608661920751, 2.593122775462074, 3.9343891733424328)
_DEFAULT_INTERCEPT = -4.248788185663511


def fit_logistic(X, y, penalty=None, max_iter: int = 4000, random_state: int = 0):
    """Unpenalised (or L2/L1) logistic regression on a DataFrame or array."""
    from sklearn.linear_model import LogisticRegression

    Xv = X.values if hasattr(X, "values") else np.asarray(X, dtype=float)
    return LogisticRegression(penalty=penalty, max_iter=max_iter,
                              random_state=random_state, solver="lbfgs"
                              if penalty in (None, "l2") else "liblinear").fit(Xv, np.asarray(y))


def predict_logistic(model, X) -> np.ndarray:
    Xv = X.values if hasattr(X, "values") else np.asarray(X, dtype=float)
    return model.predict_proba(Xv)[:, 1]


@dataclass
class CombinedModel:
    """``logit(p) = intercept + sum(beta_i * x_i)`` on the **probability** scale.

    The study's combined model is linear in the per-patient probabilities of the
    clinical model and of habitats H2 and H4 — not in their log-odds.  That is
    what makes the closed-form SHAP attribution ``beta_i (x_i - E[x_i])`` exact.
    """

    features: tuple = field(default_factory=lambda: tuple(_DEFAULT_FEATURES))
    coef: tuple = field(default_factory=lambda: tuple(_DEFAULT_COEF))
    intercept: float = _DEFAULT_INTERCEPT

    # -- construction ------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: dict) -> "CombinedModel":
        m = cfg["model"]["combined"]
        return cls(features=tuple(m["features"]), coef=tuple(m["coef"]),
                   intercept=float(m["intercept"]))

    @classmethod
    def from_csv(cls, path) -> "CombinedModel":
        import pandas as pd

        d = pd.read_csv(path, encoding="utf-8-sig")
        return cls(features=tuple(d["feature"]), coef=tuple(d["coef"].astype(float)),
                   intercept=float(d["intercept"].iloc[0]))

    # -- prediction --------------------------------------------------------
    def raw_logit(self, df) -> np.ndarray:
        """Linear predictor f(x) (log-odds), clipped inputs.

        Missing canonical column names are resolved through :data:`COLUMN_ALIASES`,
        so that ``data/shap/meta_all.csv`` (columns ``pc``, ``p2``, ``p4``) can be
        used directly.
        """
        z = np.full(len(df), float(self.intercept))
        for f, b in zip(self.features, self.coef):
            col = resolve_column(df, f)
            z = z + b * np.asarray(df[col].values, dtype=float)
        return z

    def predict(self, df) -> np.ndarray:
        """Predicted probability of pN1."""
        return 1.0 / (1.0 + np.exp(-self.raw_logit(df)))

    # -- exact (closed-form) SHAP -----------------------------------------
    def shap_values(self, df, baseline=None):
        """Exact Shapley values for the linear model: ``beta_i (x_i - E[x_i])``.

        ``baseline`` is the reference value ``E[x_i]``; it may be

        * ``None`` — the mean of *df* itself (correct only when *df* is the whole
          reference cohort, e.g. the 398 training patients);
        * a scalar — the shared baseline ``E[f(X)]`` (``data/shap/base.csv``);
        * an array of length ``len(features)`` — per-input reference values.

        Returns ``(shap_matrix, baseline_vector)`` with one column per input.
        """
        X = np.column_stack([np.asarray(df[resolve_column(df, f)].values, dtype=float)
                             for f in self.features])
        base = self.resolve_baseline(X, baseline)
        shap = (X - base) * np.asarray(self.coef, dtype=float)
        return shap, base

    @staticmethod
    def resolve_baseline(X: np.ndarray, baseline) -> np.ndarray:
        if baseline is None:
            return X.mean(axis=0)
        arr = np.asarray(baseline, dtype=float)
        if arr.ndim == 0:
            return np.full(X.shape[1], float(arr))
        if arr.size == X.shape[1]:
            return arr
        # a scalar per input was requested but a single number was implied
        return np.full(X.shape[1], float(arr.ravel()[0]))

    def mean_abs_shap(self, df, baseline=None) -> dict:
        shap, _ = self.shap_values(df, baseline)
        return {f: float(np.abs(shap[:, i]).mean()) for i, f in enumerate(self.features)}
