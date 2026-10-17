"""Statistical estimators used in the paper.

Every function is deliberately explicit about its convention, because the
paper's numbers are sensitive to the convention (see ``docs/VERIFICATION_REPORT.md``).

Conventions
-----------
* discrimination       : AUC by the rank/Mann-Whitney statistic
* AUC confidence interval : percentile bootstrap (2000 replicates)
* DeLong               : fast DeLong (Sun & Xu) for correlated ROC curves
* continuous NRI       : Pencina; event and non-event components
* IDI                  : (slope_new − slope_old) on the probability scale,
                          slope = mean(p | event) − mean(p | non-event)
* Hosmer-Lemeshow      : g equal-frequency bins, statistic
                          sum (O−E)^2 / (E (1 − E/n)), df = bins − 2
* Cox recalibration    : logit(p*) = a + b·logit(p) fitted within the cohort
* decision curve       : net benefit = TP/n − FP/n · t/(1−t)
"""
from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "logit",
    "expit",
    "auc",
    "auc_ci",
    "delong",
    "delong_p",
    "continuous_nri",
    "idi",
    "hosmer_lemeshow",
    "hosmer_lemeshow_rank",
    "cox_recalibration",
    "net_benefit",
    "decision_curve",
    "brier",
    "youden_threshold",
    "binary_metrics",
    "paired_bootstrap",
    "bootstrap_ci",
    "kruskal_wallis",
]


# --------------------------------------------------------------------------
# link functions
# --------------------------------------------------------------------------
def logit(p, eps: float = 1e-9):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def expit(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


# --------------------------------------------------------------------------
# discrimination
# --------------------------------------------------------------------------
def auc(y, p) -> float:
    """AUC via the Mann-Whitney statistic (ties count 1/2)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pos, neg = p[y == 1], p[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (pos.size * neg.size))


def _midrank(x):
    """Average ranks with ties handled (1-based)."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    z = x[order]
    n = z.size
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(n, dtype=float)
    out[order] = t
    return out


def delong(y, *predictions):
    """Fast DeLong: returns ``(aucs, covariance)``.

    ``aucs[i]`` is the AUC of ``predictions[i]``; ``covariance`` is the
    estimated covariance matrix of the AUCs (required for correlated
    comparisons).
    """
    y = np.asarray(y).astype(int)
    P = np.vstack([np.asarray(p, dtype=float) for p in predictions])
    order = np.argsort(-y)
    y = y[order]
    P = P[:, order]
    m = int(y.sum())          # positives
    n = P.shape[1] - m        # negatives
    if m == 0 or n == 0:
        raise ValueError("DeLong requires both classes")
    k = P.shape[0]
    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        x = P[r]
        tx[r] = _midrank(x[:m])
        ty[r] = _midrank(x[m:])
        tz[r] = _midrank(x)
    aucs = np.array([(tz[r, :m].sum() / m - (m + 1) / 2.0) / n for r in range(k)])
    v01 = np.empty((k, m))
    v10 = np.empty((k, n))
    for r in range(k):
        v01[r] = (tz[r, :m] - tx[r]) / n
        v10[r] = 1.0 - (tz[r, m:] - ty[r]) / m
    sx = np.cov(v01) if k > 1 else np.array([[np.var(v01, ddof=1)]])
    sy = np.cov(v10) if k > 1 else np.array([[np.var(v10, ddof=1)]])
    return aucs, np.atleast_2d(sx) / m + np.atleast_2d(sy) / n


def delong_p(y, p1, p2):
    """Two-sided DeLong P value for AUC(p1) − AUC(p2)."""
    aucs, cov = delong(y, p1, p2)
    contrast = np.array([1.0, -1.0])
    var = float(contrast @ cov @ contrast)
    if var <= 0:
        return float("nan"), float(aucs[0] - aucs[1])
    z = float(contrast @ aucs) / np.sqrt(var)
    return float(2.0 * (1.0 - stats.norm.cdf(abs(z)))), float(aucs[0] - aucs[1])


# --------------------------------------------------------------------------
# reclassification
# --------------------------------------------------------------------------
def continuous_nri(y, p_new, p_old):
    """Pencina continuous NRI, with its two components."""
    y = np.asarray(y).astype(int)
    p_new = np.asarray(p_new, dtype=float)
    p_old = np.asarray(p_old, dtype=float)
    ev, ne = y == 1, y == 0
    events = float((p_new[ev] > p_old[ev]).mean() - (p_new[ev] < p_old[ev]).mean())
    non = float((p_new[ne] < p_old[ne]).mean() - (p_new[ne] > p_old[ne]).mean())
    return events + non, events, non


def idi(y, p_new, p_old):
    """Integrated discrimination improvement on the probability scale."""
    y = np.asarray(y).astype(int)
    p_new = np.asarray(p_new, dtype=float)
    p_old = np.asarray(p_old, dtype=float)

    def slope(p):
        return float(p[y == 1].mean() - p[y == 0].mean())

    s_new, s_old = slope(p_new), slope(p_old)
    return s_new - s_old, s_new, s_old


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------
def _hl_statistic(y, p, g):
    """Hosmer-Lemeshow statistic with ``g`` equal-frequency bins.

    Binning uses ``pandas.qcut`` (duplicates dropped), which is the convention
    used for every calibration number in the paper.  The statistic is
    ``sum (O-E)^2 / (E (1 - E/n))`` and the reference distribution is
    chi-square with ``bins - 2`` degrees of freedom.
    """
    import pandas as pd

    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    d = pd.DataFrame({"y": y, "p": p, "b": pd.qcut(p, g, duplicates="drop")})
    gg = d.groupby("b", observed=True).agg(o=("y", "sum"), e=("p", "sum"), n=("y", "size"))
    stat = float((((gg["o"] - gg["e"]) ** 2) / (gg["e"] * (1.0 - gg["e"] / gg["n"]))).sum())
    return stat, len(gg)


def hosmer_lemeshow(y, p, g: int = 10):
    """Hosmer-Lemeshow P value, equal-frequency bins, df = bins − 2.

    Bins are formed with ``pandas.qcut`` (i.e. quantile edges).  For Table S9 the
    paper's convention is :func:`hosmer_lemeshow_rank` instead, which matters
    whenever the score column contains many ties (the clinical score is rounded
    to two decimals in the training cohort and has only ~10 distinct values).
    """
    stat, nbins = _hl_statistic(y, p, g)
    dof = max(nbins - 2, 1)
    return float(1.0 - stats.chi2.cdf(stat, dof))


def hosmer_lemeshow_rank(y, p, g: int = 10):
    """Hosmer-Lemeshow P value with **rank-split** equal-frequency bins.

    The scores are sorted and cut into ``g`` groups of (nearly) equal size; ties
    are split by sort position rather than being kept together.  This is the
    convention behind Table S9 of the manuscript
    (``scripts/compute_hl_clinical.py`` in the authors' archive).
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="mergesort")
    p, y = p[order], y[order]
    n = len(p)
    edges = [int(round(i * n / g)) for i in range(g + 1)]
    stat, used = 0.0, 0
    for b in range(g):
        i0, i1 = edges[b], edges[b + 1]
        if i1 <= i0:
            continue
        gp, gy = p[i0:i1], y[i0:i1]
        m = len(gp)
        o, e = gy.sum(), gp.sum()
        if e <= 0 or e >= m:
            continue
        stat += (o - e) ** 2 / (e * (1.0 - e / m))
        used += 1
    return float(1.0 - stats.chi2.cdf(stat, max(used - 2, 1)))


def cox_recalibration(y, p):
    """Within-cohort Cox (logistic) recalibration.

    Returns ``(intercept, slope, p_recalibrated)`` with
    ``logit(p*) = intercept + slope * logit(p)``.
    """
    from sklearn.linear_model import LogisticRegression

    z = logit(p).reshape(-1, 1)
    lr = LogisticRegression(max_iter=4000, penalty=None).fit(z, y)
    a = float(lr.intercept_[0])
    b = float(lr.coef_[0][0])
    return a, b, expit(a + b * logit(p))


def brier(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


# --------------------------------------------------------------------------
# decision analysis
# --------------------------------------------------------------------------
def net_benefit(y, p, threshold: float) -> float:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    n = y.size
    pos = p >= threshold
    tp = int(((y == 1) & pos).sum())
    fp = int(((y == 0) & pos).sum())
    return tp / n - fp / n * (threshold / (1.0 - threshold))


def decision_curve(y, p, thresholds):
    y = np.asarray(y).astype(int)
    n = y.size
    n1 = int((y == 1).sum())
    n0 = n - n1
    rows = []
    for t in thresholds:
        pos = np.asarray(p) >= t
        tp = int(((y == 1) & pos).sum())
        fp = int(((y == 0) & pos).sum())
        rows.append(
            dict(
                threshold=float(t),
                treat_all=int(pos.sum()),
                treat_none=0,
                eplnd=int(pos.sum()),
                unnecessary_eplnd=fp,
                avoided_unnecessary=fp,
                missed=int(n1 - tp),
                sensitivity=tp / n1 if n1 else float("nan"),
                specificity=fp / n0 if n0 else float("nan"),
                net_benefit=net_benefit(y, p, t),
                net_benefit_all=n1 / n - n0 / n * (t / (1 - t)),
            )
        )
    return rows


# --------------------------------------------------------------------------
# threshold-based metrics
# --------------------------------------------------------------------------
def youden_threshold(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    thresholds = np.unique(p)
    best = (0.0, -np.inf)
    for t in thresholds:
        pos = p >= t
        sens = ((y == 1) & pos).sum() / max((y == 1).sum(), 1)
        spec = ((y == 0) & ~pos).sum() / max((y == 0).sum(), 1)
        j = sens + spec - 1.0
        if j > best[1]:
            best = (float(t), float(j))
    return best[0]


def binary_metrics(y, p, threshold=None) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if threshold is None:
        threshold = youden_threshold(y, p)
    pos = p >= threshold
    tp = int(((y == 1) & pos).sum())
    fn = int(((y == 1) & ~pos).sum())
    tn = int(((y == 0) & ~pos).sum())
    fp = int(((y == 0) & pos).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    npv = tn / (tn + fn) if tn + fn else float("nan")
    f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) else float("nan")
    return dict(
        threshold=float(threshold),
        accuracy=(tp + tn) / len(y),
        sensitivity=sens,
        specificity=spec,
        ppv=ppv,
        npv=npv,
        precision=ppv,
        recall=sens,
        f1=f1,
    )


# --------------------------------------------------------------------------
# resampling
# --------------------------------------------------------------------------
def bootstrap_ci(y, p, n_boot: int = 2000, seed: int = 20261016, alpha: float = 0.05):
    """Percentile bootstrap CI for a single AUC."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    n = y.size
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if np.unique(y[idx]).size < 2:
            continue
        vals.append(auc(y[idx], p[idx]))
    vals = np.asarray(vals)
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2))), vals


def patient_bootstrap_ci(values, n_boot: int = 2000, seed: int = 20261016, alpha: float = 0.05):
    """Percentile bootstrap CI for an arbitrary paired per-patient statistic.

    ``values`` is a 1-D array of n per-patient contributions; the bootstrap
    resamples patients and averages.
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    n = values.size
    draws = rng.integers(0, n, size=(n_boot, n))
    means = values[draws].mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def paired_bootstrap(y, p_new, p_old, n_boot: int = 3000, seed: int = 20261016):
    """Paired bootstrap of ΔAUC (percentile CI + two-sided P)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    a = np.asarray(p_new, dtype=float)
    b = np.asarray(p_old, dtype=float)
    n = y.size
    d = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if np.unique(y[idx]).size < 2:
            continue
        d.append(auc(y[idx], a[idx]) - auc(y[idx], b[idx]))
    d = np.asarray(d)
    p = max(2.0 * min((d <= 0).mean(), (d >= 0).mean()), 1.0 / n_boot)
    return (
        float(np.percentile(d, 2.5)),
        float(np.percentile(d, 97.5)),
        float(p),
        d,
    )


def nri_ci(y, p_new, p_old, n_boot: int = 2000, seed: int = 20261016):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    a = np.asarray(p_new, dtype=float)
    b = np.asarray(p_old, dtype=float)
    n = y.size
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if np.unique(y[idx]).size < 2:
            continue
        vals.append(continuous_nri(y[idx], a[idx], b[idx])[0])
    vals = np.asarray(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), vals


def kruskal_wallis(groups):
    groups = [np.asarray(g, dtype=float) for g in groups if len(g)]
    h, p = stats.kruskal(*groups)
    return float(h), float(p)

