import random
import warnings
import zlib

import numpy as np
from scipy import stats
from sklearn.exceptions import ConvergenceWarning

from src.distillation.dataset import GenotypeDataset


def configure_convergence_warnings(verbose: bool) -> None:
    """Hide convergence warnings for quiet runs and expose them for verbose runs.

    The filter is configured before any gene-level threads are started, since
    Python's warnings filters are process-global rather than thread-local.
    """
    warnings.filterwarnings(
        "default" if verbose else "ignore",
        category=ConvergenceWarning,
    )


def safe_pearson(a, b) -> float:
    """
    Pearson correlation that never triggers numpy's divide-by-zero warnings.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 2:
        return float("nan")
    a = a[mask]; b = b[mask]
    a = a - a.mean(); b = b - b.mean()
    denom = np.sqrt(float(a @ a) * float(b @ b))
    if not np.isfinite(denom) or denom <= 1e-12:
        return float("nan")
    return float((a @ b) / denom)


def safe_spearman(a, b) -> float:
    """
    Spearman rank correlation, implemented as the Pearson correlation of the ranks
    (via `safe_pearson`) so it inherits the same divide-by-zero-safe, degenerate-input
    behavior (NaN whenever there are fewer than 2 finite paired observations, or
    either input is constant after ranking).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 2:
        return float("nan")
    a = a[mask]; b = b[mask]
    return safe_pearson(stats.rankdata(a), stats.rankdata(b))


def pearson_pvalue(r, n) -> np.ndarray:
    """
    Two-sided p-value for a Pearson correlation coefficient `r` computed from `n`
    paired observations, via the standard t-test (equivalent to what
    ``scipy.stats.pearsonr`` returns as its p-value, but computable from a
    precomputed `r` and `n` alone, without the underlying samples).

    Vectorized over `r`/`n` (broadcastable array-likes or scalars). Returns NaN
    wherever `r` is NaN/undefined or `n` gives fewer than 1 degree of freedom
    (n < 3); a perfect |r| == 1 correlation is reported as p = 0.
    """
    r = np.atleast_1d(np.asarray(r, dtype=np.float64))
    n = np.atleast_1d(np.asarray(n, dtype=np.float64))
    dof = n - 2

    with np.errstate(invalid="ignore", divide="ignore"):
        t = r * np.sqrt(dof) / np.sqrt(1.0 - r ** 2)
        p = 2.0 * stats.t.sf(np.abs(t), dof)

    p = np.where(np.abs(r) >= 1.0, 0.0, p)
    p = np.where((dof < 1) | ~np.isfinite(r), np.nan, p)
    return p


def marginal_abs_corr(
    X: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """
    Absolute marginal (univariate) correlation of every column of `X` with `y`.

    Computed with a single matrix-vector product on the *raw* `X`: centering `y`
    makes centering `X` unnecessary (``(X - mean).T @ y_centered == X.T @
    y_centered``), so the whole matrix never has to be copied. Constant columns
    score 0.

    With `weights` (per-row, non-negative), all three moments become their
    weighted counterparts, so screening scores rows the same way the weighted
    fit that follows will. The unweighted path is left untouched, since it is
    the one every non-weighted model takes on a full ~50k-column cis-window.
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=X.dtype if X.dtype == np.float32 else np.float64)

    if weights is None:
        y_centered = y - y.mean()
        num        = np.abs(X.T @ y_centered)
        sd         = X.std(axis=0)
    else:
        w       = np.asarray(weights, dtype=np.float64)
        w_total = float(w.sum())
        if not np.isfinite(w_total) or w_total <= 0:
            return np.zeros(X.shape[1], dtype=np.float64)
        y_centered = y - float(w @ y) / w_total
        # The numerator is left unscaled, exactly as in the branch above, so that
        # uniform weights reproduce the unweighted scores rather than a rescaled
        # copy of them. Only the ranking matters either way, since any constant
        # applies to every column alike.
        num = np.abs(X.T @ (w * y_centered))
        # Weighted per-column variance as E_w[x^2] - E_w[x]^2. Squaring X costs
        # one temporary of X's size, the same order as the copy `np.std` makes.
        mean_x  = (w @ X) / w_total
        mean_x2 = (w @ np.square(X)) / w_total
        sd      = np.sqrt(np.clip(mean_x2 - mean_x ** 2, 0.0, None))

    scores     = np.zeros(X.shape[1], dtype=np.float64)
    usable     = sd > 0
    scores[usable] = num[usable] / sd[usable]
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


def screen_snps(
    X: np.ndarray,
    targets: np.ndarray | list[np.ndarray],
    k: int | None,
    weights: np.ndarray | None = None,
) -> np.ndarray | None:
    """
    Marginal-correlation screening: pick the `k` columns of `X` most correlated
    with `targets`, so the elastic net solves an n x k problem instead of an
    n x p one. In a cis-window p is ~40-50k while n is a few hundred, and the
    fit selects only a few hundred SNPs, so all but the most promising columns
    cost coordinate-descent time without changing the solution much.

    With several `targets` (the mean / aleatoric / epistemic heads, which must
    share one column set because they share `snp_ids` and one X scaler), a
    column is scored by its *strongest* correlation across targets, so no head
    can be starved of the SNPs it needs.

    Pass `weights` (see `marginal_abs_corr`) to score columns under the same
    per-row weighting the subsequent fit uses, so the prefilter cannot favour
    SNPs that only look predictive on rows the loss barely counts.

    IMPORTANT: screen on training rows only when the resulting fit is going to
    be scored on held-out rows, otherwise the test fold leaks into the feature
    selection.

    Returns sorted column indices, or None when no screening applies (`k` unset
    or not smaller than the number of columns), meaning "keep every column".
    """
    if k is None or k <= 0:
        return None
    n_snps = X.shape[1]
    if n_snps <= k:
        return None

    if isinstance(targets, np.ndarray) and targets.ndim == 1:
        targets = [targets]

    scores = None
    for target in targets:
        target_scores = marginal_abs_corr(X, target, weights=weights)
        scores = target_scores if scores is None else np.maximum(scores, target_scores)

    top = np.argpartition(-scores, k - 1)[:k]
    return np.sort(top)


def train_test_indices(
    n: int,
    seed: int = 42,
    test_frac: float = 0.2,
    key: str | None = None,
    min_train: int = 3,
    min_test: int = 2,
):
    """
    Reproducible per-gene split of `n` individuals into (train_idx, test_idx).

    A stable per-gene offset (crc32 of `key`) is folded into the seed so that each
    gene gets its own split, yet the whole run stays deterministic across processes
    and parallel workers (crc32 is stable, unlike the randomized built-in `hash`).

    Returns two sorted index arrays, or ``(None, None)`` when `n` is too small to
    hold out a meaningful test fold (fewer than `min_train + min_test` individuals).
    Callers should then fall back to an in-sample fit and report held-out metrics as
    NaN so tiny-sample genes don't masquerade as generalization estimates.
    """
    if n is None or n < min_train + min_test:
        return None, None
    offset = 0 if key is None else zlib.crc32(str(key).encode("utf-8")) % 100000
    rng    = np.random.default_rng(seed + offset)
    perm   = rng.permutation(n)
    n_test = int(round(n * test_frac))
    n_test = max(min_test, min(n_test, n - min_train))
    test_idx  = np.sort(perm[:n_test])
    train_idx = np.sort(perm[n_test:])
    return train_idx, test_idx

# taken from scPrediXcan tutorial
# https://github.com/hakyimlab/scPrediXcan/blob/master/Scripts/ctPred/Tutorial.ipynb
all_chromosomes = ["1", "10", "13", "15", "16", "17", "18", "19", "2", "21", "22", "3", "4", "6", "8", "9", "X", "Y"] + ["11", "14", "7"] + ["12", "20", "5"]

def get_train_test_dataset(dataset: GenotypeDataset, seed: int = 42):
    """Load dataset and split into train, val and test sets."""
    # chromosomes split into 3 parts, with 18, 3 and 3 chromosomes respectively
    random.seed(seed)
    random.shuffle(all_chromosomes)
    train_set = dataset.split_by_chromosome(all_chromosomes[:18])
    val_set   = dataset.split_by_chromosome(all_chromosomes[18:21])
    test_set  = dataset.split_by_chromosome(all_chromosomes[21:])
    return train_set, val_set, test_set


def ld_prune(
    X: np.ndarray,
    snp_ids: np.ndarray,
    threshold: float = 0.1,
    align: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """
    Simple greedy LD pruning based on pairwise column correlation.
    Keeps the first SNP, then removes later SNPs with r^2 >= threshold
    against any already-kept SNP.
    """

    snp_ids = np.atleast_1d(np.asarray(snp_ids))

    def _finish(X_, snp_ids_, keep_var_, keep_):
        if align is None:
            return X_, snp_ids_
        aligned = []
        for a in align:
            a = a[:, keep_var_]
            if keep_ is not None:
                a = a[:, keep_]
            aligned.append(a)
        return X_, snp_ids_, aligned

    _, n_snps = X.shape
    if n_snps <= 1:
        return _finish(X, snp_ids, slice(None), None)

    # remove zero-variance SNPs first
    var      = X.var(axis=0)
    keep_var = var > 1e-8
    X        = X[:, keep_var]
    snp_ids  = snp_ids[keep_var]

    n_snps = X.shape[1]
    if n_snps <= 1:
        return _finish(X, snp_ids, keep_var, None)

    keep = []

    # standardize once for correlation computation
    Xs = X.astype(np.float64, copy=False)
    Xs = (Xs - Xs.mean(axis=0)) / Xs.std(axis=0)

    for j in range(n_snps):
        if not keep:
            keep.append(j)
            continue

        # corr(current, already-kept) because columns are standardized. Only the
        # kept columns may be compared against: including every column would
        # also include column `j` itself, whose r2 is 1 by construction, so the
        # test below could never pass and nothing past the first SNP would
        # survive pruning.
        r  = (Xs[:, keep].T @ Xs[:, j]) / Xs.shape[0]
        r2 = r ** 2

        if np.all(r2 < threshold):
            keep.append(j)

    keep = np.asarray(keep, dtype=int)
    return _finish(X[:, keep], snp_ids[keep], keep_var, keep)
