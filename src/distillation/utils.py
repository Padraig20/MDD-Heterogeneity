import random
import zlib

import numpy as np

from src.distillation.dataset import GenotypeDataset


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

        # corr(current, all kept) because columns are standardized
        r = (Xs.T @ Xs[:, j]) / Xs.shape[0]
        r2 = r ** 2

        if np.all(r2 < threshold):
            keep.append(j)

    keep = np.asarray(keep, dtype=int)
    return _finish(X[:, keep], snp_ids[keep], keep_var, keep)