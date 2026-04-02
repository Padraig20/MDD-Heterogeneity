import random
import numpy as np

from src.distillation.dataset import GenotypeDataset

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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple greedy LD pruning based on pairwise column correlation.
    Keeps the first SNP, then removes later SNPs with r^2 >= threshold
    against any already-kept SNP.
    """

    snp_ids = np.atleast_1d(np.asarray(snp_ids))

    _, n_snps = X.shape
    if n_snps <= 1:
        return X, snp_ids

    # remove zero-variance SNPs first
    var      = X.var(axis=0)
    keep_var = var > 1e-8
    X        = X[:, keep_var]
    snp_ids  = snp_ids[keep_var]

    n_snps = X.shape[1]
    if n_snps <= 1:
        return X, snp_ids

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
    return X[:, keep], snp_ids[keep]