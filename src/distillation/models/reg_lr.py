"""
reg_lr.py

Uncertainty-weighted distillation of the MLP deep ensemble teacher into a single
point-estimate elastic net (or ridge), following the variance-weighted moment
regression of scTWAS (Nat. Commun. 2026, s41467-026-70374-7).

scTWAS minimizes a weighted least-squares loss with an elastic-net penalty,

    sum_i w_i * (x_i - s_i * g_i' beta)^2 + lambda * penalty(beta),

using w_i = 1 / Var(x_i) so that individuals whose measured expression is noisy
pull the fit around less than clean ones. They need an IRLS loop because their
variance is a function of the unknown beta; we do not, because the teacher hands
us its own decomposition Var(y_i) = aleatoric_i + epistemic_i up front. That
makes this a single-pass version of the same estimator.

Unlike `ProbabilisticLR`, which distills the teacher's uncertainty into two extra
regression heads, `RegLR` consumes the uncertainty *only at fit time*: what comes
out is an ordinary linear model with the same struct, the same predictions and
the same coefficient JSON as `LR`, so it is a drop-in replacement downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import r2_score

from src.distillation.dataset import GenotypeDataset
from src.distillation.models.lr import LR, LRStruct, fitted_alpha, fitted_l1_ratio
from src.distillation.utils import (
    ld_prune,
    safe_pearson,
    safe_spearman,
    screen_snps,
    train_test_indices,
)

# Absolute variance floor, used only when a gene's teacher variances are all zero
# and there is therefore no scale to derive a relative floor from.
ABSOLUTE_VARIANCE_FLOOR = 1e-12


def uncertainty_weights(
    y_aleatoric: np.ndarray,
    y_epistemic: np.ndarray,
    *,
    power: float = 1.0,
    lambda_aleatoric: float = 1.0,
    lambda_epistemic: float = 1.0,
    eps_frac: float = 1e-3,
    clip_quantile: float = 0.05,
) -> Optional[np.ndarray]:
    """
    Per-individual regression weights from the teacher's uncertainty, following
    scTWAS's w_i = 1 / Var(x_i):

        v_i = lambda_aleatoric * aleatoric_i + lambda_epistemic * epistemic_i
        w_i = v_i ** (-power)

    normalized to mean 1. `power=1` is scTWAS's exact inverse-variance weighting;
    lower values temper it (0.5 weights by inverse std), and `power=0` turns the
    weighting off entirely.

    The two lambdas let each uncertainty source be weighted separately, which
    matters because they are not interchangeable: aleatoric variance is
    measurement/biological noise in the label (exactly what scTWAS downweights),
    while epistemic variance is the teacher's own ignorance and is a function of
    the genotype. Weighting hard on the latter can systematically downweight the
    individuals with unusual genotypes, i.e. the ones carrying most of the signal
    for rare variants, so `lambda_epistemic` is worth ablating.

    Numerical safety, in order:
      * variances are clamped at 0, then floored at `eps_frac * median(v[v > 0])`,
        so an individual the teacher happens to be near-certain about cannot take
        an unbounded weight;
      * the resulting weights are clipped to their `[clip_quantile, 1 -
        clip_quantile]` empirical quantiles, so a handful of extreme individuals
        cannot end up being the only ones the fit sees.

    Returns None whenever the weighting would be inert (no power, no lambdas, no
    finite spread in the variances, or fewer than two rows). Callers pass that
    straight through as `sample_weight=None`, which puts the fit on exactly the
    same code path as the unweighted `LR` -- so `power=0` is an exact ablation
    baseline rather than an approximate one.
    """
    if power == 0 or (lambda_aleatoric == 0 and lambda_epistemic == 0):
        return None

    aleatoric = np.clip(np.asarray(y_aleatoric, dtype=np.float64), 0.0, None)
    epistemic = np.clip(np.asarray(y_epistemic, dtype=np.float64), 0.0, None)
    variance  = lambda_aleatoric * aleatoric + lambda_epistemic * epistemic
    if variance.size < 2 or not np.all(np.isfinite(variance)):
        return None

    positive = variance[variance > 0]
    if positive.size == 0:
        return None
    floor    = max(eps_frac * float(np.median(positive)), ABSOLUTE_VARIANCE_FLOOR)
    variance = np.maximum(variance, floor)

    weights = variance ** (-float(power))
    if clip_quantile and clip_quantile > 0:
        lo, hi  = np.quantile(weights, [clip_quantile, 1.0 - clip_quantile])
        weights = np.clip(weights, lo, hi)

    mean = float(weights.mean())
    if not np.isfinite(mean) or mean <= 0:
        return None
    weights /= mean

    # Constant weights are what `sample_weight=None` already means, and saying so
    # explicitly keeps the degenerate case off the weighted code path.
    if np.allclose(weights, 1.0):
        return None
    return weights


def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, weights: Optional[np.ndarray]) -> float:
    """R^2 under the same per-individual weighting the model was fit with."""
    if weights is None:
        return float(r2_score(y_true, y_pred)) if np.size(y_true) > 1 else float("nan")

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    w      = np.asarray(weights, dtype=np.float64)
    total  = float(w.sum())
    if y_true.size < 2 or total <= 0:
        return float("nan")

    residual = float(w @ (y_true - y_pred) ** 2)
    variance = float(w @ (y_true - float(w @ y_true) / total) ** 2)
    if variance <= 1e-12:
        return float("nan")
    return 1.0 - residual / variance


def weighted_pearson(a: np.ndarray, b: np.ndarray, weights: Optional[np.ndarray]) -> float:
    """Pearson r under the same per-individual weighting the model was fit with."""
    if weights is None:
        return safe_pearson(a, b)

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)

    mask = np.isfinite(a) & np.isfinite(b) & np.isfinite(w)
    if int(mask.sum()) < 2:
        return float("nan")
    a, b, w = a[mask], b[mask], w[mask]

    total = float(w.sum())
    if total <= 0:
        return float("nan")
    a = a - float(w @ a) / total
    b = b - float(w @ b) / total
    denom = np.sqrt(float(w @ (a ** 2)) * float(w @ (b ** 2)))
    if not np.isfinite(denom) or denom <= 1e-12:
        return float("nan")
    return float((w @ (a * b)) / denom)


@dataclass
class RegLRStruct(LRStruct):
    """
    An `LRStruct` plus a record of how hard the uncertainty weighting actually
    pushed on this gene, so a run can be read as "did the weighting change
    anything, and did it help" rather than just "what came out".
    """
    # max/min weight (1.0 when unweighted) and their coefficient of variation:
    # how uneven the teacher's uncertainty was across this gene's individuals.
    weight_ratio_: float = 1.0
    weight_cv_:    float = 0.0
    # Mean teacher total predictive std, sqrt(aleatoric + epistemic), for context
    # on how noisy the gene's labels were in the first place.
    mean_target_std_: Optional[float] = None
    # Held-out metrics under the weighted loss, i.e. the objective actually being
    # optimized. The unweighted `heldout_r2_`/`heldout_pearson_r_` inherited above
    # stay the comparison metric against `LR`.
    weighted_heldout_r2_:        Optional[float] = None
    weighted_heldout_pearson_r_: Optional[float] = None


class RegLR(LR):
    """
    Elastic net / ridge fit with a per-individual inverse-uncertainty weighted
    squared-error loss (see `uncertainty_weights`), distilling the teacher's mean
    prediction while letting its uncertainty decide how much each individual
    counts.
    """

    def __init__(
        self,
        model_name: str = "elasticnet",
        l1_ratio: float = 0.5,
        cv: int         = 3,
        alphas: int     = 100,
        max_iter: int   = 10000,
        seed: int       = 42,
        n_jobs: int     = 1,
        screen: Optional[int] = 5000,
        weight_power: float             = 1.0,
        weight_lambda_aleatoric: float  = 1.0,
        weight_lambda_epistemic: float  = 1.0,
        weight_eps_frac: float          = 1e-3,
        weight_clip_quantile: float     = 0.05,
    ):
        super().__init__(
            model_name=model_name, l1_ratio=l1_ratio, cv=cv, alphas=alphas,
            max_iter=max_iter, seed=seed, n_jobs=n_jobs, screen=screen,
        )
        self.weight_power            = weight_power
        self.weight_lambda_aleatoric = weight_lambda_aleatoric
        self.weight_lambda_epistemic = weight_lambda_epistemic
        self.weight_eps_frac         = weight_eps_frac
        self.weight_clip_quantile    = weight_clip_quantile
        self.models_: Dict[str, RegLRStruct] = {}

    def _weights(self, y_aleatoric: np.ndarray, y_epistemic: np.ndarray) -> Optional[np.ndarray]:
        """This model's configured weighting, applied to one set of rows."""
        return uncertainty_weights(
            y_aleatoric,
            y_epistemic,
            power=self.weight_power,
            lambda_aleatoric=self.weight_lambda_aleatoric,
            lambda_epistemic=self.weight_lambda_epistemic,
            eps_frac=self.weight_eps_frac,
            clip_quantile=self.weight_clip_quantile,
        )

    @staticmethod
    def _weight_stats(weights: Optional[np.ndarray]) -> tuple[float, float]:
        """(max/min ratio, coefficient of variation) of a weight vector."""
        if weights is None or weights.size == 0:
            return 1.0, 0.0
        smallest = float(weights.min())
        ratio    = float(weights.max()) / smallest if smallest > 0 else float("inf")
        mean     = float(weights.mean())
        cv       = float(weights.std() / mean) if mean > 0 else float("nan")
        return ratio, cv

    def fit_gene_matrix(
        self,
        gene: str,
        X: np.ndarray,
        y: np.ndarray,
        y_aleatoric: np.ndarray,
        y_epistemic: np.ndarray,
        snp_ids: np.ndarray,
        chr: int,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        y_aleatoric_test: Optional[np.ndarray] = None,
        y_epistemic_test: Optional[np.ndarray] = None,
    ) -> RegLRStruct:
        X           = np.asarray(X, dtype=np.float32)
        y           = np.asarray(y, dtype=np.float32)
        y_aleatoric = np.asarray(y_aleatoric, dtype=np.float32)
        y_epistemic = np.asarray(y_epistemic, dtype=np.float32)

        # An individual without a usable uncertainty has no usable weight either,
        # so the same rows are dropped here as in `ProbabilisticLR`.
        valid = np.isfinite(y) & np.isfinite(y_aleatoric) & np.isfinite(y_epistemic)
        if not valid.all():
            X, y = X[valid], y[valid]
            y_aleatoric, y_epistemic = y_aleatoric[valid], y_epistemic[valid]

        if y.size == 0:
            raise ValueError("No non-missing target values available")
        if X.shape[1] == 0:
            raise ValueError("No SNPs in cis-window for this gene")

        mean_target_std = float(
            np.mean(np.sqrt(np.clip(y_aleatoric, 0.0, None) + np.clip(y_epistemic, 0.0, None)))
        )

        has_external_test = (
            X_test is not None and y_test is not None
            and y_aleatoric_test is not None and y_epistemic_test is not None
        )
        if has_external_test:
            X_test           = np.asarray(X_test, dtype=np.float32)
            y_test           = np.asarray(y_test, dtype=np.float32)
            y_aleatoric_test = np.asarray(y_aleatoric_test, dtype=np.float32)
            y_epistemic_test = np.asarray(y_epistemic_test, dtype=np.float32)
            valid_test = (
                np.isfinite(y_test) & np.isfinite(y_aleatoric_test) & np.isfinite(y_epistemic_test)
            )
            if not valid_test.all():
                X_test, y_test   = X_test[valid_test], y_test[valid_test]
                y_aleatoric_test = y_aleatoric_test[valid_test]
                y_epistemic_test = y_epistemic_test[valid_test]

        if self.model_name == "ridge":
            # perform LD pruning (dropping the same SNP columns from the external
            # test matrix, if any, so train/test stay column-aligned)
            if has_external_test:
                X, snp_ids, (X_test,) = ld_prune(X, snp_ids, align=[X_test])
            else:
                X, snp_ids = ld_prune(X, snp_ids)

        # The persisted model, once it is known that fitting it again would
        # reproduce a fit already computed for the held-out metrics.
        final          = None
        reuse_alpha    = None
        weights_final  = None

        if has_external_test:
            # Held-out evaluation on a user-supplied, disjoint test set: train on
            # *all* individuals from `X`/`y` (no internal split) and score on
            # `X_test`/`y_test`.
            if y_test.size > 0:
                weights = self._weights(y_aleatoric, y_epistemic)
                keep    = screen_snps(X, y, self.screen, weights=weights)
                if keep is not None:
                    X       = X[:, keep]
                    X_test  = X_test[:, keep]
                    snp_ids = np.asarray(snp_ids)[keep]
                xs, ys, enet_h = self._fit_scaled(X, y, sample_weight=weights)
                pred_test   = self._predict_scaled(xs, ys, enet_h, X_test)
                pred_train  = self._predict_scaled(xs, ys, enet_h, X)
                heldout_r2  = float(r2_score(y_test, pred_test))
                insample_r2 = float(r2_score(y, pred_train))
                heldout_pearson_r  = safe_pearson(y_test, pred_test)
                insample_pearson_r = safe_pearson(y, pred_train)
                heldout_spearman_r  = safe_spearman(y_test, pred_test)
                insample_spearman_r = safe_spearman(y, pred_train)
                weights_test = self._weights(y_aleatoric_test, y_epistemic_test)
                weighted_heldout_r2 = weighted_r2(y_test, pred_test, weights_test)
                weighted_heldout_pearson_r = weighted_pearson(y_test, pred_test, weights_test)
                n_train, n_test = int(y.size), int(y_test.size)
                # This fit already saw every individual, so it *is* the model that
                # would be persisted; refitting it would repeat identical work.
                final         = (xs, ys, enet_h)
                weights_final = weights
            else:
                heldout_r2, insample_r2 = float("nan"), float("nan")
                heldout_pearson_r, insample_pearson_r = float("nan"), float("nan")
                heldout_spearman_r, insample_spearman_r = float("nan"), float("nan")
                weighted_heldout_r2, weighted_heldout_pearson_r = float("nan"), float("nan")
                n_train, n_test = int(y.size), 0
        else:
            # Held-out evaluation: fit on a per-gene 80% split of the individuals and
            # score on the remaining 20%, so the numbers stay comparable to `LR`'s
            # (also held-out) and expose overfitting for p >> n cis windows. The
            # saved coefficients below are refit on *all* individuals.
            train_idx, test_idx = train_test_indices(y.size, seed=self.seed, key=gene)
            if test_idx is not None:
                # Weights, like the screening below, come from the train fold alone:
                # their floor and clipping quantiles are data-dependent, so deriving
                # them from all rows would let the test fold shape the fit.
                weights = self._weights(y_aleatoric[train_idx], y_epistemic[train_idx])
                keep = screen_snps(
                    X[train_idx], y[train_idx], self.screen, weights=weights
                )
                X_h  = X if keep is None else X[:, keep]
                xs, ys, enet_h = self._fit_scaled(
                    X_h[train_idx], y[train_idx], sample_weight=weights
                )
                pred_test   = self._predict_scaled(xs, ys, enet_h, X_h[test_idx])
                pred_train  = self._predict_scaled(xs, ys, enet_h, X_h[train_idx])
                heldout_r2  = float(r2_score(y[test_idx], pred_test))
                insample_r2 = float(r2_score(y[train_idx], pred_train))
                heldout_pearson_r  = safe_pearson(y[test_idx], pred_test)
                insample_pearson_r = safe_pearson(y[train_idx], pred_train)
                heldout_spearman_r  = safe_spearman(y[test_idx], pred_test)
                insample_spearman_r = safe_spearman(y[train_idx], pred_train)
                weights_test = self._weights(y_aleatoric[test_idx], y_epistemic[test_idx])
                weighted_heldout_r2 = weighted_r2(y[test_idx], pred_test, weights_test)
                weighted_heldout_pearson_r = weighted_pearson(y[test_idx], pred_test, weights_test)
                n_train, n_test = int(train_idx.size), int(test_idx.size)
                reuse_alpha = fitted_alpha(enet_h)
            else:
                # too few individuals to hold out: no honest generalization estimate
                heldout_r2, insample_r2 = float("nan"), float("nan")
                heldout_pearson_r, insample_pearson_r = float("nan"), float("nan")
                heldout_spearman_r, insample_spearman_r = float("nan"), float("nan")
                weighted_heldout_r2, weighted_heldout_pearson_r = float("nan"), float("nan")
                n_train, n_test = int(y.size), 0

        # Final model refit on all individuals -> these are the persisted coefficients.
        # The penalty selected on the train fold is carried over instead of searching
        # the alpha path a second time.
        if final is None:
            weights_final = self._weights(y_aleatoric, y_epistemic)
            keep = screen_snps(X, y, self.screen, weights=weights_final)
            if keep is not None:
                X       = X[:, keep]
                snp_ids = np.asarray(snp_ids)[keep]
            final = self._fit_scaled(
                X, y, alpha=reuse_alpha, sample_weight=weights_final
            )
        x_scaler, y_scaler, enet = final
        weight_ratio, weight_cv  = self._weight_stats(weights_final)

        model = RegLRStruct(
            model_name=self.model_name,
            gene=gene,
            chr=chr,
            snp_ids=np.asarray(snp_ids),
            coef_=enet.coef_.copy(),
            intercept_=enet.intercept_,
            alpha_=fitted_alpha(enet),
            l1_ratio_=fitted_l1_ratio(enet),
            x_mean_=x_scaler.mean_.copy(),
            x_scale_=x_scaler.scale_.copy(),
            y_mean_=y_scaler.mean_[0],
            y_scale_=y_scaler.scale_[0],
            heldout_r2_=heldout_r2,
            insample_r2_=insample_r2,
            n_train_=n_train,
            n_test_=n_test,
            heldout_pearson_r_=heldout_pearson_r,
            insample_pearson_r_=insample_pearson_r,
            heldout_spearman_r_=heldout_spearman_r,
            insample_spearman_r_=insample_spearman_r,
            weight_ratio_=weight_ratio,
            weight_cv_=weight_cv,
            mean_target_std_=mean_target_std,
            weighted_heldout_r2_=weighted_heldout_r2,
            weighted_heldout_pearson_r_=weighted_heldout_pearson_r,
        )
        self.models_[gene] = model
        return model

    @staticmethod
    def _empty_test_targets(X: np.ndarray, y: np.ndarray) -> tuple:
        """Zero-row stand-ins for a gene the external test set doesn't carry."""
        return (
            np.empty((0, X.shape[1]), dtype=X.dtype),
            np.empty(0, dtype=y.dtype),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    def fit_gene_from_dataset(
        self,
        dataset: GenotypeDataset,
        gene: str,
        test_dataset: Optional[GenotypeDataset] = None,
    ) -> RegLRStruct:
        X, y, y_aleatoric, y_epistemic, snp_ids, chr = dataset.get_gene_matrix(
            gene, return_uncertainty=True
        )
        X_test, y_test, y_aleatoric_test, y_epistemic_test = None, None, None, None
        if test_dataset is not None:
            try:
                X_test, y_test, y_aleatoric_test, y_epistemic_test, _, _ = (
                    test_dataset.get_gene_matrix(gene, return_uncertainty=True)
                )
            except ValueError:
                # Gene has no rows in the external test set; train on all of `X`/`y`
                # but report held-out metrics as NaN (see fit_gene_matrix).
                X_test, y_test, y_aleatoric_test, y_epistemic_test = (
                    self._empty_test_targets(X, y)
                )
        return self.fit_gene_matrix(
            gene, X, y, y_aleatoric, y_epistemic, snp_ids, chr,
            X_test=X_test, y_test=y_test,
            y_aleatoric_test=y_aleatoric_test, y_epistemic_test=y_epistemic_test,
        )

    def fit_gene_from_design(
        self,
        dataset: GenotypeDataset,
        gene: str,
        design: tuple,
        test_dataset: Optional[GenotypeDataset] = None,
    ) -> RegLRStruct:
        """
        Fit one gene against a design matrix read elsewhere (see
        `GenotypeDataset.gene_design`), so cell types sharing a cohort can share one
        genotype read per gene instead of repeating it.
        """
        X, snp_ids, chr, individuals = design
        y, y_aleatoric, y_epistemic  = dataset.gene_targets(gene, individuals)

        X_test, y_test, y_aleatoric_test, y_epistemic_test = None, None, None, None
        if test_dataset is not None:
            try:
                X_test, y_test, y_aleatoric_test, y_epistemic_test, _, _ = (
                    test_dataset.get_gene_matrix(gene, return_uncertainty=True)
                )
            except ValueError:
                X_test, y_test, y_aleatoric_test, y_epistemic_test = (
                    self._empty_test_targets(X, y)
                )
        return self.fit_gene_matrix(
            gene, X, y, y_aleatoric, y_epistemic, snp_ids, chr,
            X_test=X_test, y_test=y_test,
            y_aleatoric_test=y_aleatoric_test, y_epistemic_test=y_epistemic_test,
        )

    def _fit_one(
        self,
        dataset: GenotypeDataset,
        gene: str,
        i: int,
        n: int,
        verbose: bool,
        test_dataset: Optional[GenotypeDataset] = None,
    ) -> Optional[RegLRStruct]:
        try:
            model = self.fit_gene_from_dataset(dataset, gene, test_dataset=test_dataset)
            if verbose:
                nnz = int(np.sum(model.coef_ != 0))
                print(
                    f"[{i}/{n}] fit {gene}: "
                    f"nonzero={nnz}, heldout_r2={model.heldout_r2_:.4f}, "
                    f"heldout_pearson_r={model.heldout_pearson_r_:.4f}, "
                    f"heldout_spearman_r={model.heldout_spearman_r_:.4f} "
                    f"(insample_r2={model.insample_r2_:.4f}, n_test={model.n_test_}), "
                    f"weighted_heldout_r2={model.weighted_heldout_r2_:.4f}, "
                    f"weight_ratio={model.weight_ratio_:.2f}, "
                    f"target_std={model.mean_target_std_:.3f}"
                )
            return model
        except Exception as e:
            if verbose:
                print(f"[{i}/{n}] skip {gene}: {e}")
            return None

    def fit_dataset(
        self,
        dataset: GenotypeDataset,
        verbose: bool = True,
        test_dataset: Optional[GenotypeDataset] = None,
    ) -> Dict[str, RegLRStruct]:
        if not getattr(dataset, "has_uncertainty", False):
            raise ValueError(
                "RegLR requires a dataset built with both aleatoric and epistemic "
                "targets, since they are what the per-individual loss weights are "
                "derived from. Pass --aleatoric and --epistemic to train.py."
            )
        if test_dataset is not None and not getattr(test_dataset, "has_uncertainty", False):
            raise ValueError(
                "RegLR requires the held-out test dataset to also carry aleatoric/"
                "epistemic targets; pass --aleatoric/--epistemic to train.py."
            )
        return super().fit_dataset(dataset, verbose=verbose, test_dataset=test_dataset)

    def _summary_row(self, gene: str, model: RegLRStruct) -> dict:
        row = super()._summary_row(gene, model)
        row.update(
            {
                # Held-out under the weighted objective actually optimized; `r2` and
                # `pearson_r` above stay unweighted, i.e. comparable to `LR`.
                "weighted_r2": model.weighted_heldout_r2_,
                "weighted_pearson_r": model.weighted_heldout_pearson_r_,
                "weight_ratio": model.weight_ratio_,
                "weight_cv": model.weight_cv_,
                "mean_target_std": model.mean_target_std_,
            }
        )
        return row
