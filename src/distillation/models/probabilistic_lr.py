"""
probabilistic_lr.py

Probabilistic distillation of the MLP deep ensemble teacher into three independent,
genuinely L1+L2-regularized elastic nets: one predicting the mean, one predicting the
teacher's *aleatoric* variance, and one predicting the teacher's *epistemic* variance.

"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from src.distillation.dataset import GenotypeDataset
from src.distillation.models.lr import LR, fitted_alpha, fitted_l1_ratio
from src.distillation.utils import (
    configure_convergence_warnings,
    ld_prune,
    safe_pearson,
    safe_spearman,
    screen_snps,
    train_test_indices,
)


@dataclass
class LinearHead:
    """
    One independently-fit ElasticNet/Ridge head (mean, aleatoric, or epistemic), in
    standardized X / standardized (possibly log1p-transformed) y space.
    """
    coef_:      np.ndarray
    intercept_: float
    alpha_:     float
    l1_ratio_:  Optional[float]
    y_mean_:    float
    y_scale_:   float
    # If True, this head was fit on log1p(target) (used for the variance heads, to
    # guarantee a non-negative prediction and avoid heavy tails dominating the loss).
    # Predictions are inverted via expm1 and clipped at 0 as a numerical safety net.
    log1p: bool

    def predict(self, X_scaled: np.ndarray) -> np.ndarray:
        y_scaled = self.intercept_ + X_scaled @ self.coef_
        y = self.y_mean_ + self.y_scale_ * y_scaled
        if self.log1p:
            y = np.clip(np.expm1(y), 0.0, None)
        return y


@dataclass
class ProbabilisticLRStruct:
    model_name: str
    gene:       str
    chr:        int
    snp_ids:    np.ndarray

    mean_:      LinearHead
    aleatoric_: LinearHead
    epistemic_: LinearHead

    # X scaling is shared across all three heads (they're all fit on the same X).
    x_mean_:  np.ndarray
    x_scale_: np.ndarray

    # Diagnostics (native log-expression space). All distribution-matching metrics are
    # evaluated on HELD-OUT individuals (per-gene 20% split); the heads that produce
    # them are trained only on the 80% train fold, while the persisted heads above are
    # refit on all individuals.
    train_r2_:           Optional[float] = None  # held-out R^2 of the mean prediction
    insample_r2_:        Optional[float] = None  # in-sample R^2 on the train fold (overfitting gap)
    n_train_:            int = 0
    n_test_:             int = 0
    pearson_r_:          Optional[float] = None  # held-out Pearson r of the mean (bounded [-1, 1])
    insample_pearson_r_: Optional[float] = None  # in-sample Pearson r on the train fold
    spearman_r_:          Optional[float] = None  # held-out Spearman r of the mean (rank-based, bounded [-1, 1])
    insample_spearman_r_: Optional[float] = None  # in-sample Spearman r on the train fold
    train_w2_:           Optional[float] = None  # held-out mean 2-Wasserstein distance (mean + std parts)
    mean_w2_:            Optional[float] = None  # mean-matching part of the Wasserstein
    std_w2_:             Optional[float] = None  # std/variance-matching part (variance-fit error, lower is better)
    std_corr_:           Optional[float] = None  # corr(pred std, target std) across individuals (higher is better)
    std_ratio_:          Optional[float] = None  # mean(pred std) / mean(target std) (calibration, ~1 is ideal)
    mean_pred_std_:      Optional[float] = None  # avg predicted total std (aleatoric + epistemic)
    mean_target_std_:    Optional[float] = None  # avg target std (teacher's aleatoric + epistemic)
    diverged_:           bool = False            # True if the per-gene fit blew up (metrics are NaN)

    # Per-source breakdown of the same diagnostics, so aleatoric and epistemic
    # uncertainty can each be judged on their own (not just via their sum above).
    aleatoric_w2_:         Optional[float] = None  # mean((pred_aleatoric_std - target_aleatoric_std)^2)
    aleatoric_std_corr_:   Optional[float] = None  # within-gene corr(pred, target) aleatoric std, across individuals
    aleatoric_std_ratio_:  Optional[float] = None  # mean(pred_aleatoric_std) / mean(target_aleatoric_std)
    aleatoric_pred_std_:   Optional[float] = None  # avg predicted aleatoric std
    aleatoric_target_std_: Optional[float] = None  # avg target aleatoric std
    epistemic_w2_:         Optional[float] = None  # mean((pred_epistemic_std - target_epistemic_std)^2)
    epistemic_std_corr_:   Optional[float] = None  # within-gene corr(pred, target) epistemic std, across individuals
    epistemic_std_ratio_:  Optional[float] = None  # mean(pred_epistemic_std) / mean(target_epistemic_std)
    epistemic_pred_std_:   Optional[float] = None  # avg predicted epistemic std
    epistemic_target_std_: Optional[float] = None  # avg target epistemic std


class ProbabilisticLR:
    """
    Fits, per gene, three independent elastic-net (or ridge) regressions -- mean,
    aleatoric variance, epistemic variance -- directly against the teacher's own
    uncertainty decomposition.
    """

    def __init__(
        self,
        model_name: str      = "elasticnet",
        l1_ratio: float      = 0.5,
        cv: int              = 3,
        alphas: int          = 100,
        max_iter: int        = 10000,
        seed: int            = 42,
        n_jobs: int          = 1,
        screen: Optional[int] = 5000,
    ):
        self.model_name = model_name
        self.l1_ratio   = l1_ratio
        self.cv         = cv
        self.alphas     = alphas
        self.max_iter   = max_iter
        self.seed       = seed
        self.n_jobs     = n_jobs
        self.screen     = screen
        # A single shared `LR` instance whose `_scale_x`/`_fit_prescaled` helpers
        # are reused for all three heads: identical ElasticNet/Ridge construction
        # (alpha path, solver settings, ...), just called with a different target
        # each time. Never calls `LR.fit_gene_matrix` itself, since that bundles its
        # own held-out split + persisted-model refit for a single target.
        self._lr = LR(
            model_name=model_name, l1_ratio=l1_ratio, cv=cv, alphas=alphas,
            max_iter=max_iter, seed=seed, screen=screen,
        )
        self.models_: Dict[str, ProbabilisticLRStruct] = {}

    def _fit_heads(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_aleatoric_log: np.ndarray,
        y_epistemic_log: np.ndarray,
        alphas: Optional[dict] = None,
    ) -> dict:
        """
        Fit the mean/aleatoric/epistemic heads, all on one shared standardization of
        `X`. The three heads differ only in their target, so standardizing X once and
        reusing it avoids building three identical (n x p) copies per gene.

        Pass `alphas` (head name -> penalty) to refit at penalties selected earlier
        instead of searching the alpha path again for each head.
        """
        x_scaler, X_scaled = self._lr._scale_x(X)
        heads = {}
        for name, target, log1p in (
            ("mean",      y,               False),
            ("aleatoric", y_aleatoric_log, True),
            ("epistemic", y_epistemic_log, True),
        ):
            alpha          = None if alphas is None else alphas.get(name)
            y_scaler, enet = self._lr._fit_prescaled(X_scaled, target, alpha=alpha)
            head = LinearHead(
                coef_=enet.coef_.copy(),
                intercept_=float(enet.intercept_),
                alpha_=fitted_alpha(enet),
                l1_ratio_=fitted_l1_ratio(enet),
                y_mean_=float(y_scaler.mean_[0]),
                y_scale_=float(y_scaler.scale_[0]),
                log1p=log1p,
            )
            heads[name] = (head, x_scaler, y_scaler)
        return heads

    @staticmethod
    def _head_alphas(heads: dict) -> dict:
        """Penalty selected by each head, for refitting without a second search."""
        return {name: head.alpha_ for name, (head, _, _) in heads.items()}

    @staticmethod
    def _nan_metrics(y_aleatoric: np.ndarray, y_epistemic: np.ndarray) -> dict:
        aleatoric_std = np.sqrt(np.clip(y_aleatoric, 0.0, None))
        epistemic_std = np.sqrt(np.clip(y_epistemic, 0.0, None))
        target_std    = np.sqrt(np.clip(y_aleatoric, 0.0, None) + np.clip(y_epistemic, 0.0, None))
        nan = float("nan")
        return {
            "r2": nan, "pearson_r": nan, "spearman_r": nan, "w2": nan, "mean_w2": nan, "std_w2": nan,
            "std_corr": nan, "std_ratio": nan, "pred_std": nan,
            "target_std": float(np.mean(target_std)) if target_std.size else nan,
            "aleatoric_w2": nan, "aleatoric_std_corr": nan, "aleatoric_std_ratio": nan,
            "aleatoric_pred_std": nan,
            "aleatoric_target_std": float(np.mean(aleatoric_std)) if aleatoric_std.size else nan,
            "epistemic_w2": nan, "epistemic_std_corr": nan, "epistemic_std_ratio": nan,
            "epistemic_pred_std": nan,
            "epistemic_target_std": float(np.mean(epistemic_std)) if epistemic_std.size else nan,
            "diverged": False,
        }

    def _evaluate(
        self,
        heads: dict,
        X: np.ndarray,
        y: np.ndarray,
        y_aleatoric: np.ndarray,
        y_epistemic: np.ndarray,
    ) -> dict:
        """
        Run fitted (mean, aleatoric, epistemic) heads on rows (X, y, y_aleatoric,
        y_epistemic) and return native-space distribution-matching metrics: both the
        combined total predictive std (comparable to the teacher's own aleatoric +
        epistemic total) *and* the same set of diagnostics for each uncertainty
        source individually, so aleatoric and epistemic can each be judged on their
        own rather than only via their sum.
        """
        mean_head,      x_scaler, _ = heads["mean"]
        aleatoric_head, _,        _ = heads["aleatoric"]
        epistemic_head, _,        _ = heads["epistemic"]

        # All three heads were fit against one shared standardization of X (see
        # `_fit_heads`), so it only has to be applied once here too.
        X_scaled            = x_scaler.transform(X)
        mu                  = mean_head.predict(X_scaled)
        aleatoric_pred      = aleatoric_head.predict(X_scaled)
        epistemic_pred      = epistemic_head.predict(X_scaled)
        aleatoric_std_pred  = np.sqrt(aleatoric_pred)
        epistemic_std_pred  = np.sqrt(epistemic_pred)
        total_std           = np.sqrt(aleatoric_pred + epistemic_pred)

        aleatoric_var_native  = np.clip(y_aleatoric, 0.0, None)
        epistemic_var_native  = np.clip(y_epistemic, 0.0, None)
        aleatoric_std_native  = np.sqrt(aleatoric_var_native)
        epistemic_std_native  = np.sqrt(epistemic_var_native)
        target_std_native     = np.sqrt(aleatoric_var_native + epistemic_var_native)

        # Detect a diverged / failed fit. Native log-expression means are a few units
        # and target stds are <~0.3, so predictions many orders of magnitude larger mean
        # the per-gene optimisation blew up; such R^2 / W2 are meaningless -> NaN
        DIVERGE_ABS = 1e3
        diverged = (
            not np.all(np.isfinite(mu))
            or not np.all(np.isfinite(total_std))
            or (mu.size > 0 and np.max(np.abs(mu)) > DIVERGE_ABS)
            or (total_std.size > 0 and np.max(total_std) > DIVERGE_ABS)
        )

        nan = float("nan")
        metrics = {
            "r2": nan, "pearson_r": nan, "spearman_r": nan, "w2": nan, "mean_w2": nan, "std_w2": nan,
            "std_corr": nan, "std_ratio": nan, "pred_std": nan,
            "target_std": float(np.mean(target_std_native)) if target_std_native.size else nan,
            "aleatoric_w2": nan, "aleatoric_std_corr": nan, "aleatoric_std_ratio": nan,
            "aleatoric_pred_std": nan,
            "aleatoric_target_std": float(np.mean(aleatoric_std_native)) if aleatoric_std_native.size else nan,
            "epistemic_w2": nan, "epistemic_std_corr": nan, "epistemic_std_ratio": nan,
            "epistemic_pred_std": nan,
            "epistemic_target_std": float(np.mean(epistemic_std_native)) if epistemic_std_native.size else nan,
            "diverged": bool(diverged),
        }
        if diverged or y.size <= 1:
            return metrics

        metrics["r2"]         = float(r2_score(y, mu))
        metrics["pearson_r"]  = safe_pearson(mu, y)
        metrics["spearman_r"] = safe_spearman(mu, y)
        # Wasserstein split into its mean and std (variance) contributions
        metrics["mean_w2"]   = float(np.mean((mu - y) ** 2))
        metrics["std_w2"]    = float(np.mean((total_std - target_std_native) ** 2))
        metrics["w2"]        = metrics["mean_w2"] + metrics["std_w2"]
        # How well does the predicted per-individual std track the target std?
        metrics["std_corr"]  = safe_pearson(total_std, target_std_native)
        denom                = float(np.mean(target_std_native))
        metrics["std_ratio"] = float(np.mean(total_std) / denom) if denom > 1e-12 else nan
        metrics["pred_std"]  = float(np.mean(total_std))

        # Per-source breakdown: same diagnostics, computed separately for the
        # aleatoric and epistemic heads (not just their combined total above).
        metrics["aleatoric_w2"]       = float(np.mean((aleatoric_std_pred - aleatoric_std_native) ** 2))
        metrics["aleatoric_std_corr"] = safe_pearson(aleatoric_std_pred, aleatoric_std_native)
        denom_al                      = float(np.mean(aleatoric_std_native))
        metrics["aleatoric_std_ratio"] = (
            float(np.mean(aleatoric_std_pred) / denom_al) if denom_al > 1e-12 else nan
        )
        metrics["aleatoric_pred_std"] = float(np.mean(aleatoric_std_pred))

        metrics["epistemic_w2"]       = float(np.mean((epistemic_std_pred - epistemic_std_native) ** 2))
        metrics["epistemic_std_corr"] = safe_pearson(epistemic_std_pred, epistemic_std_native)
        denom_ep                      = float(np.mean(epistemic_std_native))
        metrics["epistemic_std_ratio"] = (
            float(np.mean(epistemic_std_pred) / denom_ep) if denom_ep > 1e-12 else nan
        )
        metrics["epistemic_pred_std"] = float(np.mean(epistemic_std_pred))

        return metrics

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
    ) -> ProbabilisticLRStruct:
        X           = np.asarray(X, dtype=np.float32)
        y           = np.asarray(y, dtype=np.float32)
        y_aleatoric = np.asarray(y_aleatoric, dtype=np.float32)
        y_epistemic = np.asarray(y_epistemic, dtype=np.float32)

        # drop individuals with a missing mean or uncertainty target
        valid = np.isfinite(y) & np.isfinite(y_aleatoric) & np.isfinite(y_epistemic)
        if not valid.all():
            X, y, y_aleatoric, y_epistemic = X[valid], y[valid], y_aleatoric[valid], y_epistemic[valid]

        if y.size == 0:
            raise ValueError("No non-missing target values available")
        if X.shape[1] == 0:
            raise ValueError("No SNPs in cis-window for this gene")

        # teacher variances must be non-negative; clamp tiny negatives from noise.
        y_aleatoric = np.clip(y_aleatoric, 0.0, None)
        y_epistemic = np.clip(y_epistemic, 0.0, None)
        y_aleatoric_log = np.log1p(y_aleatoric)
        y_epistemic_log = np.log1p(y_epistemic)

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
                X_test           = X_test[valid_test]
                y_test           = y_test[valid_test]
                y_aleatoric_test = y_aleatoric_test[valid_test]
                y_epistemic_test = y_epistemic_test[valid_test]
            y_aleatoric_test = np.clip(y_aleatoric_test, 0.0, None)
            y_epistemic_test = np.clip(y_epistemic_test, 0.0, None)

        if self.model_name == "ridge":
            # perform LD pruning (dropping the same SNP columns from the external test
            # matrix, if any, so train/test stay column-aligned); shared across all
            # three heads since they all use the same X.
            if has_external_test:
                X, snp_ids, (X_test,) = ld_prune(X, snp_ids, align=[X_test])
            else:
                X, snp_ids = ld_prune(X, snp_ids)

        # The persisted heads, once it is known that fitting them again would
        # reproduce heads already computed for the held-out metrics.
        heads_final  = None
        reuse_alphas = None

        if has_external_test:
            # Held-out evaluation on a user-supplied, disjoint test set: train on
            # *all* individuals from `X`/`y`/uncertainty targets (no internal split)
            # and evaluate on the `_test` arrays.
            if y_test.size > 0:
                keep = screen_snps(
                    X, [y, y_aleatoric_log, y_epistemic_log], self.screen
                )
                if keep is not None:
                    X       = X[:, keep]
                    X_test  = X_test[:, keep]
                    snp_ids = np.asarray(snp_ids)[keep]
                heads_h     = self._fit_heads(X, y, y_aleatoric_log, y_epistemic_log)
                metrics     = self._evaluate(heads_h, X_test, y_test, y_aleatoric_test, y_epistemic_test)
                insample    = self._evaluate(heads_h, X, y, y_aleatoric, y_epistemic)
                insample_r2 = insample["r2"]
                insample_pearson_r = insample["pearson_r"]
                insample_spearman_r = insample["spearman_r"]
                n_train, n_test = int(y.size), int(y_test.size)
                # These heads already saw every individual, so they *are* the heads
                # that would be persisted; refitting would repeat identical work.
                heads_final = heads_h
            else:
                metrics = self._nan_metrics(y_aleatoric, y_epistemic)
                insample_r2 = float("nan")
                insample_pearson_r = float("nan")
                insample_spearman_r = float("nan")
                n_train, n_test = int(y.size), 0
        else:
            # Held-out evaluation: train on a per-gene 80% split of the individuals and
            # score every distribution-matching metric on the remaining 20%, so the
            # numbers are comparable to the point-estimate LR (also held-out) and expose
            # overfitting for p >> n cis windows. The persisted heads are refit on all
            # individuals afterwards.
            train_idx, test_idx = train_test_indices(y.size, seed=self.seed, key=gene)
            if test_idx is not None:
                # Screened on the train fold alone: screening on all rows would let
                # the test fold influence which SNPs the scored heads may use.
                keep = screen_snps(
                    X[train_idx],
                    [y[train_idx], y_aleatoric_log[train_idx], y_epistemic_log[train_idx]],
                    self.screen,
                )
                X_h = X if keep is None else X[:, keep]
                heads_h = self._fit_heads(
                    X_h[train_idx], y[train_idx], y_aleatoric_log[train_idx], y_epistemic_log[train_idx]
                )
                metrics = self._evaluate(
                    heads_h, X_h[test_idx], y[test_idx], y_aleatoric[test_idx], y_epistemic[test_idx]
                )
                insample = self._evaluate(
                    heads_h, X_h[train_idx], y[train_idx], y_aleatoric[train_idx], y_epistemic[train_idx]
                )
                insample_r2 = insample["r2"]
                insample_pearson_r = insample["pearson_r"]
                insample_spearman_r = insample["spearman_r"]
                n_train, n_test = int(train_idx.size), int(test_idx.size)
                reuse_alphas = self._head_alphas(heads_h)
            else:
                # too few individuals to hold out: no honest generalization estimate
                metrics = self._nan_metrics(y_aleatoric, y_epistemic)
                insample_r2 = float("nan")
                insample_pearson_r = float("nan")
                insample_spearman_r = float("nan")
                n_train, n_test = int(y.size), 0

        # Final heads refit on all individuals -> these are the persisted coefficients.
        # The penalties selected on the train fold are carried over instead of
        # searching the alpha path a second time for each head.
        if heads_final is None:
            keep = screen_snps(
                X, [y, y_aleatoric_log, y_epistemic_log], self.screen
            )
            if keep is not None:
                X       = X[:, keep]
                snp_ids = np.asarray(snp_ids)[keep]
            heads_final = self._fit_heads(
                X, y, y_aleatoric_log, y_epistemic_log, alphas=reuse_alphas
            )
        mean_head,      x_scaler, _ = heads_final["mean"]
        aleatoric_head, _,        _ = heads_final["aleatoric"]
        epistemic_head, _,        _ = heads_final["epistemic"]

        struct = ProbabilisticLRStruct(
            model_name=self.model_name,
            gene=gene,
            chr=chr,
            snp_ids=np.asarray(snp_ids),
            mean_=mean_head,
            aleatoric_=aleatoric_head,
            epistemic_=epistemic_head,
            x_mean_=x_scaler.mean_.copy(),
            x_scale_=x_scaler.scale_.copy(),
            train_r2_=metrics["r2"],
            insample_r2_=insample_r2,
            n_train_=n_train,
            n_test_=n_test,
            pearson_r_=metrics["pearson_r"],
            insample_pearson_r_=insample_pearson_r,
            spearman_r_=metrics["spearman_r"],
            insample_spearman_r_=insample_spearman_r,
            train_w2_=metrics["w2"],
            mean_w2_=metrics["mean_w2"],
            std_w2_=metrics["std_w2"],
            std_corr_=metrics["std_corr"],
            std_ratio_=metrics["std_ratio"],
            mean_pred_std_=metrics["pred_std"],
            mean_target_std_=metrics["target_std"],
            diverged_=bool(metrics["diverged"]),
            aleatoric_w2_=metrics["aleatoric_w2"],
            aleatoric_std_corr_=metrics["aleatoric_std_corr"],
            aleatoric_std_ratio_=metrics["aleatoric_std_ratio"],
            aleatoric_pred_std_=metrics["aleatoric_pred_std"],
            aleatoric_target_std_=metrics["aleatoric_target_std"],
            epistemic_w2_=metrics["epistemic_w2"],
            epistemic_std_corr_=metrics["epistemic_std_corr"],
            epistemic_std_ratio_=metrics["epistemic_std_ratio"],
            epistemic_pred_std_=metrics["epistemic_pred_std"],
            epistemic_target_std_=metrics["epistemic_target_std"],
        )
        self.models_[gene] = struct
        return struct

    def fit_gene_from_dataset(
        self,
        dataset: GenotypeDataset,
        gene: str,
        test_dataset: Optional[GenotypeDataset] = None,
    ) -> ProbabilisticLRStruct:
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
                # Gene has no rows in the external test set; train on all of `X`/`y`/
                # uncertainty targets but report held-out metrics as NaN (see
                # fit_gene_matrix).
                X_test = np.empty((0, X.shape[1]), dtype=X.dtype)
                y_test = np.empty(0, dtype=y.dtype)
                y_aleatoric_test = np.empty(0, dtype=y_aleatoric.dtype)
                y_epistemic_test = np.empty(0, dtype=y_epistemic.dtype)
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
    ) -> ProbabilisticLRStruct:
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
                X_test = np.empty((0, X.shape[1]), dtype=X.dtype)
                y_test = np.empty(0, dtype=y.dtype)
                y_aleatoric_test = np.empty(0, dtype=y_aleatoric.dtype)
                y_epistemic_test = np.empty(0, dtype=y_epistemic.dtype)
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
    ) -> Optional[ProbabilisticLRStruct]:
        try:
            model = self.fit_gene_from_dataset(dataset, gene, test_dataset=test_dataset)
            if verbose:
                print(
                    f"[{i}/{n}] fit {gene}: "
                    f"heldout_r2={model.train_r2_:.4f}, heldout_pearson_r={model.pearson_r_:.4f}, "
                    f"heldout_spearman_r={model.spearman_r_:.4f} "
                    f"(insample_r2={model.insample_r2_:.4f}, "
                    f"insample_pearson_r={model.insample_pearson_r_:.4f}, "
                    f"insample_spearman_r={model.insample_spearman_r_:.4f}, n_test={model.n_test_}), "
                    f"W2={model.train_w2_:.4f}, "
                    f"pred_std={model.mean_pred_std_:.3f}, target_std={model.mean_target_std_:.3f}"
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
    ) -> Dict[str, ProbabilisticLRStruct]:
        if not getattr(dataset, "has_uncertainty", False):
            raise ValueError(
                "ProbabilisticLR requires a dataset built with both aleatoric and "
                "epistemic targets. Pass --aleatoric and --epistemic to train.py."
            )
        if test_dataset is not None and not getattr(test_dataset, "has_uncertainty", False):
            raise ValueError(
                "ProbabilisticLR requires the held-out test dataset to also carry "
                "aleatoric/epistemic targets; pass --aleatoric/--epistemic to train.py."
            )

        genes  = list(dataset.genes)
        n      = len(genes)
        n_jobs = max(1, int(self.n_jobs))

        configure_convergence_warnings(verbose)

        if n_jobs == 1:
            for i, gene in enumerate(genes, start=1):
                self._fit_one(dataset, gene, i, n, verbose, test_dataset=test_dataset)
            return self.models_

        # Parallel path: thread pool over genes with BLAS pinned to 1 thread per call.
        # sklearn's coordinate descent releases the GIL, so threads scale well, and
        # capping BLAS prevents N x M thread oversubscription.
        with threadpool_limits(limits=1):
            with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                futures = {
                    ex.submit(self._fit_one, dataset, gene, i, n, verbose, test_dataset): gene
                    for i, gene in enumerate(genes, start=1)
                }
                iterator = as_completed(futures)
                if not verbose:
                    iterator = tqdm(iterator, total=len(futures), desc="Fitting genes", leave=False)
                for fut in iterator:
                    fut.result()

        return self.models_

    def predict_gene_matrix(
        self, gene: str, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the full teacher-distilled Gaussian (mean, aleatoric_var,
        epistemic_var, total_var) in native (log-expression) space from the saved
        per-head elastic-net coefficients.
        """
        model    = self.models_[gene]
        X        = np.asarray(X, dtype=np.float64)
        X_scaled = (X - model.x_mean_) / model.x_scale_

        mean          = model.mean_.predict(X_scaled)
        aleatoric_var = model.aleatoric_.predict(X_scaled)
        epistemic_var = model.epistemic_.predict(X_scaled)
        total_var     = aleatoric_var + epistemic_var
        return mean, aleatoric_var, epistemic_var, total_var

    def summarize_models(self) -> pd.DataFrame:
        rows = []
        for gene, model in self.models_.items():
            rows.append(
                {
                    "gene": gene,
                    "r2": model.train_r2_,           # held-out (per-gene 20%) R^2 of the mean
                    "insample_r2": model.insample_r2_,
                    "n_train": model.n_train_,
                    "n_test": model.n_test_,
                    "pearson_r": model.pearson_r_,   # held-out, bounded [-1, 1] mean-fit correlation
                    "insample_pearson_r": model.insample_pearson_r_,
                    "spearman_r": model.spearman_r_,  # held-out, rank-based, bounded [-1, 1]
                    "insample_spearman_r": model.insample_spearman_r_,
                    "wasserstein": model.train_w2_,
                    "mean_w2": model.mean_w2_,
                    "std_w2": model.std_w2_,          # variance-fit error (lower is better)
                    "std_corr": model.std_corr_,      # rank/tracking of target uncertainty
                    "std_ratio": model.std_ratio_,    # calibration ratio (~1 is ideal)
                    "pred_std": model.mean_pred_std_,
                    "target_std": model.mean_target_std_,
                    "diverged": model.diverged_,
                    "nonzero_weights": int(np.sum(np.abs(model.mean_.coef_) > 1e-6)),
                    "mean_alpha": model.mean_.alpha_,
                    "aleatoric_alpha": model.aleatoric_.alpha_,
                    "epistemic_alpha": model.epistemic_.alpha_,
                    # Per-source breakdown (not just the combined total above), so
                    # aleatoric/epistemic uncertainty can each be judged on their own.
                    "aleatoric_w2": model.aleatoric_w2_,
                    "aleatoric_std_corr": model.aleatoric_std_corr_,
                    "aleatoric_std_ratio": model.aleatoric_std_ratio_,
                    "aleatoric_pred_std": model.aleatoric_pred_std_,
                    "aleatoric_target_std": model.aleatoric_target_std_,
                    "epistemic_w2": model.epistemic_w2_,
                    "epistemic_std_corr": model.epistemic_std_corr_,
                    "epistemic_std_ratio": model.epistemic_std_ratio_,
                    "epistemic_pred_std": model.epistemic_pred_std_,
                    "epistemic_target_std": model.epistemic_target_std_,
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # sort by Pearson r (bounded [-1, 1], so a more intuitive fit-quality
        # ranking than R^2, which is unbounded below); fall back to R^2 if
        # pearson_r is unavailable for some reason.
        sort_key = "pearson_r" if "pearson_r" in df.columns else "r2"
        df = df.sort_values(sort_key, ascending=True).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
        return df

    def save_coefficients(self, output_path: str) -> None:
        """
        Save the non-zero mean/aleatoric/epistemic elastic-net coefficients of each
        gene's model to a JSON file, following the same {"snp_ids", "coefs",
        "intercept"} convention as `LR.save_coefficients` (coefficients in
        standardized X / standardized y space; no scaler persistence). The
        aleatoric/epistemic heads additionally carry `"log1p": true`, since they were
        fit on log1p(variance) and must be inverted via expm1 (clipped at 0) at
        inference time.
        """
        def head_json(head: LinearHead, snp_ids: np.ndarray) -> dict:
            nonzero = head.coef_ != 0
            entry = {
                "snp_ids":   [str(s) for s in snp_ids[nonzero]],
                "coefs":     [float(c) for c in head.coef_[nonzero]],
                "intercept": float(head.intercept_),
            }
            if head.log1p:
                entry["log1p"] = True
            return entry

        output = {}
        for gene, model in self.models_.items():
            snp_ids = np.asarray(model.snp_ids)
            output[gene] = {
                "chr":       int(model.chr),
                "mean":      head_json(model.mean_, snp_ids),
                "aleatoric": head_json(model.aleatoric_, snp_ids),
                "epistemic": head_json(model.epistemic_, snp_ids),
            }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=4)
