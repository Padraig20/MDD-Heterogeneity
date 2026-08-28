"""Bayesian distillation of deep-ensemble members into SNP-weight posteriors.

Each teacher member is fitted independently with SparseVB under

    teacher_mean_i ~ N(intercept + genotype_i @ beta, teacher_sigma_i**2).

The heteroskedastic likelihood is converted to SparseVB's homoskedastic
likelihood by precision-weighted centering followed by row whitening.  SparseVB
then supplies a mean-field spike-and-slab posterior for every SNP.  The member
posteriors are retained separately and combined as an equally weighted mixture,
so the exported SNP-weight variance contains both within-member variational
uncertainty and between-member teacher disagreement.

``slab="gaussian"`` (the default requested for this model) is a Gaussian
spike-and-slab model.  It is sparse and serves the role of an elastic-net
student, but it is not literally the usual combined L1/L2 elastic-net prior.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sparsevb import svb_fit_linear
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from src.distillation.dataset import GenotypeDataset
from src.distillation.utils import (
    marginal_abs_corr,
    safe_pearson,
    safe_spearman,
    train_test_indices,
)


MIN_FEATURE_SCALE = 1e-12


def _nanmean(values: Sequence[float]) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    return float(values[finite].mean()) if finite.any() else float("nan")


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) < 2 or np.var(np.asarray(y_true)[mask]) <= 1e-12:
        return float("nan")
    return float(r2_score(np.asarray(y_true)[mask], np.asarray(y_pred)[mask]))


def spike_slab_moments(
    slab_mean: np.ndarray,
    slab_sd: np.ndarray,
    inclusion_probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return marginal mean/variance of a point-mass + slab posterior."""
    mu = np.asarray(slab_mean, dtype=np.float64)
    sigma = np.asarray(slab_sd, dtype=np.float64)
    gamma = np.asarray(inclusion_probability, dtype=np.float64)
    mean = gamma * mu
    variance = gamma * sigma**2 + gamma * (1.0 - gamma) * mu**2
    return mean, np.maximum(variance, 0.0)


def combine_member_moments(
    member_means: np.ndarray,
    member_variances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Combine equally weighted member posteriors by total variance."""
    means = np.asarray(member_means, dtype=np.float64)
    variances = np.asarray(member_variances, dtype=np.float64)
    if means.ndim < 1 or means.shape != variances.shape or means.shape[0] == 0:
        raise ValueError(
            "member_means and member_variances must have the same non-empty "
            "shape (n_members, ...)."
        )
    mean = means.mean(axis=0)
    within = variances.mean(axis=0)
    between = np.mean((means - mean) ** 2, axis=0)
    total = np.maximum(within + between, 0.0)
    return mean, total, within, between


@dataclass
class MemberPosterior:
    member_id: str
    slab_mean_: np.ndarray
    slab_sd_: np.ndarray
    inclusion_probability_: np.ndarray
    coef_: np.ndarray
    coef_var_: np.ndarray
    intercept_: float
    intercept_var_: float
    x_mean_: np.ndarray
    feature_scale_: np.ndarray
    sum_precision_: float
    effective_n_: float
    sigma_floor_fraction_: float


@dataclass
class EnsembleLRStruct:
    gene: str
    chr: str | int
    snp_ids: np.ndarray
    members_: tuple[MemberPosterior, ...]
    coef_: np.ndarray
    coef_var_: np.ndarray
    coef_within_var_: np.ndarray
    coef_between_var_: np.ndarray
    inclusion_probability_: np.ndarray
    intercept_: float
    intercept_var_: float
    heldout_r2_: float = float("nan")
    insample_r2_: float = float("nan")
    heldout_pearson_r_: float = float("nan")
    insample_pearson_r_: float = float("nan")
    heldout_spearman_r_: float = float("nan")
    insample_spearman_r_: float = float("nan")
    heldout_whitened_mse_: float = float("nan")
    insample_whitened_mse_: float = float("nan")
    heldout_gaussian_nll_: float = float("nan")
    insample_gaussian_nll_: float = float("nan")
    member_heldout_r2_: tuple[float, ...] = ()
    member_insample_r2_: tuple[float, ...] = ()
    n_train_: int = 0
    n_test_: int = 0

    @property
    def coef_sd_(self) -> np.ndarray:
        return np.sqrt(self.coef_var_)

    @property
    def intercept_sd_(self) -> float:
        return float(np.sqrt(max(self.intercept_var_, 0.0)))


class EnsembleGenotypeDataset:
    """Several aligned member datasets sharing one genotype design."""

    def __init__(
        self,
        members: Sequence[GenotypeDataset],
        member_ids: Optional[Sequence[str | int]] = None,
    ):
        self.members = tuple(members)
        if not self.members:
            raise ValueError("At least one ensemble-member dataset is required.")
        if not all(getattr(member, "has_sigma", False) for member in self.members):
            raise ValueError(
                "Every ensemble-member dataset must be constructed with y_sigma."
            )

        if member_ids is None:
            member_ids = [str(i) for i in range(len(self.members))]
        if len(member_ids) != len(self.members):
            raise ValueError("member_ids must have one entry per member dataset.")
        self.member_ids = tuple(str(member_id) for member_id in member_ids)
        if len(set(self.member_ids)) != len(self.member_ids):
            raise ValueError("member_ids must be unique.")

        reference = self.members[0]
        for member in self.members[1:]:
            if not reference.shares_individuals_with(member):
                raise ValueError(
                    "Ensemble members must contain identical individuals in the "
                    "same order and use the same genotype source."
                )

        common = set(str(gene) for gene in reference.genes)
        for member in self.members[1:]:
            common &= set(str(gene) for gene in member.genes)
        self.genes = np.asarray(
            [gene for gene in reference.genes if str(gene) in common]
        )
        if self.genes.size == 0:
            raise ValueError("Ensemble members have no genes in common.")

    def has_gene(self, gene: str) -> bool:
        return all(member.has_gene(gene) for member in self.members)

    def gene_design(self, gene: str):
        return self.members[0].gene_design(gene)

    def member_targets(
        self,
        gene: str,
        individuals: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        means = []
        sigmas = []
        for member in self.members:
            mean, _, _ = member.gene_targets(gene, individuals)
            means.append(mean)
            sigmas.append(member.gene_sigma(gene, individuals))
        return np.asarray(means, dtype=np.float64), np.asarray(sigmas, dtype=np.float64)

    def get_gene_matrix(self, gene: str):
        X, snp_ids, chrom, individuals = self.gene_design(gene)
        means, sigmas = self.member_targets(gene, individuals)
        return X, means, sigmas, snp_ids, chrom

    def shares_individuals_with(self, other) -> bool:
        return (
            isinstance(other, EnsembleGenotypeDataset)
            and self.member_ids == other.member_ids
            and self.members[0].shares_individuals_with(other.members[0])
        )


class EnsembleLR:
    """Fit one heteroskedastic SparseVB model per deep-ensemble member."""

    def __init__(
        self,
        *,
        max_iter: int = 2000,
        tol: float = 1e-5,
        slab: str = "gaussian",
        prior_scale: float = 1.0,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        sigma_floor: float = 1e-4,
        pip_threshold: float = 0.5,
        seed: int = 42,
        n_jobs: int = 1,
        screen: Optional[int] = 5000,
    ):
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be finite and positive.")
        if slab not in {"gaussian", "laplace"}:
            raise ValueError("slab must be 'gaussian' or 'laplace'.")
        if not np.isfinite(prior_scale) or prior_scale <= 0:
            raise ValueError("prior_scale must be finite and positive.")
        if not np.isfinite(sigma_floor) or sigma_floor <= 0:
            raise ValueError("sigma_floor must be finite and positive.")
        if not 0.0 <= pip_threshold <= 1.0:
            raise ValueError("pip_threshold must be in [0, 1].")
        if alpha is not None and (not np.isfinite(alpha) or alpha <= 0):
            raise ValueError("alpha must be finite and positive when supplied.")
        if beta is not None and (not np.isfinite(beta) or beta <= 0):
            raise ValueError("beta must be finite and positive when supplied.")

        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.slab = slab
        self.prior_scale = float(prior_scale)
        self.alpha = alpha
        self.beta = beta
        self.sigma_floor = float(sigma_floor)
        self.pip_threshold = float(pip_threshold)
        self.seed = int(seed)
        self.n_jobs = max(1, int(n_jobs))
        self.screen = screen
        self.models_: Dict[str, EnsembleLRStruct] = {}

    def _fit_member(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sigma: np.ndarray,
        member_id: str,
    ) -> MemberPosterior:
        raw_sigma = np.asarray(sigma, dtype=np.float64)
        sigma = np.maximum(raw_sigma, self.sigma_floor)
        precision = sigma**-2
        sum_precision = float(precision.sum())
        if not np.isfinite(sum_precision) or sum_precision <= 0:
            raise ValueError(f"Member {member_id} has no finite positive precision.")

        x_mean = (precision @ X) / sum_precision
        y_mean = float(precision @ y / sum_precision)
        X_white = (X - x_mean) / sigma[:, None]
        y_white = (y - y_mean) / sigma

        # Scale the already centered/whitened columns to RMS 1. This preserves
        # the heteroskedastic likelihood while putting the common slab prior on
        # comparable coordinates. Constant columns were removed before this call.
        feature_scale = np.sqrt(np.mean(X_white**2, axis=0))
        if np.any(~np.isfinite(feature_scale) | (feature_scale <= MIN_FEATURE_SCALE)):
            raise ValueError(f"Member {member_id} has an unusable SNP column.")
        X_fit = X_white / feature_scale

        result = svb_fit_linear(
            X_fit,
            y_white,
            max_iter=self.max_iter,
            tol=self.tol,
            slab=self.slab,
            intercept=False,
            alpha=self.alpha,
            beta=self.beta,
            prior_scale=self.prior_scale,
            noise_sd=1.0,
        )
        slab_mean = np.asarray(result["mu"], dtype=np.float64) / feature_scale
        slab_sd = np.asarray(result["sigma"], dtype=np.float64) / feature_scale
        gamma = np.asarray(result["gamma"], dtype=np.float64)
        if (
            slab_mean.shape != (X.shape[1],)
            or slab_sd.shape != slab_mean.shape
            or gamma.shape != slab_mean.shape
            or not np.all(np.isfinite(slab_mean))
            or not np.all(np.isfinite(slab_sd))
            or not np.all(np.isfinite(gamma))
            or np.any(slab_sd < 0)
            or np.any((gamma < 0) | (gamma > 1))
        ):
            raise RuntimeError(
                f"SparseVB returned invalid posterior parameters for member {member_id}."
            )

        coef, coef_var = spike_slab_moments(slab_mean, slab_sd, gamma)
        intercept = float(y_mean - x_mean @ coef)
        # With a flat intercept prior, its conditional variance contributes
        # 1/sum(precision); coefficient uncertainty adds the centering term.
        intercept_var = float(1.0 / sum_precision + np.sum(x_mean**2 * coef_var))
        effective_n = float(sum_precision**2 / np.sum(precision**2))
        floor_fraction = float(np.mean(raw_sigma < self.sigma_floor))

        return MemberPosterior(
            member_id=str(member_id),
            slab_mean_=slab_mean,
            slab_sd_=slab_sd,
            inclusion_probability_=gamma,
            coef_=coef,
            coef_var_=coef_var,
            intercept_=intercept,
            intercept_var_=intercept_var,
            x_mean_=np.asarray(x_mean, dtype=np.float64),
            feature_scale_=feature_scale,
            sum_precision_=sum_precision,
            effective_n_=effective_n,
            sigma_floor_fraction_=floor_fraction,
        )

    def _common_valid_rows(
        self,
        X: np.ndarray,
        means: np.ndarray,
        sigmas: np.ndarray,
    ) -> np.ndarray:
        valid = np.all(np.isfinite(X), axis=1)
        valid &= np.all(np.isfinite(means), axis=0)
        valid &= np.all(np.isfinite(sigmas) & (sigmas >= 0.0), axis=0)
        return valid

    def _select_snps(
        self,
        X: np.ndarray,
        means: np.ndarray,
        sigmas: np.ndarray,
    ) -> np.ndarray:
        variable = np.var(X, axis=0) > MIN_FEATURE_SCALE
        keep = np.flatnonzero(variable)
        if keep.size == 0:
            raise ValueError("No variable SNPs in the cis-window for this gene.")
        X_variable = X[:, keep]

        if self.screen is None or self.screen <= 0 or keep.size <= self.screen:
            return keep

        scores = np.zeros(keep.size, dtype=np.float64)
        for mean, sigma in zip(means, sigmas):
            sigma = np.maximum(sigma, self.sigma_floor)
            member_scores = marginal_abs_corr(
                X_variable,
                mean,
                weights=sigma**-2,
            )
            scores = np.maximum(scores, member_scores)
        top = np.argpartition(-scores, self.screen - 1)[: self.screen]
        return keep[np.sort(top)]

    def _fit_members(
        self,
        X: np.ndarray,
        means: np.ndarray,
        sigmas: np.ndarray,
        member_ids: Sequence[str],
    ) -> tuple[tuple[MemberPosterior, ...], np.ndarray]:
        selected = self._select_snps(X, means, sigmas)
        X_selected = X[:, selected]
        members = tuple(
            self._fit_member(X_selected, mean, sigma, member_id)
            for member_id, mean, sigma in zip(member_ids, means, sigmas)
        )
        return members, selected

    @staticmethod
    def _member_predictions(
        members: Sequence[MemberPosterior],
        X: np.ndarray,
    ) -> np.ndarray:
        return np.stack(
            [member.intercept_ + X @ member.coef_ for member in members]
        )

    def _metrics(
        self,
        members: Sequence[MemberPosterior],
        X: np.ndarray,
        means: np.ndarray,
        sigmas: np.ndarray,
    ) -> dict:
        if X.shape[0] == 0:
            return {
                "r2": float("nan"),
                "pearson": float("nan"),
                "spearman": float("nan"),
                "whitened_mse": float("nan"),
                "gaussian_nll": float("nan"),
                "member_r2": tuple(float("nan") for _ in members),
            }
        predictions = self._member_predictions(members, X)
        member_r2 = tuple(
            _safe_r2(target, prediction)
            for target, prediction in zip(means, predictions)
        )
        target_mean = means.mean(axis=0)
        prediction_mean = predictions.mean(axis=0)
        sigma_safe = np.maximum(sigmas, self.sigma_floor)
        standardized_residual = (means - predictions) / sigma_safe
        gaussian_nll = 0.5 * (
            standardized_residual**2
            + 2.0 * np.log(sigma_safe)
            + np.log(2.0 * np.pi)
        )
        return {
            "r2": _safe_r2(target_mean, prediction_mean),
            "pearson": safe_pearson(target_mean, prediction_mean),
            "spearman": safe_spearman(target_mean, prediction_mean),
            "whitened_mse": float(np.mean(standardized_residual**2)),
            "gaussian_nll": float(np.mean(gaussian_nll)),
            "member_r2": member_r2,
        }

    def fit_gene_matrix(
        self,
        gene: str,
        X: np.ndarray,
        member_means: np.ndarray,
        member_sigmas: np.ndarray,
        snp_ids: np.ndarray,
        chr: str | int,
        *,
        member_ids: Optional[Sequence[str | int]] = None,
        X_test: Optional[np.ndarray] = None,
        member_means_test: Optional[np.ndarray] = None,
        member_sigmas_test: Optional[np.ndarray] = None,
    ) -> EnsembleLRStruct:
        X = np.asarray(X, dtype=np.float64)
        means = np.asarray(member_means, dtype=np.float64)
        sigmas = np.asarray(member_sigmas, dtype=np.float64)
        snp_ids = np.asarray(snp_ids)
        if X.ndim != 2:
            raise ValueError("X must have shape (n_individuals, n_snps).")
        if means.ndim != 2 or sigmas.shape != means.shape:
            raise ValueError(
                "member_means/member_sigmas must share shape "
                "(n_members, n_individuals)."
            )
        if means.shape[1] != X.shape[0]:
            raise ValueError("Member targets are not row-aligned with X.")
        if snp_ids.shape != (X.shape[1],):
            raise ValueError("snp_ids must contain one ID per X column.")
        if means.shape[0] == 0:
            raise ValueError("At least one ensemble member is required.")
        if member_ids is None:
            member_ids = [str(i) for i in range(means.shape[0])]
        if len(member_ids) != means.shape[0]:
            raise ValueError("member_ids must contain one ID per member.")
        member_ids = tuple(str(member_id) for member_id in member_ids)
        if len(set(member_ids)) != len(member_ids):
            raise ValueError("member_ids must be unique.")

        valid = self._common_valid_rows(X, means, sigmas)
        X, means, sigmas = X[valid], means[:, valid], sigmas[:, valid]
        if X.shape[0] == 0:
            raise ValueError("No rows are finite across every ensemble member.")

        external_test_inputs = (
            X_test,
            member_means_test,
            member_sigmas_test,
        )
        if any(value is not None for value in external_test_inputs) and not all(
            value is not None for value in external_test_inputs
        ):
            raise ValueError(
                "X_test, member_means_test, and member_sigmas_test must be "
                "provided together."
            )
        has_external_test = all(value is not None for value in external_test_inputs)
        if has_external_test:
            X_test = np.asarray(X_test, dtype=np.float64)
            means_test = np.asarray(member_means_test, dtype=np.float64)
            sigmas_test = np.asarray(member_sigmas_test, dtype=np.float64)
            if (
                X_test.ndim != 2
                or X_test.shape[1] != X.shape[1]
                or means_test.shape != sigmas_test.shape
                or means_test.shape != (means.shape[0], X_test.shape[0])
            ):
                raise ValueError("Held-out ensemble targets/design are misaligned.")
            valid_test = self._common_valid_rows(X_test, means_test, sigmas_test)
            X_test = X_test[valid_test]
            means_test = means_test[:, valid_test]
            sigmas_test = sigmas_test[:, valid_test]

            final_members, selected = self._fit_members(
                X, means, sigmas, member_ids
            )
            train_metrics = self._metrics(
                final_members, X[:, selected], means, sigmas
            )
            test_metrics = self._metrics(
                final_members, X_test[:, selected], means_test, sigmas_test
            )
            n_train, n_test = X.shape[0], X_test.shape[0]
        else:
            train_idx, test_idx = train_test_indices(
                X.shape[0], seed=self.seed, key=gene
            )
            if test_idx is not None:
                heldout_members, heldout_selected = self._fit_members(
                    X[train_idx], means[:, train_idx], sigmas[:, train_idx], member_ids
                )
                train_metrics = self._metrics(
                    heldout_members,
                    X[train_idx][:, heldout_selected],
                    means[:, train_idx],
                    sigmas[:, train_idx],
                )
                test_metrics = self._metrics(
                    heldout_members,
                    X[test_idx][:, heldout_selected],
                    means[:, test_idx],
                    sigmas[:, test_idx],
                )
                n_train, n_test = train_idx.size, test_idx.size
            else:
                n_train, n_test = X.shape[0], 0

            final_members, selected = self._fit_members(
                X, means, sigmas, member_ids
            )
            if test_idx is None:
                train_metrics = self._metrics(
                    final_members,
                    X[:, selected],
                    means,
                    sigmas,
                )
                test_metrics = self._metrics(
                    final_members,
                    np.empty((0, selected.size), dtype=np.float64),
                    means[:, :0],
                    sigmas[:, :0],
                )

        snp_ids = snp_ids[selected]
        member_coef = np.stack([member.coef_ for member in final_members])
        member_var = np.stack([member.coef_var_ for member in final_members])
        coef, coef_var, within_var, between_var = combine_member_moments(
            member_coef, member_var
        )
        member_intercepts = np.asarray(
            [member.intercept_ for member in final_members]
        )
        member_intercept_vars = np.asarray(
            [member.intercept_var_ for member in final_members]
        )
        intercept, intercept_var, _, _ = combine_member_moments(
            member_intercepts[:, None], member_intercept_vars[:, None]
        )
        inclusion_probability = np.mean(
            np.stack(
                [member.inclusion_probability_ for member in final_members]
            ),
            axis=0,
        )

        model = EnsembleLRStruct(
            gene=gene,
            chr=chr,
            snp_ids=snp_ids,
            members_=final_members,
            coef_=coef,
            coef_var_=coef_var,
            coef_within_var_=within_var,
            coef_between_var_=between_var,
            inclusion_probability_=inclusion_probability,
            intercept_=float(intercept[0]),
            intercept_var_=float(intercept_var[0]),
            heldout_r2_=test_metrics["r2"],
            insample_r2_=train_metrics["r2"],
            heldout_pearson_r_=test_metrics["pearson"],
            insample_pearson_r_=train_metrics["pearson"],
            heldout_spearman_r_=test_metrics["spearman"],
            insample_spearman_r_=train_metrics["spearman"],
            heldout_whitened_mse_=test_metrics["whitened_mse"],
            insample_whitened_mse_=train_metrics["whitened_mse"],
            heldout_gaussian_nll_=test_metrics["gaussian_nll"],
            insample_gaussian_nll_=train_metrics["gaussian_nll"],
            member_heldout_r2_=test_metrics["member_r2"],
            member_insample_r2_=train_metrics["member_r2"],
            n_train_=int(n_train),
            n_test_=int(n_test),
        )
        self.models_[gene] = model
        return model

    @staticmethod
    def _empty_test_targets(
        n_members: int,
        n_snps: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.empty((0, n_snps), dtype=np.float64),
            np.empty((n_members, 0), dtype=np.float64),
            np.empty((n_members, 0), dtype=np.float64),
        )

    def fit_gene_from_dataset(
        self,
        dataset: EnsembleGenotypeDataset,
        gene: str,
        test_dataset: Optional[EnsembleGenotypeDataset] = None,
    ) -> EnsembleLRStruct:
        X, means, sigmas, snp_ids, chrom = dataset.get_gene_matrix(gene)
        X_test = means_test = sigmas_test = None
        if test_dataset is not None:
            try:
                X_test, means_test, sigmas_test, test_snps, _ = (
                    test_dataset.get_gene_matrix(gene)
                )
            except ValueError:
                X_test, means_test, sigmas_test = self._empty_test_targets(
                    len(dataset.members), X.shape[1]
                )
            else:
                if not np.array_equal(snp_ids, test_snps):
                    raise ValueError("Training and held-out SNP IDs are not aligned.")
        return self.fit_gene_matrix(
            gene,
            X,
            means,
            sigmas,
            snp_ids,
            chrom,
            member_ids=dataset.member_ids,
            X_test=X_test,
            member_means_test=means_test,
            member_sigmas_test=sigmas_test,
        )

    def fit_gene_from_design(
        self,
        dataset: EnsembleGenotypeDataset,
        gene: str,
        design: tuple,
        test_dataset: Optional[EnsembleGenotypeDataset] = None,
    ) -> EnsembleLRStruct:
        X, snp_ids, chrom, individuals = design
        means, sigmas = dataset.member_targets(gene, individuals)
        X_test = means_test = sigmas_test = None
        if test_dataset is not None:
            try:
                X_test, means_test, sigmas_test, test_snps, _ = (
                    test_dataset.get_gene_matrix(gene)
                )
            except ValueError:
                X_test, means_test, sigmas_test = self._empty_test_targets(
                    len(dataset.members), X.shape[1]
                )
            else:
                if not np.array_equal(snp_ids, test_snps):
                    raise ValueError("Training and held-out SNP IDs are not aligned.")
        return self.fit_gene_matrix(
            gene,
            X,
            means,
            sigmas,
            snp_ids,
            chrom,
            member_ids=dataset.member_ids,
            X_test=X_test,
            member_means_test=means_test,
            member_sigmas_test=sigmas_test,
        )

    def _fit_one(
        self,
        dataset: EnsembleGenotypeDataset,
        gene: str,
        i: int,
        n: int,
        verbose: bool,
        test_dataset: Optional[EnsembleGenotypeDataset],
    ) -> Optional[EnsembleLRStruct]:
        try:
            model = self.fit_gene_from_dataset(
                dataset, gene, test_dataset=test_dataset
            )
            if verbose:
                selected = np.any(
                    np.stack(
                        [
                            member.inclusion_probability_ >= self.pip_threshold
                            for member in model.members_
                        ]
                    ),
                    axis=0,
                )
                print(
                    f"[{i}/{n}] fit {gene}: selected={int(selected.sum())}, "
                    f"heldout_r2={model.heldout_r2_:.4f}, "
                    f"heldout_pearson_r={model.heldout_pearson_r_:.4f}, "
                    f"whitened_mse={model.heldout_whitened_mse_:.4f}, "
                    f"n_test={model.n_test_}"
                )
            return model
        except Exception as exc:
            if verbose:
                print(f"[{i}/{n}] skip {gene}: {exc}")
            return None

    def fit_dataset(
        self,
        dataset: EnsembleGenotypeDataset,
        verbose: bool = True,
        test_dataset: Optional[EnsembleGenotypeDataset] = None,
    ) -> Dict[str, EnsembleLRStruct]:
        if not isinstance(dataset, EnsembleGenotypeDataset):
            raise TypeError("EnsembleLR requires an EnsembleGenotypeDataset.")
        if test_dataset is not None and dataset.member_ids != test_dataset.member_ids:
            raise ValueError("Training and held-out ensemble member IDs differ.")
        genes = list(dataset.genes)
        n = len(genes)

        if self.n_jobs == 1:
            for i, gene in enumerate(genes, start=1):
                self._fit_one(dataset, gene, i, n, verbose, test_dataset)
            return self.models_

        with threadpool_limits(limits=1):
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [
                    executor.submit(
                        self._fit_one,
                        dataset,
                        gene,
                        i,
                        n,
                        verbose,
                        test_dataset,
                    )
                    for i, gene in enumerate(genes, start=1)
                ]
                iterator = as_completed(futures)
                if not verbose:
                    iterator = tqdm(
                        iterator,
                        total=len(futures),
                        desc="Fitting genes",
                        leave=False,
                    )
                for future in iterator:
                    future.result()
        return self.models_

    def predict_gene_matrix(
        self,
        gene: str,
        X: np.ndarray,
        *,
        return_members: bool = False,
    ) -> np.ndarray:
        model = self.models_[gene]
        X = np.asarray(X, dtype=np.float64)
        predictions = self._member_predictions(model.members_, X)
        return predictions if return_members else predictions.mean(axis=0)

    def predict_gene_posterior(
        self,
        gene: str,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return mixture mean/variance of the latent genetic prediction."""
        model = self.models_[gene]
        X = np.asarray(X, dtype=np.float64)
        means = self._member_predictions(model.members_, X)
        variances = np.stack(
            [
                1.0 / member.sum_precision_
                + (X - member.x_mean_) ** 2 @ member.coef_var_
                for member in model.members_
            ]
        )
        mean, variance, _, _ = combine_member_moments(means, variances)
        return mean, variance

    def summarize_models(self) -> pd.DataFrame:
        rows = []
        for gene, model in self.models_.items():
            member_pips = np.stack(
                [member.inclusion_probability_ for member in model.members_]
            )
            selected = np.any(member_pips >= self.pip_threshold, axis=0)
            rows.append(
                {
                    "gene": gene,
                    "r2": model.heldout_r2_,
                    "insample_r2": model.insample_r2_,
                    "n_train": model.n_train_,
                    "n_test": model.n_test_,
                    "pearson_r": model.heldout_pearson_r_,
                    "insample_pearson_r": model.insample_pearson_r_,
                    "spearman_r": model.heldout_spearman_r_,
                    "insample_spearman_r": model.insample_spearman_r_,
                    "whitened_mse": model.heldout_whitened_mse_,
                    "gaussian_nll": model.heldout_gaussian_nll_,
                    "nonzero_weights": int(selected.sum()),
                    "n_members": len(model.members_),
                    "mean_weight_sd": float(np.mean(model.coef_sd_)),
                    "mean_within_weight_var": float(
                        np.mean(model.coef_within_var_)
                    ),
                    "mean_between_weight_var": float(
                        np.mean(model.coef_between_var_)
                    ),
                    "mean_effective_n": _nanmean(
                        [member.effective_n_ for member in model.members_]
                    ),
                    "mean_sigma_floor_fraction": _nanmean(
                        [
                            member.sigma_floor_fraction_
                            for member in model.members_
                        ]
                    ),
                }
            )
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        frame = frame.sort_values("pearson_r", ascending=True).reset_index(drop=True)
        frame["rank"] = np.arange(1, len(frame) + 1)
        return frame

    @staticmethod
    def _json_chromosome(chromosome: str | int):
        text = str(chromosome)
        return int(text) if text.isdigit() else text

    def save_coefficients(self, output_path: str | Path) -> None:
        """Save raw-dosage posterior moments and member factors to JSON."""
        output = {}
        for gene, model in self.models_.items():
            member_pips = np.stack(
                [member.inclusion_probability_ for member in model.members_]
            )
            selected = np.any(member_pips >= self.pip_threshold, axis=0)
            output[gene] = {
                # Backward-compatible point-prediction fields.
                "snp_ids": [str(snp) for snp in model.snp_ids[selected]],
                "coefs": [float(value) for value in model.coef_[selected]],
                "chr": self._json_chromosome(model.chr),
                "intercept": float(model.intercept_),
                # Aggregate equal-mixture posterior moments.
                "coef_sds": [float(value) for value in model.coef_sd_[selected]],
                "coef_variances": [
                    float(value) for value in model.coef_var_[selected]
                ],
                "coef_within_variances": [
                    float(value) for value in model.coef_within_var_[selected]
                ],
                "coef_between_variances": [
                    float(value) for value in model.coef_between_var_[selected]
                ],
                "inclusion_probabilities": [
                    float(value)
                    for value in model.inclusion_probability_[selected]
                ],
                "intercept_sd": float(model.intercept_sd_),
                # Keeping member vectors preserves the low-rank between-member
                # covariance needed by uncertainty-aware summary TWAS.
                "members": [
                    {
                        "member_id": member.member_id,
                        "coefs": [float(value) for value in member.coef_[selected]],
                        "coef_sds": [
                            float(value)
                            for value in np.sqrt(member.coef_var_[selected])
                        ],
                        "coef_variances": [
                            float(value) for value in member.coef_var_[selected]
                        ],
                        "slab_means": [
                            float(value) for value in member.slab_mean_[selected]
                        ],
                        "slab_sds": [
                            float(value) for value in member.slab_sd_[selected]
                        ],
                        "inclusion_probabilities": [
                            float(value)
                            for value in member.inclusion_probability_[selected]
                        ],
                        "intercept": float(member.intercept_),
                        "intercept_sd": float(
                            np.sqrt(max(member.intercept_var_, 0.0))
                        ),
                    }
                    for member in model.members_
                ],
                "metadata": {
                    "model": "sparsevb_spike_and_slab",
                    "sparsevb_version": package_version("sparsevb"),
                    "slab": self.slab,
                    "prior_scale": self.prior_scale,
                    "sigma_floor": self.sigma_floor,
                    "pip_threshold": self.pip_threshold,
                    "coefficient_scale": "raw_dosage",
                },
            }

        with Path(output_path).open("w") as handle:
            json.dump(output, handle, indent=4)
