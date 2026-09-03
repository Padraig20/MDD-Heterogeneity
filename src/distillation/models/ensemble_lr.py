"""Fast frequentist distillation of deep-ensemble members.

For every gene, one common SNP screen is shared by all teacher members and all
bootstrap replicates.  Each member mean is then distilled with an ordinary
elastic net, optionally using inverse aleatoric variance as its sample weight.
Bootstrap resampling is represented by integer donor multiplicities in
``sample_weight``; the genotype matrix is never physically resampled.

The resulting ``n_members * n_bootstraps`` raw-dosage coefficient vectors form
an empirical SNP-weight distribution.  Its non-zero frequency is the empirical
inclusion probability, while its variance contains both within-member bootstrap
variation (finite-cohort uncertainty) and between-member teacher disagreement.
"""

from __future__ import annotations

import json
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from src.distillation.dataset import GenotypeDataset
from src.distillation.utils import (
    configure_convergence_warnings,
    marginal_abs_corr,
    safe_pearson,
    safe_spearman,
)


MIN_FEATURE_SCALE = 1e-12
DEFAULT_ZERO_TOL = 1e-12


def _nanmean(values: Sequence[float]) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    return float(values[finite].mean()) if finite.any() else float("nan")


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) < 2 or np.var(y_true[mask]) <= MIN_FEATURE_SCALE:
        return float("nan")
    return float(r2_score(y_true[mask], y_pred[mask]))


def combine_member_moments(
    member_means: np.ndarray,
    member_variances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Combine equally weighted member distributions by total variance."""
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


def empirical_weight_moments(
    coefficient_samples: np.ndarray,
    *,
    zero_tol: float = DEFAULT_ZERO_TOL,
) -> dict[str, np.ndarray]:
    """Summarize coefficient samples with shape ``(member, bootstrap, SNP)``."""
    samples = np.asarray(coefficient_samples, dtype=np.float64)
    if samples.ndim != 3 or samples.shape[0] == 0 or samples.shape[1] == 0:
        raise ValueError(
            "coefficient_samples must have shape "
            "(n_members, n_bootstraps, n_snps)."
        )
    if not np.all(np.isfinite(samples)):
        raise ValueError("coefficient_samples must be finite.")

    member_means = samples.mean(axis=1)
    member_variances = np.var(samples, axis=1, ddof=0)
    mean, variance, within, between = combine_member_moments(
        member_means,
        member_variances,
    )
    included = np.abs(samples) > zero_tol
    member_pips = included.mean(axis=1)
    pip = included.mean(axis=(0, 1))

    signs = np.sign(samples) * included
    inclusion_count = included.sum(axis=(0, 1))
    sign_stability = np.divide(
        np.abs(signs.sum(axis=(0, 1))),
        inclusion_count,
        out=np.zeros(samples.shape[2], dtype=np.float64),
        where=inclusion_count > 0,
    )
    return {
        "mean": mean,
        "variance": variance,
        "within_variance": within,
        "between_variance": between,
        "pip": pip,
        "sign_stability": sign_stability,
        "member_means": member_means,
        "member_variances": member_variances,
        "member_pips": member_pips,
    }


@dataclass
class MemberBootstrapDistribution:
    member_id: str
    coef_samples_: np.ndarray
    intercept_samples_: np.ndarray
    coef_: np.ndarray
    coef_var_: np.ndarray
    inclusion_probability_: np.ndarray
    intercept_: float
    intercept_var_: float
    alpha_: float
    effective_n_: float
    sigma_floor_fraction_: float
    mean_n_iter_: float

    @property
    def coef_sd_(self) -> np.ndarray:
        return np.sqrt(self.coef_var_)

    @property
    def intercept_sd_(self) -> float:
        return float(np.sqrt(max(self.intercept_var_, 0.0)))


@dataclass
class EnsembleLRStruct:
    gene: str
    chr: str | int
    snp_ids: np.ndarray
    members_: tuple[MemberBootstrapDistribution, ...]
    coef_: np.ndarray
    coef_var_: np.ndarray
    coef_within_var_: np.ndarray
    coef_between_var_: np.ndarray
    inclusion_probability_: np.ndarray
    sign_stability_: np.ndarray
    intercept_: float
    intercept_var_: float
    n_bootstraps_: int
    n_screened_snps_: int
    alpha_mode_: str
    evaluation_: str
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
        return (
            np.asarray(means, dtype=np.float64),
            np.asarray(sigmas, dtype=np.float64),
        )

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
    """Fit bootstrapped, heteroskedastic elastic nets for every teacher member."""

    def __init__(
        self,
        *,
        l1_ratio: float = 0.5,
        cv: int = 3,
        alphas: int = 15,
        max_iter: int = 2000,
        n_bootstraps: int = 5,
        alpha_mode: str = "shared",
        alpha: Optional[float] = None,
        sigma_floor: float = 1e-4,
        aleatoric_weighting: bool = True,
        pip_threshold: float = 0.5,
        zero_tol: float = DEFAULT_ZERO_TOL,
        seed: int = 42,
        n_jobs: int = 1,
        screen: Optional[int] = 5000,
    ):
        if not 0.0 < l1_ratio <= 1.0:
            raise ValueError("l1_ratio must be in (0, 1] for sparse elastic nets.")
        if cv < 2:
            raise ValueError("cv must be at least 2.")
        if alphas <= 0:
            raise ValueError("alphas must be positive.")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if n_bootstraps <= 0:
            raise ValueError("n_bootstraps must be positive.")
        if alpha_mode not in {"shared", "member"}:
            raise ValueError("alpha_mode must be 'shared' or 'member'.")
        if alpha is not None and (not np.isfinite(alpha) or alpha <= 0):
            raise ValueError("alpha must be finite and positive when supplied.")
        if not np.isfinite(sigma_floor) or sigma_floor <= 0:
            raise ValueError("sigma_floor must be finite and positive.")
        if not 0.0 <= pip_threshold <= 1.0:
            raise ValueError("pip_threshold must be in [0, 1].")
        if not np.isfinite(zero_tol) or zero_tol < 0:
            raise ValueError("zero_tol must be finite and non-negative.")

        self.l1_ratio = float(l1_ratio)
        self.cv = int(cv)
        self.alphas = int(alphas)
        self.max_iter = int(max_iter)
        self.n_bootstraps = int(n_bootstraps)
        self.alpha_mode = alpha_mode
        self.alpha = None if alpha is None else float(alpha)
        self.sigma_floor = float(sigma_floor)
        self.aleatoric_weighting = bool(aleatoric_weighting)
        self.pip_threshold = float(pip_threshold)
        self.zero_tol = float(zero_tol)
        self.seed = int(seed)
        self.n_jobs = max(1, int(n_jobs))
        self.screen = screen
        self.models_: Dict[str, EnsembleLRStruct] = {}

    @staticmethod
    def _common_valid_rows(
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
        if self.screen is None or self.screen <= 0 or keep.size <= self.screen:
            return keep

        X_variable = X[:, keep]
        scores = np.zeros(keep.size, dtype=np.float64)
        for mean, sigma in zip(means, sigmas):
            weights = (
                np.maximum(sigma, self.sigma_floor) ** -2
                if self.aleatoric_weighting
                else None
            )
            scores = np.maximum(
                scores,
                marginal_abs_corr(X_variable, mean, weights=weights),
            )
        top = np.argpartition(-scores, self.screen - 1)[: self.screen]
        return keep[np.sort(top)]

    @staticmethod
    def _scale_design(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_mean = np.mean(X, axis=0, dtype=np.float64)
        x_scale = np.std(X, axis=0, dtype=np.float64)
        if np.any(~np.isfinite(x_scale) | (x_scale <= MIN_FEATURE_SCALE)):
            raise ValueError("Selected design contains an unusable SNP column.")
        # Coordinate descent is fastest on Fortran-contiguous dense matrices.
        X_scaled = np.asfortranarray(
            (X - x_mean) / x_scale,
            dtype=np.float32,
        )
        return X_scaled, x_mean, x_scale

    @staticmethod
    def _normalize_weights(weights: np.ndarray) -> np.ndarray:
        weights = np.asarray(weights, dtype=np.float64)
        if np.any(~np.isfinite(weights) | (weights < 0.0)):
            raise ValueError("Elastic-net sample weights must be finite and non-negative.")
        total = float(weights.sum())
        if total <= 0.0:
            raise ValueError("A bootstrap replicate has zero total sample weight.")
        return weights * (weights.size / total)

    @staticmethod
    def _scale_target(
        y: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        total = float(weights.sum())
        y_mean = float(weights @ y / total)
        variance = float(weights @ (y - y_mean) ** 2 / total)
        y_scale = float(np.sqrt(max(variance, 0.0)))
        if not np.isfinite(y_scale) or y_scale <= MIN_FEATURE_SCALE:
            y_scale = 1.0
        y_scaled = np.asarray((y - y_mean) / y_scale, dtype=np.float32)
        return y_scaled, y_mean, y_scale

    def _make_elastic_net(
        self,
        alpha: float,
        random_state: int,
        *,
        warm_start: bool,
    ) -> ElasticNet:
        return ElasticNet(
            alpha=float(alpha),
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter,
            fit_intercept=True,
            random_state=int(random_state),
            selection="random",
            warm_start=warm_start,
            precompute=False,
            copy_X=True,
        )

    def _fit_elastic_net(
        self,
        estimator: ElasticNet,
        X_scaled: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        x_mean: np.ndarray,
        x_scale: np.ndarray,
        random_state: int,
    ) -> tuple[np.ndarray, float, int]:
        weights = self._normalize_weights(sample_weight)
        y_scaled, y_mean, y_scale = self._scale_target(y, weights)
        estimator.set_params(random_state=int(random_state))
        estimator.fit(X_scaled, y_scaled, sample_weight=weights)

        coef = y_scale * np.asarray(estimator.coef_, dtype=np.float64) / x_scale
        intercept = float(
            y_mean
            + y_scale * float(estimator.intercept_)
            - x_mean @ coef
        )
        n_iter = int(np.max(np.atleast_1d(estimator.n_iter_)))
        return coef, intercept, n_iter

    def _cv_alpha(
        self,
        X_scaled: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        random_state: int,
    ) -> float:
        weights = self._normalize_weights(sample_weight)
        y_scaled, _, _ = self._scale_target(y, weights)
        cv = min(self.cv, X_scaled.shape[0])
        if cv < 2:
            raise ValueError("At least two individuals are required to tune alpha.")
        model = ElasticNetCV(
            l1_ratio=self.l1_ratio,
            cv=cv,
            n_alphas=self.alphas,
            max_iter=self.max_iter,
            fit_intercept=True,
            random_state=int(random_state),
            selection="random",
            n_jobs=1,
            precompute=False,
            copy_X=True,
        )
        model.fit(X_scaled, y_scaled, sample_weight=weights)
        return float(model.alpha_)

    def _select_alphas(
        self,
        gene: str,
        X_scaled: np.ndarray,
        means: np.ndarray,
        sigmas: np.ndarray,
    ) -> np.ndarray:
        n_members = means.shape[0]
        if self.alpha is not None:
            return np.full(n_members, self.alpha, dtype=np.float64)

        base_seed = self._gene_seed(gene)
        sigma_safe = np.maximum(sigmas, self.sigma_floor)
        if self.alpha_mode == "shared":
            reference_y = means.mean(axis=0)
            sample_weight = (
                np.mean(sigma_safe**2, axis=0) ** -1
                if self.aleatoric_weighting
                else np.ones(means.shape[1], dtype=np.float64)
            )
            alpha = self._cv_alpha(
                X_scaled,
                reference_y,
                sample_weight,
                base_seed,
            )
            return np.full(n_members, alpha, dtype=np.float64)

        return np.asarray(
            [
                self._cv_alpha(
                    X_scaled,
                    mean,
                    (
                        sigma**-2
                        if self.aleatoric_weighting
                        else np.ones_like(sigma)
                    ),
                    (base_seed + member_idx + 1) % (2**32),
                )
                for member_idx, (mean, sigma) in enumerate(
                    zip(means, sigma_safe)
                )
            ],
            dtype=np.float64,
        )

    def _gene_seed(self, gene: str) -> int:
        offset = zlib.crc32(str(gene).encode("utf-8"))
        return int((self.seed + offset) % (2**32))

    def _bootstrap_counts(self, gene: str, n: int) -> np.ndarray:
        """Shared donor multiplicities for all members of one gene."""
        rng = np.random.default_rng(self._gene_seed(gene))
        probabilities = np.full(n, 1.0 / n, dtype=np.float64)
        return rng.multinomial(
            n,
            probabilities,
            size=self.n_bootstraps,
        ).astype(np.float64)

    def _fit_bootstraps(
        self,
        gene: str,
        X: np.ndarray,
        means: np.ndarray,
        sigmas: np.ndarray,
        member_ids: Sequence[str],
    ) -> tuple[
        tuple[MemberBootstrapDistribution, ...],
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        X_scaled, x_mean, x_scale = self._scale_design(X)
        alphas = self._select_alphas(gene, X_scaled, means, sigmas)
        counts = self._bootstrap_counts(gene, X.shape[0])
        sigma_safe = np.maximum(sigmas, self.sigma_floor)
        precision = (
            sigma_safe**-2
            if self.aleatoric_weighting
            else np.ones_like(sigma_safe)
        )

        n_members = means.shape[0]
        n_snps = X.shape[1]
        coef_samples = np.empty(
            (n_members, self.n_bootstraps, n_snps),
            dtype=np.float32,
        )
        intercept_samples = np.empty(
            (n_members, self.n_bootstraps),
            dtype=np.float64,
        )
        iterations = np.empty_like(intercept_samples)
        base_seed = self._gene_seed(gene)

        # Member-major ordering permits warm starts between nearby bootstrap
        # objectives. Outer parallelism remains gene-level to avoid oversubscription.
        for member_idx, (mean, member_precision, alpha) in enumerate(
            zip(means, precision, alphas)
        ):
            estimator = self._make_elastic_net(
                alpha,
                (base_seed + member_idx) % (2**32),
                warm_start=True,
            )
            for bootstrap_idx, multiplicity in enumerate(counts):
                fit_seed = (
                    base_seed
                    + member_idx * self.n_bootstraps
                    + bootstrap_idx
                ) % (2**32)
                coef, intercept, n_iter = self._fit_elastic_net(
                    estimator,
                    X_scaled,
                    mean,
                    multiplicity * member_precision,
                    x_mean,
                    x_scale,
                    fit_seed,
                )
                coef_samples[member_idx, bootstrap_idx] = coef
                intercept_samples[member_idx, bootstrap_idx] = intercept
                iterations[member_idx, bootstrap_idx] = n_iter

        moments = empirical_weight_moments(
            coef_samples,
            zero_tol=self.zero_tol,
        )
        members = []
        for member_idx, member_id in enumerate(member_ids):
            member_intercepts = intercept_samples[member_idx]
            member_precision = precision[member_idx]
            members.append(
                MemberBootstrapDistribution(
                    member_id=str(member_id),
                    coef_samples_=coef_samples[member_idx],
                    intercept_samples_=member_intercepts,
                    coef_=moments["member_means"][member_idx],
                    coef_var_=moments["member_variances"][member_idx],
                    inclusion_probability_=moments["member_pips"][member_idx],
                    intercept_=float(member_intercepts.mean()),
                    intercept_var_=float(np.var(member_intercepts, ddof=0)),
                    alpha_=float(alphas[member_idx]),
                    effective_n_=float(
                        member_precision.sum() ** 2
                        / np.sum(member_precision**2)
                    ),
                    sigma_floor_fraction_=float(
                        np.mean(sigmas[member_idx] < self.sigma_floor)
                    ),
                    mean_n_iter_=float(iterations[member_idx].mean()),
                )
            )
        return tuple(members), coef_samples, intercept_samples, counts

    @staticmethod
    def _sample_predictions(
        X: np.ndarray,
        coef_samples: np.ndarray,
        intercept_samples: np.ndarray,
    ) -> np.ndarray:
        return (
            np.einsum("np,mbp->mbn", X, coef_samples, optimize=True)
            + intercept_samples[:, :, None]
        )

    def _metrics_from_predictions(
        self,
        member_predictions: np.ndarray,
        means: np.ndarray,
        sigmas: np.ndarray,
    ) -> dict:
        if member_predictions.shape[1] == 0:
            return {
                "r2": float("nan"),
                "pearson": float("nan"),
                "spearman": float("nan"),
                "whitened_mse": float("nan"),
                "gaussian_nll": float("nan"),
                "member_r2": tuple(float("nan") for _ in member_predictions),
            }
        member_r2 = tuple(
            _safe_r2(target, prediction)
            for target, prediction in zip(means, member_predictions)
        )
        target_mean = means.mean(axis=0)
        prediction_mean = member_predictions.mean(axis=0)
        sigma_safe = np.maximum(sigmas, self.sigma_floor)
        standardized_residual = (means - member_predictions) / sigma_safe
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
        if X.shape[0] < 2:
            raise ValueError("At least two finite individuals are required.")

        external_inputs = (X_test, member_means_test, member_sigmas_test)
        if any(value is not None for value in external_inputs) and not all(
            value is not None for value in external_inputs
        ):
            raise ValueError(
                "X_test, member_means_test, and member_sigmas_test must be "
                "provided together."
            )
        has_external_test = all(value is not None for value in external_inputs)
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

        # This is the only target-dependent screening pass for the gene. Its
        # selected columns are reused by alpha tuning and every member/bootstrap.
        selected = self._select_snps(X, means, sigmas)
        X_selected = X[:, selected]
        snp_ids = snp_ids[selected]
        n_screened_snps = int(selected.size)
        members, coef_samples, intercept_samples, counts = self._fit_bootstraps(
            gene,
            X_selected,
            means,
            sigmas,
            member_ids,
        )

        moments = empirical_weight_moments(
            coef_samples,
            zero_tol=self.zero_tol,
        )

        # Keep the full screened matrix only while fitting. Persisting B*M
        # coefficients for every screened SNP over ~20k genes would dominate
        # memory, so discard below-threshold SNPs immediately after their
        # empirical inclusion frequencies have been computed.
        retained = moments["pip"] >= self.pip_threshold
        X_selected = X_selected[:, retained]
        snp_ids = snp_ids[retained]
        coef_samples = coef_samples[:, :, retained]
        for member in members:
            member.coef_samples_ = member.coef_samples_[:, retained]
            member.coef_ = member.coef_[retained]
            member.coef_var_ = member.coef_var_[retained]
            member.inclusion_probability_ = member.inclusion_probability_[retained]
        moments = empirical_weight_moments(
            coef_samples,
            zero_tol=self.zero_tol,
        )
        intercept_mean, intercept_var, _, _ = combine_member_moments(
            intercept_samples.mean(axis=1)[:, None],
            np.var(intercept_samples, axis=1, ddof=0)[:, None],
        )

        train_sample_predictions = self._sample_predictions(
            X_selected,
            coef_samples,
            intercept_samples,
        )
        train_member_predictions = train_sample_predictions.mean(axis=1)
        train_metrics = self._metrics_from_predictions(
            train_member_predictions,
            means,
            sigmas,
        )

        if has_external_test:
            test_samples = self._sample_predictions(
                X_test[:, selected][:, retained],
                coef_samples,
                intercept_samples,
            )
            test_metrics = self._metrics_from_predictions(
                test_samples.mean(axis=1),
                means_test,
                sigmas_test,
            )
            evaluation = "external"
            n_test = int(X_test.shape[0])
        else:
            oob = counts == 0.0
            oob_count = oob.sum(axis=0)
            has_oob = oob_count > 0
            if has_oob.any():
                oob_sum = np.einsum(
                    "bn,mbn->mn",
                    oob.astype(np.float64),
                    train_sample_predictions,
                    optimize=True,
                )
                oob_member_predictions = (
                    oob_sum[:, has_oob] / oob_count[has_oob]
                )
                test_metrics = self._metrics_from_predictions(
                    oob_member_predictions,
                    means[:, has_oob],
                    sigmas[:, has_oob],
                )
            else:
                test_metrics = self._metrics_from_predictions(
                    np.empty((means.shape[0], 0)),
                    means[:, :0],
                    sigmas[:, :0],
                )
            evaluation = "bootstrap_oob"
            n_test = int(has_oob.sum())

        model = EnsembleLRStruct(
            gene=gene,
            chr=chr,
            snp_ids=snp_ids,
            members_=members,
            coef_=moments["mean"],
            coef_var_=moments["variance"],
            coef_within_var_=moments["within_variance"],
            coef_between_var_=moments["between_variance"],
            inclusion_probability_=moments["pip"],
            sign_stability_=moments["sign_stability"],
            intercept_=float(intercept_mean[0]),
            intercept_var_=float(intercept_var[0]),
            n_bootstraps_=self.n_bootstraps,
            n_screened_snps_=n_screened_snps,
            alpha_mode_="fixed" if self.alpha is not None else self.alpha_mode,
            evaluation_=evaluation,
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
            n_train_=int(X.shape[0]),
            n_test_=n_test,
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
                dataset,
                gene,
                test_dataset=test_dataset,
            )
            if verbose:
                selected = model.inclusion_probability_ >= self.pip_threshold
                print(
                    f"[{i}/{n}] fit {gene}: selected={int(selected.sum())}, "
                    f"fits={len(model.members_) * model.n_bootstraps_}, "
                    f"{model.evaluation_}_r2={model.heldout_r2_:.4f}, "
                    f"pearson_r={model.heldout_pearson_r_:.4f}, "
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
        configure_convergence_warnings(verbose)

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

    @staticmethod
    def _member_mean_predictions(
        members: Sequence[MemberBootstrapDistribution],
        X: np.ndarray,
    ) -> np.ndarray:
        return np.stack(
            [member.intercept_ + X @ member.coef_ for member in members]
        )

    def predict_gene_matrix(
        self,
        gene: str,
        X: np.ndarray,
        *,
        return_members: bool = False,
    ) -> np.ndarray:
        model = self.models_[gene]
        X = np.asarray(X, dtype=np.float64)
        predictions = self._member_mean_predictions(model.members_, X)
        return predictions if return_members else predictions.mean(axis=0)

    def predict_gene_distribution(
        self,
        gene: str,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return empirical mean/variance across all member-bootstrap fits."""
        model = self.models_[gene]
        X = np.asarray(X, dtype=np.float64)
        coef_samples = np.stack(
            [member.coef_samples_ for member in model.members_]
        )
        intercept_samples = np.stack(
            [member.intercept_samples_ for member in model.members_]
        )
        predictions = self._sample_predictions(
            X,
            coef_samples,
            intercept_samples,
        )
        return predictions.mean(axis=(0, 1)), np.var(predictions, axis=(0, 1))

    # Backward-compatible name used by the earlier Bayesian implementation.
    predict_gene_posterior = predict_gene_distribution

    def summarize_models(self) -> pd.DataFrame:
        rows = []
        for gene, model in self.models_.items():
            selected = model.inclusion_probability_ >= self.pip_threshold
            rows.append(
                {
                    "gene": gene,
                    "r2": model.heldout_r2_,
                    "insample_r2": model.insample_r2_,
                    "n_train": model.n_train_,
                    "n_test": model.n_test_,
                    "evaluation": model.evaluation_,
                    "pearson_r": model.heldout_pearson_r_,
                    "insample_pearson_r": model.insample_pearson_r_,
                    "spearman_r": model.heldout_spearman_r_,
                    "insample_spearman_r": model.insample_spearman_r_,
                    "whitened_mse": model.heldout_whitened_mse_,
                    "gaussian_nll": model.heldout_gaussian_nll_,
                    "nonzero_weights": int(selected.sum()),
                    "n_members": len(model.members_),
                    "n_bootstraps": model.n_bootstraps_,
                    "n_fits": len(model.members_) * model.n_bootstraps_,
                    "mean_weight_sd": _nanmean(model.coef_sd_),
                    "mean_bootstrap_weight_var": _nanmean(
                        model.coef_within_var_
                    ),
                    "mean_between_member_weight_var": _nanmean(
                        model.coef_between_var_
                    ),
                    "mean_empirical_pip": _nanmean(
                        model.inclusion_probability_
                    ),
                    "mean_sign_stability": float(
                        np.mean(model.sign_stability_[selected])
                    ) if selected.any() else float("nan"),
                    "mean_alpha": _nanmean(
                        [member.alpha_ for member in model.members_]
                    ),
                    "mean_n_iter": _nanmean(
                        [member.mean_n_iter_ for member in model.members_]
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
        """Save empirical raw-dosage weight distributions to JSON."""
        output = {}
        for gene, model in self.models_.items():
            selected = model.inclusion_probability_ >= self.pip_threshold
            output[gene] = {
                # Backward-compatible point-prediction fields.
                "snp_ids": [str(snp) for snp in model.snp_ids[selected]],
                "coefs": [float(value) for value in model.coef_[selected]],
                "chr": self._json_chromosome(model.chr),
                "intercept": float(model.intercept_),
                # Empirical distribution across all member-bootstrap fits.
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
                "sign_stabilities": [
                    float(value) for value in model.sign_stability_[selected]
                ],
                "intercept_sd": float(model.intercept_sd_),
                # Full bootstrap vectors preserve cross-SNP covariance for TWAS.
                "members": [
                    {
                        "member_id": member.member_id,
                        "coefs": [
                            float(value) for value in member.coef_[selected]
                        ],
                        "coef_sds": [
                            float(value) for value in member.coef_sd_[selected]
                        ],
                        "coef_variances": [
                            float(value) for value in member.coef_var_[selected]
                        ],
                        "inclusion_probabilities": [
                            float(value)
                            for value in member.inclusion_probability_[selected]
                        ],
                        "bootstrap_coefs": [
                            [float(value) for value in replicate[selected]]
                            for replicate in member.coef_samples_
                        ],
                        "intercept": float(member.intercept_),
                        "intercept_sd": float(member.intercept_sd_),
                        "bootstrap_intercepts": [
                            float(value) for value in member.intercept_samples_
                        ],
                        "alpha": float(member.alpha_),
                    }
                    for member in model.members_
                ],
                "metadata": {
                    "model": "bootstrapped_elastic_net_ensemble",
                    "n_members": len(model.members_),
                    "n_bootstraps": model.n_bootstraps_,
                    "n_fits": len(model.members_) * model.n_bootstraps_,
                    "n_screened_snps": model.n_screened_snps_,
                    "n_retained_snps": int(model.snp_ids.size),
                    "l1_ratio": self.l1_ratio,
                    "alpha_mode": model.alpha_mode_,
                    "sigma_floor": self.sigma_floor,
                    "aleatoric_weighting": self.aleatoric_weighting,
                    "pip_threshold": self.pip_threshold,
                    "pip_definition": "fraction_nonzero_across_member_bootstrap_fits",
                    "evaluation": model.evaluation_,
                    "screening": "once_per_gene_reused_for_all_fits",
                    "storage": "below_pip_threshold_discarded_after_fitting",
                    "design_scaling": "once_per_gene_reused_for_all_fits",
                    "coefficient_scale": "raw_dosage",
                },
            }

        with Path(output_path).open("w") as handle:
            json.dump(output, handle, indent=4)
