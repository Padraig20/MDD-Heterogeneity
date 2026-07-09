"""
ensemble_lr.py

Probabilistic ("vector-to-distribution") distillation of the MLP deep ensemble
teacher into a *linear* deep ensemble.

Where `lr.py` fits a point-estimate linear model per gene (predicting a scalar
expression), this module fits, per gene, a small ensemble of linear models that
emits a Gaussian predictive distribution N(mu, sigma^2) for every individual:

- each ensemble member has its own linear *mean* head (the probabilistic SNP
  effects -> spread across members gives the epistemic uncertainty of each SNP
  weight) and its own linear *variance* head (aleatoric noise, made positive via
  softplus), exactly mirroring `MLPPredictor`/`MLPEnsemble` in the training code;
- the teacher provides, per (gene, individual), a target mean and a target total
  variance (aleatoric + epistemic) from `get_student_data.py`. We treat these as
  the parameters of a target Gaussian and match them.

Distribution matching uses the closed-form 2-Wasserstein distance between two
1-D Gaussians:

    W2( N(mu_p, s_p^2), N(mu_t, s_t^2) )^2 = (mu_p - mu_t)^2 + (s_p - s_t)^2

which is the optimal-transport cost between the predicted and target Gaussians.
Each ensemble member is trained against the target Gaussian independently (as in
a standard deep ensemble); ensemble diversity from the random initialisation then
yields the epistemic component of the aggregated predictive variance.

Everything is fit in standardised X / standardised y space so the
learned coefficients are directly comparable to the point-estimate models.
"""

from __future__ import annotations

import json
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.distillation.dataset import GenotypeDataset
from src.distillation.utils import safe_pearson, train_test_indices


# --------------------------------------------------------------------------- #
# ----------------------------- Wasserstein loss ---------------------------- #
# --------------------------------------------------------------------------- #

def gaussian_w2_loss(
    mean_pred: torch.Tensor,
    std_pred: torch.Tensor,
    mean_target: torch.Tensor,
    std_target: torch.Tensor,
) -> torch.Tensor:
    """
    Squared 2-Wasserstein distance between predicted and target 1-D Gaussians,
    averaged over all elements:

        W2^2 = (mu_pred - mu_target)^2 + (sigma_pred - sigma_target)^2

    All tensors are broadcast against each other. `std_*` are standard deviations
    (not variances).
    """
    return torch.mean((mean_pred - mean_target) ** 2 + (std_pred - std_target) ** 2)


# --------------------------------------------------------------------------- #
# ------------------------- Linear deep-ensemble model ---------------------- #
# --------------------------------------------------------------------------- #

class LinearEnsemble(nn.Module):
    """
    A deep ensemble of linear models with probabilistic weights.

    For `n_models` members and `input_dim` SNPs, this holds two linear layers that
    each map the design matrix to `n_models` outputs at once (one column per
    member), so a forward pass is a single mat-mul:

    - `mean_head`: per-member mean prediction (mu_m = w_m . x + b_m). The spread of
      `mean_head.weight` across members is the distribution over each SNP effect.
    - `std_head` : per-member aleatoric *standard deviation*, softplus(v_m . x + c_m) > 0.
      We parameterise the std directly (rather than a variance) so the Wasserstein
      loss `(sigma_pred - sigma_target)^2` never needs a `sqrt`, whose gradient
      blows up as the variance approaches zero.
    """

    def __init__(self, n_models: int, input_dim: int):
        super().__init__()
        self.n_models  = n_models
        self.input_dim = input_dim
        self.mean_head = nn.Linear(input_dim, n_models)
        self.std_head  = nn.Linear(input_dim, n_models)
        self.softplus  = nn.Softplus()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (n, input_dim) -> (means, stds), each (n, n_models)."""
        means = self.mean_head(x)
        std   = self.softplus(self.std_head(x))
        return means, std

    @torch.no_grad()
    def predictive(self, x: torch.Tensor, eps: float = 1e-8):
        """
        Aggregate the ensemble into a single Gaussian predictive distribution.

        Returns (mu, total_var, aleatoric, epistemic), each (n,), where
            mu         = mean_m(mu_m)
            aleatoric  = mean_m(sigma_m^2)                   (data noise)
            epistemic  = mean_m(mu_m^2) - mu^2               (disagreement)
            total_var  = aleatoric + epistemic
        """
        means, std = self.forward(x)
        mu         = means.mean(dim=1)
        aleatoric  = (std ** 2).mean(dim=1)
        epistemic  = (means ** 2).mean(dim=1) - mu ** 2
        epistemic  = epistemic.clamp_min(0.0)
        total_var  = (aleatoric + epistemic).clamp_min(eps)
        return mu, total_var, aleatoric, epistemic


@dataclass
class EnsembleLRStruct:
    model_name: str
    gene:       str
    chr:        int
    snp_ids:    np.ndarray

    # distribution over SNP effects (mean head), in standardized X / standardized y space
    # (mean head fits y scaled to unit variance; predictions rescale by y_scale_).
    # coef_mean_[j] / coef_std_[j] summarize the effect of SNP j across ensemble members.
    coef_mean_: np.ndarray
    coef_std_:  np.ndarray
    intercept_mean_: float
    intercept_std_:  float

    # scalers needed for inference + inverse transform. y is only centered (y_scale_ == 1).
    x_mean_:  np.ndarray
    x_scale_: np.ndarray
    y_mean_:  float
    y_scale_: float

    # Diagnostics (native log-expression space). All distribution-matching metrics
    # are evaluated on HELD-OUT individuals (per-gene 20% split); the model that
    # produces them is trained only on the 80% train fold, while the persisted
    # coefficients above are refit on all individuals.
    train_r2_:          Optional[float] = None  # held-out R^2 of the aggregated mean prediction
    insample_r2_:       Optional[float] = None  # in-sample R^2 on the train fold (overfitting gap)
    n_train_:           int = 0
    n_test_:            int = 0
    pearson_r_:         Optional[float] = None  # held-out Pearson r of the means (bounded [-1, 1])
    insample_pearson_r_: Optional[float] = None  # in-sample Pearson r on the train fold
    train_w2_:          Optional[float] = None  # held-out mean 2-Wasserstein distance (mean + std parts)
    mean_w2_:           Optional[float] = None  # mean-matching part of the Wasserstein
    std_w2_:            Optional[float] = None  # std/variance-matching part (variance-fit error, lower is better)
    std_corr_:          Optional[float] = None  # corr(pred std, target std) across individuals (higher is better)
    std_ratio_:         Optional[float] = None  # mean(pred std) / mean(target std) (calibration, ~1 is ideal)
    mean_pred_std_:     Optional[float] = None  # avg predicted total std (std-y space)
    mean_target_std_:   Optional[float] = None  # avg target std (std-y space)
    diverged_:          bool = False            # True if the per-gene fit blew up (metrics are NaN)


class EnsembleLR:
    """
    Fits one `LinearEnsemble` per gene by minimising the Gaussian 2-Wasserstein
    loss against the teacher's (mean, total-variance) targets.
    """

    def __init__(
        self,
        n_models: int      = 5,
        epochs: int        = 300,
        lr: float          = 1e-2,
        weight_decay: float = 1e-4,
        l1: float          = 0.0,
        seed: int          = 42,
        n_jobs: int        = 1,
        model_name: str    = "ensemble-lr",
        device: str | torch.device = "cpu",
        grad_clip: float   = 10.0,
    ):
        self.n_models     = n_models
        self.epochs       = epochs
        self.lr           = lr
        self.weight_decay = weight_decay
        self.l1           = l1  # group-lasso strength on the mean head (0 disables)
        self.seed         = seed
        self.n_jobs       = n_jobs
        self.model_name   = model_name
        self.device       = torch.device(device)
        self.grad_clip    = grad_clip
        self.models_: Dict[str, EnsembleLRStruct] = {}

    @staticmethod
    @torch.no_grad()
    def _group_soft_threshold(weight: torch.Tensor, thresh: float) -> None:
        """
        In-place group-lasso proximal operator on a (n_models, n_snps) weight matrix.

        Each SNP column (its `n_models` per-member weights) is one group. The column is
        rescaled by max(0, 1 - thresh / ||col||_2); columns whose joint L2 norm is
        <= thresh are set exactly to zero, so a SNP is dropped for the whole ensemble
        at once. This is the standard prox of the group-L1 penalty
        `thresh * sum_j ||w_{:,j}||_2`, i.e. a proximal-gradient (ISTA) step layered on
        top of the Adam update.
        """
        col_norm = weight.norm(dim=0, keepdim=True)                 # (1, n_snps)
        scale    = torch.clamp(1.0 - thresh / (col_norm + 1e-12), min=0.0)
        weight.mul_(scale)

    def _fit_ensemble(self, gene: str, X: np.ndarray, y: np.ndarray, y_var: np.ndarray):
        """
        Train one `LinearEnsemble` on the given rows and return
        (model, scalers), where `scalers` carries the per-gene standardization used
        to map raw member outputs back to native (log-expression) space.

        We use 2 scales: one for X and one for the y mean head, and scale the std
        head by its own mean. Near-constant genes may have std(y) ~ 1e-4 while the
        teacher std is ~0.3 (four orders of magnitude apart), so scaling both heads
        keeps Adam well-conditioned.
        """
        device = self.device

        x_scaler = StandardScaler()
        X_scaled = x_scaler.fit_transform(X)

        y_mean   = float(np.mean(y))
        mu_scale = float(np.std(y))
        mu_scale = mu_scale if mu_scale > 1e-8 else 1.0  # constant gene -> no scaling

        std_native = np.sqrt(y_var)                      # native teacher std, ~[0, 0.3]
        std_scale  = float(np.mean(std_native))
        std_scale  = std_scale if std_scale > 1e-8 else 1.0

        y_scaled   = (y - y_mean) / mu_scale             # mean-head target
        std_scaled = std_native / std_scale              # std-head target

        # crc32 gives a process-stable per-gene offset (unlike the randomized built-in hash).
        # The generator stays on CPU (nn.init reads from it) and the initialised model is
        # then moved to the target device.
        gene_offset = zlib.crc32(str(gene).encode("utf-8")) % 100000
        gen = torch.Generator().manual_seed(self.seed + gene_offset)

        Xt    = torch.from_numpy(X_scaled).float().to(device)
        mu_t  = torch.from_numpy(y_scaled).float().unsqueeze(1).to(device)    # (n, 1)
        std_t = torch.from_numpy(std_scaled).float().unsqueeze(1).to(device)  # (n, 1)

        model = LinearEnsemble(self.n_models, X_scaled.shape[1])
        # re-init deterministically per gene for reproducible ensemble diversity
        for layer in (model.mean_head, model.std_head):
            nn.init.normal_(layer.weight, mean=0.0, std=0.25, generator=gen)
            nn.init.zeros_(layer.bias)
        model = model.to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            means, std = model(Xt)
            # each member is matched against the (broadcast) target Gaussian
            loss = gaussian_w2_loss(means, std, mu_t, std_t)
            loss.backward()
            # clip gradients to stop rare exploding steps (ill-conditioned cis-windows
            # with p >> n and standardized rare-SNP columns can otherwise diverge).
            if self.grad_clip is not None and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            optimizer.step()
            # proximal step for L1 (group-lasso) sparsity after L2 (weight decay), applies
            # shrinkage of the mean head weights across ensemble members, grouped by SNP.
            if self.l1 and self.l1 > 0:
                self._group_soft_threshold(model.mean_head.weight, self.lr * self.l1)

        scalers = {
            "x_scaler":  x_scaler,
            "y_mean":    y_mean,
            "mu_scale":  mu_scale,
            "std_scale": std_scale,
        }
        return model, scalers

    def _evaluate(self, model: LinearEnsemble, scalers: dict, X: np.ndarray, y: np.ndarray, y_var: np.ndarray) -> dict:
        """
        Run a trained ensemble on rows (X, y, y_var) and return native-space
        distribution-matching metrics. Used on the held-out fold, so the numbers are
        honest generalization estimates.
        """
        x_scaler   = scalers["x_scaler"]
        y_mean     = scalers["y_mean"]
        mu_scale   = scalers["mu_scale"]
        std_scale  = scalers["std_scale"]

        std_native = np.sqrt(y_var)
        X_scaled   = x_scaler.transform(X)
        Xt         = torch.from_numpy(X_scaled).float().to(self.device)

        model.eval()
        with torch.no_grad():
            means_raw, stds_raw = model(Xt)
        means_raw = means_raw.cpu().numpy()   # (n, M) in standardized-y space
        stds_raw  = stds_raw.cpu().numpy()    # (n, M) in std-scaled space

        mu_members  = y_mean + mu_scale * means_raw       # (n, M) native means
        std_members = std_scale * stds_raw                # (n, M) native stds

        mu_np      = mu_members.mean(axis=1)                                  # (n,)
        aleatoric  = (std_members ** 2).mean(axis=1)                          # (n,)
        epistemic  = np.clip((mu_members ** 2).mean(axis=1) - mu_np ** 2, 0.0, None)
        total_std  = np.sqrt(aleatoric + epistemic)                           # (n,)

        # Detect a diverged / failed fit. Native log-expression means are a few units
        # and target stds are <~0.3, so predictions many orders of magnitude larger mean
        # the per-gene optimisation blew up; such R^2 / W2 are meaningless -> NaN
        DIVERGE_ABS = 1e3
        diverged = (
            not np.all(np.isfinite(mu_np))
            or not np.all(np.isfinite(total_std))
            or np.max(np.abs(mu_np)) > DIVERGE_ABS
            or np.max(total_std) > DIVERGE_ABS
        )

        nan = float("nan")
        metrics = {
            "r2": nan, "pearson_r": nan, "w2": nan, "mean_w2": nan, "std_w2": nan,
            "std_corr": nan, "std_ratio": nan, "pred_std": nan,
            "target_std": float(np.mean(std_native)) if std_native.size else nan,
            "diverged": bool(diverged),
        }
        if diverged or y.size <= 1:
            return metrics

        metrics["r2"]        = float(r2_score(y, mu_np))
        metrics["pearson_r"] = safe_pearson(mu_np, y)
        # Wasserstein split into its mean and std (variance) contributions
        metrics["mean_w2"]   = float(np.mean((mu_np - y) ** 2))
        metrics["std_w2"]    = float(np.mean((total_std - std_native) ** 2))
        metrics["w2"]        = metrics["mean_w2"] + metrics["std_w2"]
        # How well does the predicted per-individual std track the target std?
        metrics["std_corr"]  = safe_pearson(total_std, std_native)
        denom                = float(np.mean(std_native))
        metrics["std_ratio"] = float(np.mean(total_std) / denom) if denom > 1e-12 else nan
        metrics["pred_std"]  = float(np.mean(total_std))
        return metrics

    def fit_gene_matrix(
        self,
        gene: str,
        X: np.ndarray,
        y: np.ndarray,
        y_var: np.ndarray,
        snp_ids: np.ndarray,
        chr: int,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        y_var_test: Optional[np.ndarray] = None,
    ) -> EnsembleLRStruct:
        X     = np.asarray(X, dtype=np.float32)
        y     = np.asarray(y, dtype=np.float32)
        y_var = np.asarray(y_var, dtype=np.float32)

        # drop individuals with a missing mean or variance target
        valid = ~np.isnan(y) & ~np.isnan(y_var)
        if not valid.all():
            X     = X[valid]
            y     = y[valid]
            y_var = y_var[valid]

        if y.size == 0:
            raise ValueError("No non-missing target values available")
        if X.shape[1] == 0:
            raise ValueError("No SNPs in cis-window for this gene")

        # teacher variances must be non-negative; clamp tiny negatives from noise.
        y_var = np.clip(y_var, 0.0, None)

        has_external_test = X_test is not None and y_test is not None and y_var_test is not None
        if has_external_test:
            X_test     = np.asarray(X_test, dtype=np.float32)
            y_test     = np.asarray(y_test, dtype=np.float32)
            y_var_test = np.asarray(y_var_test, dtype=np.float32)
            valid_test = ~np.isnan(y_test) & ~np.isnan(y_var_test)
            if not valid_test.all():
                X_test     = X_test[valid_test]
                y_test     = y_test[valid_test]
                y_var_test = y_var_test[valid_test]
            y_var_test = np.clip(y_var_test, 0.0, None)

        if has_external_test:
            # Held-out evaluation on a user-supplied, disjoint test set: train on
            # *all* individuals from `X`/`y`/`y_var` (no internal split) and evaluate
            # on `X_test`/`y_test`/`y_var_test`.
            if y_test.size > 0:
                model_h, scalers_h = self._fit_ensemble(gene, X, y, y_var)
                metrics     = self._evaluate(model_h, scalers_h, X_test, y_test, y_var_test)
                insample    = self._evaluate(model_h, scalers_h, X, y, y_var)
                insample_r2 = insample["r2"]
                insample_pearson_r = insample["pearson_r"]
                n_train, n_test = int(y.size), int(y_test.size)
            else:
                metrics = {
                    "r2": float("nan"), "pearson_r": float("nan"), "w2": float("nan"),
                    "mean_w2": float("nan"), "std_w2": float("nan"), "std_corr": float("nan"),
                    "std_ratio": float("nan"), "pred_std": float("nan"),
                    "target_std": float(np.mean(np.sqrt(y_var))), "diverged": False,
                }
                insample_r2 = float("nan")
                insample_pearson_r = float("nan")
                n_train, n_test = int(y.size), 0
        else:
            # Held-out evaluation: train on a per-gene 80% split of the individuals and
            # score every distribution-matching metric on the remaining 20%, so the
            # numbers are comparable to the point-estimate LR (also held-out) and expose
            # overfitting for p >> n cis windows. The persisted coefficients are refit on
            # all individuals afterwards.
            train_idx, test_idx = train_test_indices(y.size, seed=self.seed, key=gene)
            if test_idx is not None:
                model_h, scalers_h = self._fit_ensemble(
                    gene, X[train_idx], y[train_idx], y_var[train_idx]
                )
                metrics     = self._evaluate(model_h, scalers_h, X[test_idx], y[test_idx], y_var[test_idx])
                insample    = self._evaluate(model_h, scalers_h, X[train_idx], y[train_idx], y_var[train_idx])
                insample_r2 = insample["r2"]
                insample_pearson_r = insample["pearson_r"]
                n_train, n_test = int(train_idx.size), int(test_idx.size)
            else:
                # too few individuals to hold out: no honest generalization estimate
                metrics = {
                    "r2": float("nan"), "pearson_r": float("nan"), "w2": float("nan"),
                    "mean_w2": float("nan"), "std_w2": float("nan"), "std_corr": float("nan"),
                    "std_ratio": float("nan"), "pred_std": float("nan"),
                    "target_std": float(np.mean(np.sqrt(y_var))), "diverged": False,
                }
                insample_r2 = float("nan")
                insample_pearson_r = float("nan")
                n_train, n_test = int(y.size), 0

        # Final model refit on all individuals -> these are the persisted coefficients.
        model, _ = self._fit_ensemble(gene, X, y, y_var)

        # summarize the learned distribution over SNP effects (mean head) across members
        W_mean = model.mean_head.weight.detach().cpu().numpy()  # (n_models, n_snps)
        b_mean = model.mean_head.bias.detach().cpu().numpy()    # (n_models,)

        # persisted scalers come from the full-data refit (mean head fits standardized y)
        x_scaler_full = StandardScaler().fit(X)
        y_mean_full   = float(np.mean(y))
        mu_scale_full = float(np.std(y))
        mu_scale_full = mu_scale_full if mu_scale_full > 1e-8 else 1.0

        struct = EnsembleLRStruct(
            model_name=self.model_name,
            gene=gene,
            chr=chr,
            snp_ids=np.asarray(snp_ids),
            coef_mean_=W_mean.mean(axis=0),
            coef_std_=W_mean.std(axis=0),
            intercept_mean_=float(b_mean.mean()),
            intercept_std_=float(b_mean.std()),
            x_mean_=x_scaler_full.mean_.copy(),
            x_scale_=x_scaler_full.scale_.copy(),
            y_mean_=y_mean_full,
            y_scale_=mu_scale_full,  # mean head fits standardized y; predictions rescale by this
            train_r2_=metrics["r2"],
            insample_r2_=insample_r2,
            n_train_=n_train,
            n_test_=n_test,
            pearson_r_=metrics["pearson_r"],
            insample_pearson_r_=insample_pearson_r,
            train_w2_=metrics["w2"],
            mean_w2_=metrics["mean_w2"],
            std_w2_=metrics["std_w2"],
            std_corr_=metrics["std_corr"],
            std_ratio_=metrics["std_ratio"],
            mean_pred_std_=metrics["pred_std"],
            mean_target_std_=metrics["target_std"],
            diverged_=bool(metrics["diverged"]),
        )
        self.models_[gene] = struct
        return struct

    def fit_gene_from_dataset(
        self,
        dataset: GenotypeDataset,
        gene: str,
        test_dataset: Optional[GenotypeDataset] = None,
    ) -> EnsembleLRStruct:
        X, y, y_var, snp_ids, chr = dataset.get_gene_matrix(gene, return_variance=True)
        X_test, y_test, y_var_test = None, None, None
        if test_dataset is not None:
            try:
                X_test, y_test, y_var_test, _, _ = test_dataset.get_gene_matrix(gene, return_variance=True)
            except ValueError:
                # Gene has no rows in the external test set; train on all of `X`/`y`/`y_var`
                # but report held-out metrics as NaN (see fit_gene_matrix).
                X_test = np.empty((0, X.shape[1]), dtype=X.dtype)
                y_test = np.empty(0, dtype=y.dtype)
                y_var_test = np.empty(0, dtype=y_var.dtype)
        return self.fit_gene_matrix(
            gene, X, y, y_var, snp_ids, chr,
            X_test=X_test, y_test=y_test, y_var_test=y_var_test,
        )

    def _fit_one(
        self,
        dataset: GenotypeDataset,
        gene: str,
        i: int,
        n: int,
        verbose: bool,
        test_dataset: Optional[GenotypeDataset] = None,
    ) -> Optional[EnsembleLRStruct]:
        try:
            model = self.fit_gene_from_dataset(dataset, gene, test_dataset=test_dataset)
            if verbose:
                print(
                    f"[{i}/{n}] fit {gene}: "
                    f"heldout_r2={model.train_r2_:.4f}, heldout_pearson_r={model.pearson_r_:.4f} "
                    f"(insample_r2={model.insample_r2_:.4f}, "
                    f"insample_pearson_r={model.insample_pearson_r_:.4f}, n_test={model.n_test_}), "
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
    ) -> Dict[str, EnsembleLRStruct]:
        if not getattr(dataset, "has_variance", False):
            raise ValueError(
                "EnsembleLR requires a dataset built with variance targets (`y_var`). "
                "Pass --variance to train.py."
            )
        if test_dataset is not None and not getattr(test_dataset, "has_variance", False):
            raise ValueError(
                "EnsembleLR requires the held-out test dataset to also carry variance "
                "targets (`y_var`); pass --variance to train.py."
            )

        genes  = list(dataset.genes)
        n      = len(genes)
        n_jobs = max(1, int(self.n_jobs))

        if n_jobs == 1:
            for i, gene in enumerate(genes, start=1):
                self._fit_one(dataset, gene, i, n, verbose, test_dataset=test_dataset)
            return self.models_

        # Parallel path: thread pool over genes. torch releases the GIL during the
        # tensor ops, and we pin intra-op threads to 1 so per-gene fits don't
        # oversubscribe cores.
        prev_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
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
        finally:
            torch.set_num_threads(prev_threads)

        return self.models_

    def predict_gene_matrix(self, gene: str, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict a Gaussian (mean, epistemic variance) in original y space from the
        saved distribution over SNP effects.

        The mean uses the ensemble-mean of the SNP weights. The variance is the
        *epistemic* component only, propagated from the per-SNP weight spread under
        an independent-weights approximation:
            Var(mu) = Var(b) + sum_j x_j^2 * Var(w_j).
        The aleatoric head is not persisted in the saved struct, so it is not
        reflected here; refit and use `LinearEnsemble.predictive` if you need the
        full aleatoric + epistemic total variance.
        """
        model = self.models_[gene]
        X     = np.asarray(X, dtype=np.float64)

        X_scaled = (X - model.x_mean_) / model.x_scale_
        mu_scaled = model.intercept_mean_ + X_scaled @ model.coef_mean_
        mu        = model.y_mean_ + model.y_scale_ * mu_scaled
        epistemic_scaled = (
            model.intercept_std_ ** 2 + (X_scaled ** 2) @ (model.coef_std_ ** 2)
        )
        epistemic_var = (model.y_scale_ ** 2) * epistemic_scaled
        return mu, epistemic_var

    def summarize_models(self) -> pd.DataFrame:
        rows = []
        for gene, model in self.models_.items():
            rows.append(
                {
                    "gene": gene,
                    "r2": model.train_r2_,           # held-out (per-gene 20%) R^2
                    "insample_r2": model.insample_r2_,
                    "n_train": model.n_train_,
                    "n_test": model.n_test_,
                    "pearson_r": model.pearson_r_,   # held-out, bounded [-1, 1] mean-fit correlation
                    "insample_pearson_r": model.insample_pearson_r_,
                    "wasserstein": model.train_w2_,
                    "mean_w2": model.mean_w2_,
                    "std_w2": model.std_w2_,          # variance-fit error (lower is better)
                    "std_corr": model.std_corr_,      # rank/tracking of target uncertainty
                    "std_ratio": model.std_ratio_,    # calibration ratio (~1 is ideal)
                    "pred_std": model.mean_pred_std_,
                    "target_std": model.mean_target_std_,
                    "diverged": model.diverged_,
                    "nonzero_weights": int(np.sum(np.abs(model.coef_mean_) > 1e-6)),
                    "mean_coef_std": float(np.mean(model.coef_std_)),
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
        Save the probabilistic SNP effects of each gene's model to a JSON file.

        For every gene we store, per SNP, the ensemble mean and std of the effect
        (i.e. the distribution over the SNP weight), plus the intercept
        distribution. Coefficients are in standardized X / standardized y space,
        matching `lr.LR.save_coefficients`.
        """
        output = {}
        for gene, model in self.models_.items():
            output[gene] = {
                "snp_ids":        [str(snp) for snp in model.snp_ids],
                "coef_mean":      [float(c) for c in model.coef_mean_],
                "coef_std":       [float(c) for c in model.coef_std_],
                "intercept_mean": float(model.intercept_mean_),
                "intercept_std":  float(model.intercept_std_),
                "chr":            int(model.chr),
            }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=4)
