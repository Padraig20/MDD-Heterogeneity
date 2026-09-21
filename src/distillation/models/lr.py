import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, ElasticNetCV, Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from src.distillation.dataset import GenotypeDataset
from src.distillation.utils import (
    configure_convergence_warnings,
    ld_prune,
    safe_pearson,
    safe_spearman,
    screen_snps,
    train_test_indices,
)


def build_linear_model(
    model_name: str,
    l1_ratio: float,
    cv: int,
    alphas: int,
    max_iter: int,
    seed: int,
    alpha: float | None = None,
):
    """
    Construct a linear model shared by `LR` and `ProbabilisticLR`.

    The elastic-net mix is the scPrediXcan value (`l1_ratio=0.5` by default)
    and is never searched. The penalty (sklearn `alpha`, glmnet `lambda`) is
    chosen by inner CV unless an explicit `alpha` is supplied.
    """
    if model_name == "ridge":
        if alpha is not None:
            return Ridge(alpha=alpha, fit_intercept=True)
        return RidgeCV(
            cv=cv,
            alphas=np.logspace(-6, 6, alphas),
            fit_intercept=True,
            scoring="r2",
            gcv_mode="auto",
        )
    if model_name == "elasticnet":
        if alpha is not None:
            return ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                max_iter=max_iter,
                fit_intercept=True,
                random_state=seed,
                selection="random",
            )
        # Data-dependent path: alpha_max down by `eps`, not a fixed 1e-6..1e6
        # grid whose tiny values thrash coordinate descent.
        return ElasticNetCV(
            l1_ratio=l1_ratio,
            cv=cv,
            n_alphas=alphas,
            max_iter=max_iter,
            fit_intercept=True,
            random_state=seed,
            selection="random",
            n_jobs=1,
        )
    raise ValueError(f"Unknown model name: {model_name}")


def fitted_alpha(model) -> float:
    """
    Penalty of a fitted model (`alpha_` or `alpha`).
    """
    alpha = getattr(model, "alpha_", None)
    if alpha is None:
        alpha = getattr(model, "alpha", None)
    return float(alpha) if alpha is not None else float("nan")


def fitted_l1_ratio(model) -> Optional[float]:
    """L1/L2 mixing ratio of a fitted model, or None for the ridge models."""
    l1_ratio = getattr(model, "l1_ratio_", None)
    if l1_ratio is None:
        l1_ratio = getattr(model, "l1_ratio", None)
    return float(l1_ratio) if l1_ratio is not None else None


@dataclass
class LRStruct:
    model_name: str
    gene:       str
    chr:        int
    snp_ids:    np.ndarray

    # model learned in standardized X / standardized y space
    coef_:      np.ndarray
    intercept_: float
    alpha_:     float
    l1_ratio_:  Optional[float]

    # scalers needed for inference + inverse transform
    x_mean_:  np.ndarray
    x_scale_: np.ndarray
    y_mean_:  float
    y_scale_: float

    # R^2 on the per-gene 20% held-out split.
    heldout_r2_:  Optional[float] = None
    # in-sample R^2 of the held-out model on its 80% train fold.
    insample_r2_: Optional[float] = None
    n_train_: int = 0
    n_test_:  int = 0
    # Pearson r counterparts of the above, bounded in [-1, 1] (unlike R^2, which is
    # unbounded below for a badly-fit gene).
    heldout_pearson_r_:  Optional[float] = None
    insample_pearson_r_: Optional[float] = None
    # Spearman rank correlation counterparts: rank-based, so robust to outliers/
    # nonlinear-but-monotonic fits (unlike Pearson r / R^2).
    heldout_spearman_r_:  Optional[float] = None
    insample_spearman_r_: Optional[float] = None


class LR:
    def __init__(
        self,
        model_name: str = "elasticnet",
        l1_ratio: float = 0.5,  # scPrediXcan mix; not cross-validated
        alpha: Optional[float] = None,
        cv: int         = 3,
        alphas: int     = 15,
        max_iter: int   = 10000,
        seed: int       = 42,
        n_jobs: int     = 1,
        screen: Optional[int] = 5000,
    ):
        if alpha is not None and (not np.isfinite(alpha) or alpha <= 0):
            raise ValueError("alpha must be finite and positive when supplied.")
        if cv < 2:
            raise ValueError("cv must be at least 2.")
        if alphas <= 0:
            raise ValueError("alphas must be positive.")
        self.l1_ratio   = l1_ratio
        self.alpha      = None if alpha is None else float(alpha)
        self.cv         = int(cv)
        self.alphas     = int(alphas)
        self.max_iter   = max_iter
        self.seed       = seed
        self.n_jobs     = n_jobs
        self.model_name = model_name
        self.screen     = screen
        self.models_: Dict[str, LRStruct] = {}

    def _make_model(self, alpha: Optional[float] = None):
        chosen = self.alpha if alpha is None else float(alpha)
        return build_linear_model(
            self.model_name,
            self.l1_ratio,
            self.cv,
            self.alphas,
            self.max_iter,
            self.seed,
            alpha=chosen,
        )

    @staticmethod
    def _scale_x(X: np.ndarray):
        """Fit the X standardizer and return it together with the scaled matrix."""
        x_scaler = StandardScaler().fit(X)
        return x_scaler, x_scaler.transform(X)

    def _fit_prescaled(
        self,
        X_scaled: np.ndarray,
        y: np.ndarray,
        alpha: Optional[float] = None,
        sample_weight: Optional[np.ndarray] = None,
    ):
        """
        Fit the y standardizer + the linear model on an already-standardized X, so
        several targets sharing one design matrix (see `ProbabilisticLR`) also share
        its standardization instead of each rebuilding a full copy of it.

        `sample_weight` weights each individual's squared error; the CV
        estimators apply it to their inner CV loss as well. Leave it None
        for the ordinary, unweighted fit.
        """
        y_scaler = StandardScaler().fit(y.reshape(-1, 1))
        y_scaled = y_scaler.transform(y.reshape(-1, 1)).reshape(-1)
        enet = self._make_model(alpha=alpha)
        enet.fit(X_scaled, y_scaled, sample_weight=sample_weight)
        return y_scaler, enet

    def _fit_scaled(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: Optional[float] = None,
        sample_weight: Optional[np.ndarray] = None,
    ):
        """Fit X/y standardizers + the (CV or fixed-penalty) linear model."""
        x_scaler, X_scaled = self._scale_x(X)
        y_scaler, enet     = self._fit_prescaled(
            X_scaled, y, alpha=alpha, sample_weight=sample_weight
        )
        return x_scaler, y_scaler, enet

    @staticmethod
    def _predict_scaled(x_scaler, y_scaler, enet, X: np.ndarray) -> np.ndarray:
        y_hat_scaled = enet.predict(x_scaler.transform(X))
        return y_scaler.inverse_transform(y_hat_scaled.reshape(-1, 1)).reshape(-1)

    def fit_gene_matrix(
        self,
        gene: str,
        X: np.ndarray,
        y: np.ndarray,
        snp_ids: np.ndarray,
        chr: int,
    ) -> LRStruct:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        has_target = ~np.isnan(y)
        if not has_target.all():
            X = X[has_target]
            y = y[has_target]

        if y.size == 0:
            raise ValueError("No non-missing target values available")

        nan = float("nan")
        reuse_alpha = self.alpha
        train_idx, test_idx = train_test_indices(y.size, seed=self.seed, key=gene)
        if test_idx is not None:
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[test_idx], y[test_idx]
            if self.model_name == "ridge":
                X_tr, _, (X_te,) = ld_prune(X_tr, snp_ids, align=[X_te])
            keep = screen_snps(X_tr, y_tr, self.screen)
            if keep is not None:
                X_tr = X_tr[:, keep]
                X_te = X_te[:, keep]
            xs, ys, enet_h = self._fit_scaled(X_tr, y_tr, alpha=reuse_alpha)
            if reuse_alpha is None:
                reuse_alpha = fitted_alpha(enet_h)
            pred_test = self._predict_scaled(xs, ys, enet_h, X_te)
            pred_train = self._predict_scaled(xs, ys, enet_h, X_tr)
            heldout_r2 = float(r2_score(y_te, pred_test))
            insample_r2 = float(r2_score(y_tr, pred_train))
            heldout_pearson_r = safe_pearson(y_te, pred_test)
            insample_pearson_r = safe_pearson(y_tr, pred_train)
            heldout_spearman_r = safe_spearman(y_te, pred_test)
            insample_spearman_r = safe_spearman(y_tr, pred_train)
            n_train, n_test = int(train_idx.size), int(test_idx.size)
        else:
            heldout_r2 = heldout_pearson_r = heldout_spearman_r = nan
            insample_r2 = insample_pearson_r = insample_spearman_r = nan
            n_train, n_test = int(y.size), 0

        if self.model_name == "ridge":
            X, snp_ids = ld_prune(X, snp_ids)
        keep = screen_snps(X, y, self.screen)
        if keep is not None:
            X = X[:, keep]
            snp_ids = np.asarray(snp_ids)[keep]
        x_scaler, y_scaler, enet = self._fit_scaled(X, y, alpha=reuse_alpha)
        if test_idx is None:
            pred_train = self._predict_scaled(x_scaler, y_scaler, enet, X)
            insample_r2 = float(r2_score(y, pred_train))
            insample_pearson_r = safe_pearson(y, pred_train)
            insample_spearman_r = safe_spearman(y, pred_train)

        model = LRStruct(
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
        )
        self.models_[gene] = model
        return model

    def fit_gene_from_dataset(
        self,
        dataset: GenotypeDataset,
        gene: str,
    ) -> LRStruct:
        X, y, snp_ids, chr = dataset.get_gene_matrix(gene)
        return self.fit_gene_matrix(gene, X, y, snp_ids, chr)

    def fit_gene_from_design(
        self,
        dataset: GenotypeDataset,
        gene: str,
        design: tuple,
    ) -> LRStruct:
        """
        Fit one gene against a design matrix read elsewhere (see
        `GenotypeDataset.gene_design`), so cell types sharing a cohort can share one
        genotype read per gene instead of repeating it.
        """
        X, snp_ids, chr, individuals = design
        y, _, _ = dataset.gene_targets(gene, individuals)
        return self.fit_gene_matrix(gene, X, y, snp_ids, chr)

    def _fit_one(
        self,
        dataset: GenotypeDataset,
        gene: str,
        i: int,
        n: int,
        verbose: bool,
    ) -> Optional[LRStruct]:
        try:
            model = self.fit_gene_from_dataset(dataset, gene)
            if verbose:
                nnz = int(np.sum(model.coef_ != 0))
                print(
                    f"[{i}/{n}] fit {gene}: "
                    f"nonzero={nnz}, heldout_r2={model.heldout_r2_:.4f}, "
                    f"heldout_pearson_r={model.heldout_pearson_r_:.4f}, "
                    f"heldout_spearman_r={model.heldout_spearman_r_:.4f} "
                    f"(insample_r2={model.insample_r2_:.4f}, "
                    f"insample_pearson_r={model.insample_pearson_r_:.4f}, "
                    f"insample_spearman_r={model.insample_spearman_r_:.4f}, n_test={model.n_test_})"
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
    ) -> Dict[str, LRStruct]:
        genes = list(dataset.genes)
        n     = len(genes)
        n_jobs = max(1, int(self.n_jobs))

        configure_convergence_warnings(verbose)

        if n_jobs == 1:
            for i, gene in enumerate(genes, start=1):
                self._fit_one(dataset, gene, i, n, verbose)
            return self.models_

        # Parallel path: thread pool over genes with BLAS pinned to 1 thread per
        # call. sklearn's coordinate descent releases the GIL, so threads scale
        # well, and capping BLAS prevents N x M thread oversubscription.
        with threadpool_limits(limits=1):
            with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                futures = {
                    ex.submit(self._fit_one, dataset, gene, i, n, verbose): gene
                    for i, gene in enumerate(genes, start=1)
                }
                iterator = as_completed(futures)
                if not verbose:
                    iterator = tqdm(iterator, total=len(futures), desc="Fitting genes", leave=False)
                for fut in iterator:
                    fut.result()

        return self.models_

    def predict_gene_matrix(self, gene: str, X: np.ndarray) -> np.ndarray:
        model = self.models_[gene]
        X     = np.asarray(X, dtype=np.float64)

        X_scaled      = (X - model.x_mean_) / model.x_scale_
        y_scaled_pred = model.intercept_ + X_scaled @ model.coef_
        y_pred        = model.y_mean_ + model.y_scale_ * y_scaled_pred
        return y_pred

    def _summary_row(self, gene: str, model: LRStruct) -> dict:
        """
        One gene's row of `summarize_models`. Split out so subclasses can add their
        own columns without re-implementing the sorting/ranking below.
        """
        return {
            "gene": gene,
            "r2": model.heldout_r2_,
            "insample_r2": model.insample_r2_,
            "n_train": model.n_train_,
            "n_test": model.n_test_,
            "pearson_r": model.heldout_pearson_r_,
            "insample_pearson_r": model.insample_pearson_r_,
            "spearman_r": model.heldout_spearman_r_,
            "insample_spearman_r": model.insample_spearman_r_,
            "nonzero_weights": int(np.sum(model.coef_ != 0)),
            "alpha": model.alpha_,
            "l1_ratio": model.l1_ratio_,
        }

    def summarize_models(self) -> pd.DataFrame:
        rows = [self._summary_row(gene, model) for gene, model in self.models_.items()]

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
        """Save the non-zero coefficients of each gene's model to a JSON file."""
        output = {}
        for gene, model in self.models_.items():
            snp_ids_nonzero = model.snp_ids[model.coef_ != 0]
            coefs_nonzero   = model.coef_[model.coef_ != 0]

            output[gene] = {}
            output[gene]["snp_ids"]   = [str(snp) for snp in snp_ids_nonzero]
            output[gene]["coefs"]     = [float(c) for c in coefs_nonzero]
            output[gene]["chr"]       = int(model.chr)
            output[gene]["intercept"] = float(model.intercept_)
        
        with open(output_path, "w") as f:
            json.dump(output, f, indent=4)

if __name__ == "__main__":
    # example usage
    from pathlib import Path

    bim = pd.read_csv(
        "chr1.bim",
        sep=r"\s+",
        header=None,
        names=["chrom", "snp", "cm", "bp", "a1", "a2"],
        dtype={"chrom": str, "snp": str, "bp": np.int64},
    )

    bims = {"chr1": bim}

    idx2ind_arr = pd.read_csv(
        "chr1.fam",
        sep=r"\s+",
        header=None,
        usecols=[0, 1],
        names=["family_id", "individual_id"],
    )
    idx2ind_arr = idx2ind_arr["individual_id"].to_numpy()
    idx2ind     = {"chr1": idx2ind_arr}

    y_path = Path("student-target/0.csv")

    dataset = GenotypeDataset(bims=bims, idx2ind=idx2ind, y=y_path)
    dataset = dataset.split_by_chromosome(["chr1"])
    print(f"Dataset size: {len(dataset)}")

    model  = LR()
    models = model.fit_dataset(dataset)
    print(f"Fitted models for {len(models)} genes.")
