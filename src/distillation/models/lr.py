import json
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from src.distillation.dataset import GenotypeDataset
from src.distillation.utils import ld_prune, safe_pearson, safe_spearman, train_test_indices


def build_linear_model(
    model_name: str,
    l1_ratio: float,
    cv: int,
    alphas: int,
    max_iter: int,
    seed: int,
):
    """
    Construct a (CV-tuned) linear model, shared by `LR` and `ProbabilisticLR` so both
    point-estimate and probabilistic (mean/aleatoric/epistemic) fits use the exact same
    ElasticNet/Ridge construction (alpha path, solver settings, etc.) rather than each
    approximating it independently.

    Inner CV is small (e.g. cv=3, alphas=5), so spawning a joblib pool per ENCV.fit
    usually costs more than it saves. Outer parallelism over genes (see `fit_dataset`)
    does the heavy lifting instead.
    """
    if model_name == "ridge":
        return RidgeCV(
            cv=cv,
            alphas=np.logspace(-6, 6, alphas),
            fit_intercept=True,
            scoring="r2",
            gcv_mode="auto",
        )
    elif model_name == "elasticnet":
        # Let ElasticNetCV build the alpha path *from the data*: it computes
        # alpha_max (the smallest penalty that zeros all coefficients) and
        # logspaces down by `eps`. This is far better conditioned than a fixed
        # 1e-6..1e6 grid, whose tiny alphas leave the coordinate-descent solver
        # thrashing against max_iter (slow + ConvergenceWarnings) with almost no
        # regularization. `selection="random"` also speeds up convergence.
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
    else:
        raise ValueError(f"Unknown model name: {model_name}")


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

    # R^2 on held-out individuals (per-gene 20% split); the fair comparison metric.
    heldout_r2_:  Optional[float] = None
    # in-sample R^2 of the held-out model on its own train fold (overfitting gap).
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
        l1_ratio: float = 0.5,  # scPrediXcan has 0.5
        cv: int         = 5,
        alphas: int     = 100,
        max_iter: int   = 10000,
        seed: int       = 42,
        n_jobs: int     = 1,
    ):
        self.l1_ratio   = l1_ratio
        self.cv         = cv
        self.alphas     = alphas
        self.max_iter   = max_iter
        self.seed       = seed
        self.n_jobs     = n_jobs
        self.model_name = model_name
        self.models_: Dict[str, LRStruct] = {}

    def _make_model(self):
        return build_linear_model(
            self.model_name, self.l1_ratio, self.cv, self.alphas, self.max_iter, self.seed
        )

    def _fit_scaled(self, X: np.ndarray, y: np.ndarray):
        """Fit X/y standardizers + the (CV) linear model on the given rows."""
        x_scaler = StandardScaler().fit(X)
        X_scaled = x_scaler.transform(X)
        y_scaler = StandardScaler().fit(y.reshape(-1, 1))
        y_scaled = y_scaler.transform(y.reshape(-1, 1)).reshape(-1)
        enet = self._make_model()
        enet.fit(X_scaled, y_scaled)
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
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
    ) -> LRStruct:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        has_target = ~np.isnan(y)
        if not has_target.all():
            X = X[has_target]
            y = y[has_target]

        if y.size == 0:
            raise ValueError("No non-missing target values available")

        has_external_test = X_test is not None and y_test is not None
        if has_external_test:
            X_test = np.asarray(X_test, dtype=np.float32)
            y_test = np.asarray(y_test, dtype=np.float32)
            has_target_test = ~np.isnan(y_test)
            if not has_target_test.all():
                X_test = X_test[has_target_test]
                y_test = y_test[has_target_test]

        if self.model_name == "ridge":
            # perform LD pruning (dropping the same SNP columns from the external
            # test matrix, if any, so train/test stay column-aligned)
            if has_external_test:
                X, snp_ids, (X_test,) = ld_prune(X, snp_ids, align=[X_test])
            else:
                X, snp_ids = ld_prune(X, snp_ids)

        if has_external_test:
            # Held-out evaluation on a user-supplied, disjoint test set: train on
            # *all* individuals from `X`/`y` (no internal split) and score on `X_test`/`y_test`.
            if y_test.size > 0:
                xs, ys, enet_h = self._fit_scaled(X, y)
                pred_test   = self._predict_scaled(xs, ys, enet_h, X_test)
                pred_train  = self._predict_scaled(xs, ys, enet_h, X)
                heldout_r2  = float(r2_score(y_test, pred_test))
                insample_r2 = float(r2_score(y, pred_train))
                heldout_pearson_r  = safe_pearson(y_test, pred_test)
                insample_pearson_r = safe_pearson(y, pred_train)
                heldout_spearman_r  = safe_spearman(y_test, pred_test)
                insample_spearman_r = safe_spearman(y, pred_train)
                n_train, n_test = int(y.size), int(y_test.size)
            else:
                heldout_r2, insample_r2 = float("nan"), float("nan")
                heldout_pearson_r, insample_pearson_r = float("nan"), float("nan")
                heldout_spearman_r, insample_spearman_r = float("nan"), float("nan")
                n_train, n_test = int(y.size), 0
        else:
            # Held-out evaluation: fit on a per-gene 80% split of the individuals and
            # score R^2 on the remaining 20%, so the number is comparable to any model
            # evaluated out-of-sample (and exposes overfitting for p >> n cis windows).
            # The saved coefficients below are refit on *all* individuals.
            train_idx, test_idx = train_test_indices(y.size, seed=self.seed, key=gene)
            if test_idx is not None:
                xs, ys, enet_h = self._fit_scaled(X[train_idx], y[train_idx])
                pred_test   = self._predict_scaled(xs, ys, enet_h, X[test_idx])
                pred_train  = self._predict_scaled(xs, ys, enet_h, X[train_idx])
                heldout_r2  = float(r2_score(y[test_idx], pred_test))
                insample_r2 = float(r2_score(y[train_idx], pred_train))
                heldout_pearson_r  = safe_pearson(y[test_idx], pred_test)
                insample_pearson_r = safe_pearson(y[train_idx], pred_train)
                heldout_spearman_r  = safe_spearman(y[test_idx], pred_test)
                insample_spearman_r = safe_spearman(y[train_idx], pred_train)
                n_train, n_test = int(train_idx.size), int(test_idx.size)
            else:
                # too few individuals to hold out: no honest generalization estimate
                heldout_r2, insample_r2 = float("nan"), float("nan")
                heldout_pearson_r, insample_pearson_r = float("nan"), float("nan")
                heldout_spearman_r, insample_spearman_r = float("nan"), float("nan")
                n_train, n_test = int(y.size), 0

        # Final model refit on all individuals -> these are the persisted coefficients.
        x_scaler, y_scaler, enet = self._fit_scaled(X, y)

        model = LRStruct(
            model_name=self.model_name,
            gene=gene,
            chr=chr,
            snp_ids=np.asarray(snp_ids),
            coef_=enet.coef_.copy(),
            intercept_=enet.intercept_,
            alpha_=enet.alpha_,
            l1_ratio_=getattr(enet, "l1_ratio_", None),
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
        test_dataset: Optional[GenotypeDataset] = None,
    ) -> LRStruct:
        X, y, snp_ids, chr = dataset.get_gene_matrix(gene)
        X_test, y_test = None, None
        if test_dataset is not None:
            try:
                X_test, y_test, _, _ = test_dataset.get_gene_matrix(gene)
            except ValueError:
                # Gene has no rows in the external test set; train on all of `X`/`y`
                # but report held-out metrics as NaN (see fit_gene_matrix).
                X_test, y_test = np.empty((0, X.shape[1]), dtype=X.dtype), np.empty(0, dtype=y.dtype)
        return self.fit_gene_matrix(gene, X, y, snp_ids, chr, X_test=X_test, y_test=y_test)

    def _fit_one(
        self,
        dataset: GenotypeDataset,
        gene: str,
        i: int,
        n: int,
        verbose: bool,
        test_dataset: Optional[GenotypeDataset] = None,
    ) -> Optional[LRStruct]:
        try:
            model = self.fit_gene_from_dataset(dataset, gene, test_dataset=test_dataset)
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
        test_dataset: Optional[GenotypeDataset] = None,
    ) -> Dict[str, LRStruct]:
        genes = list(dataset.genes)
        n     = len(genes)
        n_jobs = max(1, int(self.n_jobs))

        # A handful of ill-conditioned cis-windows may still hit max_iter even with
        # the data-driven alpha path; silence those so the progress bar stays clean.
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        if n_jobs == 1:
            for i, gene in enumerate(genes, start=1):
                self._fit_one(dataset, gene, i, n, verbose, test_dataset=test_dataset)
            return self.models_

        # Parallel path: thread pool over genes with BLAS pinned to 1 thread per
        # call. sklearn's coordinate descent releases the GIL, so threads scale
        # well, and capping BLAS prevents N x M thread oversubscription.
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

    def predict_gene_matrix(self, gene: str, X: np.ndarray) -> np.ndarray:
        model = self.models_[gene]
        X     = np.asarray(X, dtype=np.float64)

        X_scaled      = (X - model.x_mean_) / model.x_scale_
        y_scaled_pred = model.intercept_ + X_scaled @ model.coef_
        y_pred        = model.y_mean_ + model.y_scale_ * y_scaled_pred
        return y_pred

    def summarize_models(self) -> pd.DataFrame:
        rows = []
        for gene, model in self.models_.items():
            rows.append(
                {
                    "gene": gene,
                    "r2": model.heldout_r2_,          # held-out (per-gene 20%) R^2
                    "insample_r2": model.insample_r2_,
                    "n_train": model.n_train_,
                    "n_test": model.n_test_,
                    "pearson_r": model.heldout_pearson_r_,   # held-out, bounded [-1, 1]
                    "insample_pearson_r": model.insample_pearson_r_,
                    "spearman_r": model.heldout_spearman_r_,   # held-out, rank-based, bounded [-1, 1]
                    "insample_spearman_r": model.insample_spearman_r_,
                    "nonzero_weights": int(np.sum(model.coef_ != 0)),
                    "alpha": model.alpha_,
                    "l1_ratio": model.l1_ratio_,
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