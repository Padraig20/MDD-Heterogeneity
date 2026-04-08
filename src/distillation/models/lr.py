import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

from src.distillation.dataset import GenotypeDataset
from src.distillation.utils import ld_prune


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

    train_r2_: Optional[float] = None


class LR:
    def __init__(
        self,
        model_name: str = "elasticnet",
        l1_ratio: float = 0.5,  # scPrediXcan has 0.5
        cv: int         = 3,
        alphas: int     = 100,
        max_iter: int   = 10000,
        seed: int       = 42,
    ):
        self.l1_ratio   = l1_ratio
        self.cv         = cv
        self.alphas     = alphas
        self.max_iter   = max_iter
        self.seed       = seed
        self.model_name = model_name
        self.models_: Dict[str, LRStruct] = {}

    def _make_model(self):
        if self.model_name == "ridge":
            return RidgeCV(
                cv=self.cv,
                alphas=np.logspace(-6, 6, self.alphas),
                fit_intercept=True,
                scoring="r2",
                gcv_mode="auto",
            )
        elif self.model_name == "elasticnet":
            return ElasticNetCV(
                l1_ratio=self.l1_ratio,
                cv=self.cv,
                alphas=np.logspace(-6, 6, self.alphas),
                max_iter=self.max_iter,
                fit_intercept=True,
                random_state=self.seed,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

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

        if self.model_name == "ridge":
            # perform LD pruning
            X, snp_ids = ld_prune(X, snp_ids)

        x_scaler = StandardScaler()
        X_scaled = x_scaler.fit_transform(X)

        y_scaler = StandardScaler()
        y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)

        enet = self._make_model()
        enet.fit(X_scaled, y_scaled)

        # predict in standardized space, then invert to original y space
        y_hat_scaled = enet.predict(X_scaled)
        y_hat        = y_scaler.inverse_transform(y_hat_scaled.reshape(-1, 1)).reshape(-1)
        train_r2     = r2_score(y, y_hat)

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
            train_r2_=train_r2,
        )
        self.models_[gene] = model
        return model

    def fit_gene_from_dataset(self, dataset: GenotypeDataset, gene: str) -> LRStruct:
        X, y, snp_ids, chr = dataset.get_gene_matrix(gene)
        return self.fit_gene_matrix(gene, X, y, snp_ids, chr)

    def fit_dataset(
        self,
        dataset: GenotypeDataset,
        verbose: bool = True,
    ) -> Dict[str, LRStruct]:
        for i, gene in enumerate(dataset.genes, start=1):
            try:
                model = self.fit_gene_from_dataset(dataset, gene)

                if verbose:
                    nnz = int(np.sum(model.coef_ != 0))
                    print(
                        f"[{i}/{len(dataset.genes)}] fit {gene}: "
                        f"nonzero={nnz}, r2={model.train_r2_:.4f}"
                    )
            except Exception as e:
                if verbose:
                    print(f"[{i}/{len(dataset.genes)}] skip {gene}: {e}")

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
                    "r2": model.train_r2_,
                    "nonzero_weights": int(np.sum(model.coef_ != 0)),
                    "alpha": model.alpha_,
                    "l1_ratio": model.l1_ratio_,
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df = df.sort_values("r2", ascending=True).reset_index(drop=True)
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