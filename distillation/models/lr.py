import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Sequence, Optional
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from distillation.dataset import GenotypeDataset

# We use this as a stuct to store fitted model parameters for each gene,
# and also to store the collection of fitted models for all genes.
@dataclass
class LRStruct:
    gene: str
    snp_ids: np.ndarray
    coef_: np.ndarray
    intercept_: float
    alpha_: float
    l1_ratio_: float
    train_r2_: Optional[float] = None

# This is the actual model class that we will use to fit and predict
# gene expression from genotypes.
class LR:
    def __init__(
        self,
        l1_ratio: float = 0.5, # same as PrediXcan
        cv: int = 10, # standard of PrediXcan
        n_alphas: int = 100,
        max_iter: int = 10000,
        seed: int = 42,
    ):
        self.l1_ratio  = l1_ratio
        self.cv        = cv
        self.n_alphas  = n_alphas
        self.max_iter  = max_iter
        self.seed      = seed
        self.models_: Dict[str, LRStruct] = {}

    def _make_pipeline(self) -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                (
                    "enet",
                    ElasticNetCV(
                        l1_ratio=self.l1_ratio,
                        cv=self.cv,
                        alphas=self.n_alphas,
                        max_iter=self.max_iter,
                        fit_intercept=True,
                        random_state=self.seed,
                    ),
                ),
            ]
        )

    @staticmethod
    def _recover_original_scale_coefficients(pipe: Pipeline):
        scaler = pipe.named_steps["scaler"]
        enet   = pipe.named_steps["enet"]

        coef_std  = enet.coef_.copy()
        mean      = scaler.mean_.copy()
        scale     = scaler.scale_.copy()
        scale     = np.where(scale == 0, 1.0, scale)
        coef_orig = coef_std / scale

        intercept_orig = enet.intercept_ - np.sum(coef_std * mean / scale)
        return coef_orig, float(intercept_orig)

    def fit_gene_matrix(
        self,
        gene: str,
        X: np.ndarray,
        y: np.ndarray,
        snp_ids: np.ndarray,
    ) -> LRStruct:
        if X.ndim != 2:
            raise ValueError(f"{gene}: X must be 2D (num_indiv, num_snps), got {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"{gene}: y must be 1D (num_indiv,), got {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"{gene}: X rows != y length")
        if X.shape[1] != len(snp_ids):
            raise ValueError(f"{gene}: X cols != number of SNP ids")

        pipe = self._make_pipeline()
        pipe.fit(X, y)

        y_hat    = pipe.predict(X)
        train_r2 = r2_score(y, y_hat)

        coef_orig, intercept_orig = self._recover_original_scale_coefficients(pipe)
        enet = pipe.named_steps["enet"]

        model = LRStruct(
            gene=gene,
            snp_ids=np.asarray(snp_ids),
            coef_=coef_orig,
            intercept_=intercept_orig,
            alpha_=float(enet.alpha_),
            l1_ratio_=float(enet.l1_ratio_),
            train_r2_=float(train_r2),
        )
        self.models_[gene] = model
        return model

    def fit_gene_from_dataset(self, dataset: GenotypeDataset, gene: str) -> LRStruct:
        X, y, snp_ids = dataset.get_gene_matrix(gene)
        return self.fit_gene_matrix(gene, X, y, snp_ids)

    def fit_dataset(
        self,
        dataset: GenotypeDataset,
        genes: Optional[Sequence[str]] = None,
        verbose: bool = True,
    ) -> Dict[str, LRStruct]:
        if genes is None:
            genes = dataset.genes

        for i, gene in enumerate(genes, start=1):
            try:
                X, y, snp_ids = dataset.get_gene_matrix(gene)

                self.fit_gene_matrix(gene, X, y, snp_ids)

                if verbose:
                    model = self.models_[gene]
                    nnz = int(np.sum(model.coef_ != 0))
                    print(
                        f"[{i}/{len(genes)}] fit {gene}: "
                        f"samples={X.shape[0]}, snps={X.shape[1]}, "
                        f"nonzero={nnz}, r2={model.train_r2_:.4f}"
                    )

            except Exception as e:
                if verbose:
                    print(f"[{i}/{len(genes)}] skip {gene}: {e}")

        return self.models_

    def predict_gene_matrix(self, gene: str, X: np.ndarray) -> np.ndarray:
        model = self.models_[gene]
        return model.intercept_ + X @ model.coef_
    
    def summarize_models(self) -> pd.DataFrame:
        rows = []
        for gene, model in self.models_.items():
            r2 = model.train_r2_
            nnz = int(np.sum(model.coef_ != 0))
            rows.append({
                "gene": gene,
                "r2": float(r2) if r2 is not None else np.nan,
                "nonzero_weights": nnz,
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("r2", ascending=True).reset_index(drop=True)
        df["rank"] = np.arange(len(df))
        return df
    
if __name__ == "__main__":
    # example usage
    import pandas as pd
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
        names=["family_id", "individual_id"]
    )
    idx2ind_arr = idx2ind_arr["individual_id"].to_numpy()
    idx2ind = {"chr1": idx2ind_arr}
    y_path = Path("student-target/0.csv")

    dataset = GenotypeDataset(bims=bims, idx2ind=idx2ind, y=y_path)
    dataset = dataset.split_by_chromosome(['chr1'])
    print(f"Dataset size: {len(dataset)}")

    model = LR()
    models = model.fit_dataset(dataset)
    print(f"Fitted models for {len(models)} genes.")
