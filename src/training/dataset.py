import logging

import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
from pathlib import Path

import anndata as ad
import scipy.sparse as sp
from scipy.stats import skew
from scipy.stats import rankdata

class MddDataset(Dataset):
    
    def __init__(self, X_feats: Path|np.ndarray, X_ensids: Path|np.ndarray, X_chroms: Path|np.ndarray, y: Path|pd.DataFrame, normalize: str = "log"):
        """
        Args:
            X_feats (Path|np.ndarray):  Path to input features file (npy) or numpy array:
                                        Contains the pLM/gLM embeddings.
            X_ensids (Path|np.ndarray): Path to input ensids (for features) file (npy) or numpy array:
                                        Contains the ensids corresponding to the rows in X_feats.
            X_chroms (Path|np.ndarray): Path to input chroms (for features) file (npy) or numpy array:
                                        Contains the chromosomes corresponding to the rows in X_feats.
            y (Path|pd.DataFrame):      Path to target labels file (csv) or pandas DataFrame:
                                        2d array; (rows: cell-types, columns: ensid)
            normalize (str):            Whether to transform the target labels to log-scale or percentiles.
        """
        if isinstance(X_feats, Path) and isinstance(X_ensids, Path) and \
           isinstance(X_chroms, Path) and isinstance(y, Path):
            self.X_feats  = np.load(X_feats, mmap_mode='r')
            self.X_ensids = np.load(X_ensids, allow_pickle=True)
            self.X_chroms = np.load(X_chroms, allow_pickle=True).astype(str)
            self.y        = pd.read_csv(y, index_col=0)
            if normalize == "percentiles":
                self.y = self.to_percentiles(self.y)
        else:
            self.X_feats  = X_feats
            self.X_ensids = X_ensids
            self.X_chroms = X_chroms
            self.y        = y

        self.normalize     = normalize
        self.norm_features = None

        # The gene axis is decided by y (its columns), not by X. Restrict and
        # order X to the genes present in y, dropping any X genes that have no
        # target row. Genes in y without features cannot be served and are
        # reported.
        self._align_genes_to_y()

    def _align_genes_to_y(self) -> None:
        """Align the gene axis (X_*) to the genes in y so the gene count is
        decided by y rather than by X."""
        y_genes = self.y.columns.to_numpy().astype(str)
        x_ensids = self.X_ensids.astype(str)

        x_pos = {g: i for i, g in enumerate(x_ensids)}
        order = np.fromiter(
            (x_pos[g] for g in y_genes if g in x_pos),
            dtype=np.int64,
        )

        n_missing = len(y_genes) - len(order)
        if n_missing:
            logging.warning(
                "%d / %d genes in y have no feature row in X and cannot be "
                "served.", n_missing, len(y_genes),
            )

        self.X_feats  = self.X_feats[order]
        self.X_ensids = self.X_ensids[order]
        self.X_chroms = self.X_chroms[order]
    
    def split_by_chromosome(self, chrom: list[str]) -> Dataset:
        """Return a new MddDataset containing only data from the specified chromosomes."""
        mask = np.isin(self.X_chroms, chrom)
        X_feats_chrom  = self.X_feats[mask]
        X_ensids_chrom = self.X_ensids[mask]
        X_chroms_chrom = self.X_chroms[mask]
        return MddDataset(X_feats_chrom, X_ensids_chrom, X_chroms_chrom, self.y, normalize=self.normalize)
    
    def select_genes(self, gene_set: set[str]) -> Dataset:
        """Return a new MddDataset containing only data from the specified set of genes."""
        mask = np.isin(self.X_ensids, list(gene_set))
        X_feats_sel  = self.X_feats[mask]
        X_ensids_sel = self.X_ensids[mask]
        X_chroms_sel = self.X_chroms[mask]
        return MddDataset(X_feats_sel, X_ensids_sel, X_chroms_sel, self.y, normalize=self.normalize)

    def apply_feature_log_transform(self, threshold=1.0) -> None:
        """Apply log-transform to input features."""
        self.norm_features = torch.zeros(self.X_feats.shape[1])
        for i in range(self.X_feats.shape[1]):
            col = self.X_feats[:, i]
            # if all values are non-negative and skewed, apply log-transform
            if np.all(col >= 0) and skew(col) > threshold:
                self.norm_features[i] = 1.0

    @staticmethod
    def to_percentiles(y_df: pd.DataFrame) -> pd.DataFrame:
        """For each cell type (row), rank its genes by expression and divide
        by n_genes to get percentiles in (0, 1] (scPrediXcan style).
        """
        n_genes = y_df.shape[1]
        ranks = rankdata(y_df.values, method="average", axis=1)
        return pd.DataFrame(ranks / n_genes, index=y_df.index, columns=y_df.columns)
    
    def __len__(self) -> int:
        return self.X_ensids.shape[0]
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        ensid = self.X_ensids[idx]
        if ensid not in self.y.columns:
            raise KeyError(f"Ensid {ensid} not found in target labels.")
        x = torch.from_numpy(self.X_feats[idx])
        y = torch.from_numpy(self.y[ensid].values)
        if self.norm_features is not None:
            x = x.clone()
            x[self.norm_features == 1.0] = torch.log(1 + x[self.norm_features == 1.0])
        if self.normalize == "log":
            y = torch.log(1 + y)
        return x, y


class MultiIndividualMddDataset(Dataset):
    """
    Multi-individual variant of MddDataset.

    A "sample" is one (gene, individual) pair. The dataset interface mirrors
    `MddDataset` (same `__getitem__` shapes, same `split_by_chromosome`,
    `select_genes`, `apply_feature_log_transform`, and `norm_features`),
    so it can be substituted for `MddDataset` plug-and-play in training
    code that goes through `get_train_test_dataset` and the standard
    DataLoader pipeline.

    Inputs
    ------
    X_dir : root directory containing one subdirectory per individual.
            Each subdirectory must contain `features.npy`, `ensids.npy`,
            `chroms.npy` (and optionally `tss.npy`, currently unused).
            `ensids.npy` and `chroms.npy` are required to be identical
            across individuals (same gene set, same order).
    y     : path to a pseudo-bulk `.h5ad` produced by `get_ge_per_ind.py`,
            or an already-loaded AnnData. `obs` must contain
            `individual_id` and `cell_type` columns; `var.index` holds
            gene IDs (e.g. ENSEMBL).

    Per-sample shapes
    -----------------
    x : (D_feat,)         features for one (individual, gene)
    y : (n_cell_types,)   pseudo-bulk expression for that gene in that
                          individual, across cell types
    """

    def __init__(
        self,
        X_dir: Path | str | None = None,
        y: Path | str | ad.AnnData | None = None,
        normalize: str = "log",
        cell_type: str | None = None,
        individuals: list[str] | None = None,
        _state: dict | None = None,
    ):
        if _state is not None:
            for k, v in _state.items():
                setattr(self, k, v)
            return

        if X_dir is None or y is None:
            raise ValueError("X_dir and y are required when not constructing from _state.")

        X_dir = Path(X_dir)
        if not X_dir.is_dir():
            raise NotADirectoryError(f"X_dir must be a directory: {X_dir}")

        ind_dirs = self._discover_individual_dirs(X_dir, individuals)
        canonical_ensids, canonical_chroms = self._load_and_validate_gene_axis(ind_dirs)

        self.individuals: list[str] = [d.name for d in ind_dirs]
        self.X_dir: Path = X_dir
        self.X_feats_per_ind: list[np.ndarray] = [
            np.load(d / "features.npy", mmap_mode="r") for d in ind_dirs
        ]

        for d, feats in zip(ind_dirs, self.X_feats_per_ind):
            if feats.shape[0] != len(canonical_ensids):
                raise RuntimeError(
                    f"features.npy for {d.name} has {feats.shape[0]} rows but "
                    f"ensids.npy has {len(canonical_ensids)} entries"
                )

        adata = ad.read_h5ad(y) if not isinstance(y, ad.AnnData) else y

        for required in ("individual_id", "cell_type"):
            if required not in adata.obs.columns:
                raise ValueError(
                    f"Target h5ad obs is missing required column '{required}'. "
                    f"Available: {list(adata.obs.columns)}"
                )

        if cell_type is not None:
            adata = adata[adata.obs["cell_type"].astype(str) == cell_type, :].copy()
            if adata.n_obs == 0:
                raise ValueError(f"No obs rows for cell_type={cell_type!r}")

        ind_in_y = set(adata.obs["individual_id"].astype(str).unique())
        kept_individuals = [ind for ind in self.individuals if ind in ind_in_y]
        dropped_no_y = sorted(set(self.individuals) - set(kept_individuals))
        if dropped_no_y:
            logging.warning(
                "%d individuals have feature directories but no target rows; "
                "dropping them: %s%s",
                len(dropped_no_y),
                dropped_no_y[:5],
                "..." if len(dropped_no_y) > 5 else "",
            )

        self.individuals = kept_individuals
        self.X_feats_per_ind = [
            self.X_feats_per_ind[i]
            for i, ind in enumerate([d.name for d in ind_dirs])
            if ind in kept_individuals
        ]
        if not self.individuals:
            raise RuntimeError("No individuals overlap between X_dir and y.")

        adata = adata[adata.obs["individual_id"].astype(str).isin(self.individuals), :].copy()

        self.cell_types: list[str] = sorted(adata.obs["cell_type"].astype(str).unique().tolist())

        # Build the 3D target tensor (n_ind, n_cell_types, n_y_genes), then drop
        # individuals that don't have rows for all cell types (rare in practice).
        y_tensor_full, present = self._build_y_tensor(adata, self.individuals, self.cell_types)
        complete_mask = present.all(axis=1)
        if not complete_mask.all():
            incomplete = [self.individuals[i] for i, ok in enumerate(complete_mask) if not ok]
            logging.warning(
                "%d individuals are missing at least one cell type and will be dropped: %s%s",
                len(incomplete),
                incomplete[:5],
                "..." if len(incomplete) > 5 else "",
            )
            keep = np.flatnonzero(complete_mask)
            self.individuals = [self.individuals[i] for i in keep]
            self.X_feats_per_ind = [self.X_feats_per_ind[i] for i in keep]
            y_tensor_full = y_tensor_full[keep]

        # Align genes: intersect canonical X ensids with y var, keeping X order.
        y_genes = adata.var.index.astype(str).to_numpy()
        y_gene_to_pos = {g: i for i, g in enumerate(y_genes)}

        in_y = np.fromiter(
            (str(e) in y_gene_to_pos for e in canonical_ensids),
            dtype=bool,
            count=len(canonical_ensids),
        )
        n_dropped_X = int((~in_y).sum())
        if n_dropped_X:
            logging.warning(
                "%d / %d genes from features have no expression row in y; "
                "they will be dropped from the dataset.",
                n_dropped_X,
                len(canonical_ensids),
            )

        gene_indices = np.flatnonzero(in_y)
        kept_ensids = canonical_ensids[gene_indices]
        kept_chroms = canonical_chroms[gene_indices]
        y_idx_for_each_X = np.fromiter(
            (y_gene_to_pos[str(e)] for e in kept_ensids),
            dtype=np.int64,
            count=len(kept_ensids),
        )
        y_tensor = y_tensor_full[:, :, y_idx_for_each_X].astype(np.float32, copy=False)

        if normalize == "percentiles":
            # Within each (individual, cell_type), rank genes by expression and
            # divide by n_genes (scPrediXcan style; same direction as
            # MddDataset.to_percentiles).
            n_genes_y = y_tensor.shape[2]
            ranks = rankdata(y_tensor, method="average", axis=2)
            y_tensor = (ranks / n_genes_y).astype(np.float32, copy=False)

        self.gene_indices: np.ndarray = gene_indices
        self.X_ensids: np.ndarray = kept_ensids
        self.X_chroms: np.ndarray = kept_chroms
        self.y_tensor: np.ndarray = y_tensor
        self.normalize: str = normalize
        self.norm_features: torch.Tensor | None = None

        logging.info(
            "MultiIndividualMddDataset: %d individuals x %d cell types x %d genes "
            "(%d total samples)",
            len(self.individuals),
            len(self.cell_types),
            len(self.X_ensids),
            len(self),
        )

    @staticmethod
    def _discover_individual_dirs(X_dir: Path, individuals: list[str] | None) -> list[Path]:
        all_dirs = sorted(p for p in X_dir.iterdir() if p.is_dir())
        valid = []
        for d in all_dirs:
            if all((d / fname).is_file() for fname in ("features.npy", "ensids.npy", "chroms.npy")):
                valid.append(d)
        if individuals is not None:
            wanted = set(individuals)
            valid = [d for d in valid if d.name in wanted]
            missing = wanted - {d.name for d in valid}
            if missing:
                raise FileNotFoundError(
                    f"Requested individuals are not present (or missing required npy files): {sorted(missing)}"
                )
        if not valid:
            raise FileNotFoundError(
                f"No individual subdirectories with the required npy files were found under {X_dir}"
            )
        return valid

    @staticmethod
    def _load_and_validate_gene_axis(ind_dirs: list[Path]) -> tuple[np.ndarray, np.ndarray]:
        first = ind_dirs[0]
        canonical_ensids = np.load(first / "ensids.npy", allow_pickle=True)
        canonical_chroms = np.load(first / "chroms.npy", allow_pickle=True).astype(str)

        for d in ind_dirs[1:]:
            ensids = np.load(d / "ensids.npy", allow_pickle=True)
            if not np.array_equal(ensids, canonical_ensids):
                raise RuntimeError(
                    f"ensids.npy in {d.name} does not match {first.name}. "
                    "All individual directories must share the same gene set and order."
                )
            chroms = np.load(d / "chroms.npy", allow_pickle=True).astype(str)
            if not np.array_equal(chroms, canonical_chroms):
                raise RuntimeError(
                    f"chroms.npy in {d.name} does not match {first.name}. "
                    "All individual directories must share the same chromosome ordering."
                )
        return canonical_ensids, canonical_chroms

    @staticmethod
    def _build_y_tensor(
        adata: ad.AnnData,
        individuals: list[str],
        cell_types: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        ind_to_pos = {ind: i for i, ind in enumerate(individuals)}
        ct_to_pos = {ct: i for i, ct in enumerate(cell_types)}

        n_ind = len(individuals)
        n_ct = len(cell_types)
        n_var = adata.n_vars

        X_dense = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)

        y_full = np.zeros((n_ind, n_ct, n_var), dtype=np.float32)
        present = np.zeros((n_ind, n_ct), dtype=bool)

        ind_col = adata.obs["individual_id"].astype(str).to_numpy()
        ct_col = adata.obs["cell_type"].astype(str).to_numpy()
        for obs_pos in range(adata.n_obs):
            i = ind_to_pos.get(ind_col[obs_pos])
            j = ct_to_pos.get(ct_col[obs_pos])
            if i is None or j is None:
                continue
            if present[i, j]:
                raise RuntimeError(
                    f"Duplicate (individual_id, cell_type) row in y for "
                    f"({individuals[i]!r}, {cell_types[j]!r})."
                )
            y_full[i, j, :] = X_dense[obs_pos, :]
            present[i, j] = True
        return y_full, present

    def _clone_with_gene_subset(self, mask: np.ndarray) -> "MultiIndividualMddDataset":
        if mask.shape[0] != len(self.X_ensids):
            raise ValueError(
                f"mask length {mask.shape[0]} does not match number of genes {len(self.X_ensids)}"
            )
        state = {
            "individuals": list(self.individuals),
            "X_dir": self.X_dir,
            "X_feats_per_ind": list(self.X_feats_per_ind),
            "cell_types": list(self.cell_types),
            "gene_indices": self.gene_indices[mask],
            "X_ensids": self.X_ensids[mask],
            "X_chroms": self.X_chroms[mask],
            "y_tensor": self.y_tensor[:, :, mask],
            "normalize": self.normalize,
            "norm_features": self.norm_features,
        }
        return MultiIndividualMddDataset(_state=state)

    def split_by_chromosome(self, chrom: list[str]) -> "MultiIndividualMddDataset":
        """Return a new dataset containing only data from the specified chromosomes."""
        mask = np.isin(self.X_chroms, chrom)
        return self._clone_with_gene_subset(mask)

    def select_genes(self, gene_set: set[str]) -> "MultiIndividualMddDataset":
        """Return a new dataset containing only data from the specified set of genes."""
        mask = np.isin(self.X_ensids, list(gene_set))
        return self._clone_with_gene_subset(mask)

    def apply_feature_log_transform(self, threshold: float = 1.0, max_rows: int = 10000) -> None:
        """Decide which feature columns to log-transform based on per-column skew.

        Skewness is estimated on a bounded random sample drawn from each
        individual's in-scope gene rows, capped at `max_rows` rows total to
        keep the call cheap with many individuals.
        """
        if not self.X_feats_per_ind:
            raise RuntimeError("No individuals available to estimate feature skew.")

        rng = np.random.default_rng(42)
        n_ind = len(self.X_feats_per_ind)
        n_per_ind = max(1, max_rows // n_ind)

        gene_indices = self.gene_indices
        if gene_indices.size == 0:
            raise RuntimeError("No genes in scope to estimate feature skew.")

        samples = []
        for feats in self.X_feats_per_ind:
            n_avail = gene_indices.size
            take = min(n_per_ind, n_avail)
            sel = rng.choice(gene_indices, size=take, replace=False)
            sel.sort()
            samples.append(np.asarray(feats[sel]))
        sample = np.concatenate(samples, axis=0)

        n_features = sample.shape[1]
        norm_features = torch.zeros(n_features)
        for i in range(n_features):
            col = sample[:, i]
            if np.all(col >= 0) and skew(col) > threshold:
                norm_features[i] = 1.0
        self.norm_features = norm_features

    def __len__(self) -> int:
        return len(self.individuals) * len(self.X_ensids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        n_genes = len(self.X_ensids)
        ind_idx, gene_idx = divmod(int(idx), n_genes)

        feat_row = int(self.gene_indices[gene_idx])
        # mmap'd arrays are read-only; copy so the resulting tensor is writable.
        x = torch.from_numpy(np.array(self.X_feats_per_ind[ind_idx][feat_row], dtype=np.float32))
        y = torch.from_numpy(self.y_tensor[ind_idx, :, gene_idx].copy())

        if self.norm_features is not None:
            x = x.clone()
            mask = self.norm_features == 1.0
            x[mask] = torch.log(1 + x[mask])
        if self.normalize == "log":
            y = torch.log(1 + y)
        return x, y


if __name__ == "__main__":
    # example usage
    dataset = MddDataset(X_feats=Path("X_sub.features.npy"), X_ensids=Path("X_sub.ensids.npy"), X_chroms=Path("X_sub.chroms.npy"), y=Path("y_sub.csv"), normalize="percentiles")
    print(f"Dataset size: {len(dataset)}")
    for i in range(len(dataset)):
        input_data, label = dataset[i]
        print(f"Input shape at index {i}: {input_data.shape}, Label: {label.shape}")
    print(dataset.y.head())
    print("Splitting dataset by chromosomes 1 and 2...")
    new_dataset = dataset.split_by_chromosome(['1', '2'])
    print(f"New dataset size (chromosomes 1 and 2): {len(new_dataset)}")
    print(new_dataset.y.head())