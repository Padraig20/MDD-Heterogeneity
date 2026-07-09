import os

import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
from pathlib import Path

from bed_reader import open_bed

from scipy.stats import rankdata

"""
Quick reminder: Make sure to convert the VCF files to BED/BIM/FAM via plink2.

Example command:

plink2 \
  --vcf ALL.chr1.shapeit2_integrated_SNPs_v2a_27022019.GRCh38.phased.vcf.gz \
  --make-bed \
  --out chr1

We assume here that the BIM files have been read into memory (that should fit!)
and are nicely put into a dict with keys according to their chromosome (e.g. "chr1").
"""

# Default PLINK fileset basename template (UKB naming). `{chrom}` is filled in
# with the chromosome label used to key the `bims`/`idx2ind` dicts and the `y`
# `chrom` column. Override via the `bed_template` argument for other cohorts
# (e.g. OneK1K: "OneK1K.GrCH38_chr{chrom}.biallelic").
DEFAULT_BED_TEMPLATE = "ukb_imp_v3_chr{chrom}.unrelatedbritishqced.maf001geno9.biallelic"


class GenotypeDataset(Dataset):
    
    def __init__(
        self,
        bims: dict[str],
        idx2ind: dict[np.ndarray],
        y: Path | pd.DataFrame,
        bim_dir: str = "",
        window_size=1_000_000,
        select_genes: Path | None = None,
        normalize: str = "log",
        max_individuals: int | None = None,
        bed_template: str = DEFAULT_BED_TEMPLATE,
        maf_threshold: float | None = None,
        y_var: Path | pd.DataFrame | None = None,
    ):
        """
        Args:
            bims (dict[str]):             Dictionary of BIM files indexed by chromosome.
            idx2ind (dict[np.ndarray]):   Dictionary mapping chromosome to array of individual indices.
            y (Path | pd.DataFrame): Path to target labels file (csv) or a DataFrame.
            bim_dir (Path):               Directory containing the BIM files.
            window_size (int):            Size of the genomic window to consider around each TSS.
            select_genes (Path | None):   Path to a file containing a list of genes to select. If None, use all genes.
            normalize (str):              Normalization method for expression values. Options are "log" or "percentiles".
            max_individuals (int | None):  Maximum number of individuals to use. If None, use all individuals.
            bed_template (str):           Template for the PLINK fileset basename, with a `{chrom}` placeholder.
            maf_threshold (float | None): Minimum minor allele frequency a SNP must have to be kept (e.g. 0.05
                                          for MAF >= 5%). MAF is computed across the loaded cohort. If None,
                                          no MAF filtering is applied.
            y_var (Path | pd.DataFrame | None): Optional per-(gene, individual) target *variances* in the same
                                          wide format as `y` (gene,chrom,tss,individual1,...). Used for
                                          probabilistic distillation, where the teacher provides a Gaussian
                                          (mean, variance) per target. The variances are taken as-is (no
                                          normalization is applied) since they live in the model's output space,
                                          which matches the log-transformed mean targets. If None, no variance
                                          targets are attached.
        """
        self.bims         = bims
        self.bim_dir      = bim_dir
        self.idx2ind      = idx2ind
        self.window_size  = window_size
        self.select_genes = select_genes
        self.normalize    = normalize
        self.max_individuals = max_individuals
        self.bed_template = bed_template
        self.maf_threshold = maf_threshold
        if isinstance(y, Path) or isinstance(y, str):
            self.y    = pd.read_csv(y)
        else:
            self.y    = y # assume it's already a DataFrame
        # we will slightly change the y format to make training easier...
        # we currently have per row: gene,chrom,tss,individual1,individual2,...
        # we will denormalize this to have one row per gene-individual pair
        # with columns: gene, chrom, tss, individual, expression
        if "individual" not in self.y.columns:
            if self.max_individuals is not None:
                metadata_cols = ["gene", "chrom", "tss"]
                individual_cols = [col for col in self.y.columns if col not in metadata_cols]
                keep_cols = metadata_cols + individual_cols[:self.max_individuals]
                self.y = self.y.loc[:, keep_cols].copy()
            self.y = self.y.melt(id_vars=["gene", "chrom", "tss"], var_name="individual", value_name="expression")
            if self.normalize == "percentiles":
                self.y = self.to_percentiles(self.y)
            elif self.normalize == "log":
                self.y["expression"] = np.log1p(self.y["expression"])
            else:
                raise ValueError(f"Invalid normalization method: {self.normalize}")
            if y_var is not None:
                self.y = self._attach_variance(self.y, y_var)
        elif self.max_individuals is not None:
            selected_individuals = self.y["individual"].drop_duplicates().head(self.max_individuals)
            self.y = self.y[self.y["individual"].isin(selected_individuals)].copy()
        
        # get all different genes
        self.genes = self.y["gene"].unique()
    
        if self.select_genes is not None:
            if self.select_genes == Path("random"):
                # select 227 genes at random (same number as in MDD gene list) for quick testing
                np.random.seed(42)
                if len(self.genes) < 227:
                    selected_genes = self.genes
                else:
                    selected_genes = np.random.choice(self.genes, size=227, replace=False)
            else:
                selected_genes = set(pd.read_csv(self.select_genes, sep="\t")["ENSID"])
            self.genes = np.array([g for g in self.genes if g in selected_genes])
            self.y     = self.y[self.y["gene"].isin(selected_genes)].copy()

        # Build per-chromosome and per-gene caches so that get_gene_matrix is
        # O(log M + W) instead of repeating O(N) scans / dict builds per call.
        self._build_caches()

    def _build_caches(self) -> None:
        # Per-chromosome: sorted bp array (BIM files from plink are bp-sorted by
        # chrom), snp id array, individual->row dict.
        self._chrom_bps: dict[str, np.ndarray] = {}
        self._chrom_snp_ids: dict[str, np.ndarray] = {}
        for chrom, bim in self.bims.items():
            self._chrom_bps[chrom] = bim["bp"].to_numpy()
            self._chrom_snp_ids[chrom] = bim["snp"].astype(str).to_numpy()

        self._ind_to_idx: dict[str, dict[str, int]] = {}
        for chrom, ind_arr in self.idx2ind.items():
            ind_str = np.asarray(ind_arr).astype(str)
            self._ind_to_idx[chrom] = {ind: i for i, ind in enumerate(ind_str)}

        # Cache of per-window MAF keep-masks so repeated reads (e.g. __getitem__
        # over many individuals of the same gene) don't recompute allele freqs.
        self._maf_mask_cache: dict[tuple[str, int, int], np.ndarray] = {}

        # Per-gene: metadata + the row-individuals/expression (and optional variance) vectors.
        self.has_variance = "variance" in self.y.columns
        self._gene_meta: dict[str, tuple[str, int, np.ndarray, np.ndarray, np.ndarray | None]] = {}
        if "gene" in self.y.columns and len(self.y) > 0:
            for gene, group in self.y.groupby("gene", sort=False):
                variance = (
                    group["variance"].to_numpy() if self.has_variance else None
                )
                self._gene_meta[gene] = (
                    str(group["chrom"].iloc[0]),
                    int(group["tss"].iloc[0]),
                    group["individual"].astype(str).to_numpy(),
                    group["expression"].to_numpy(),
                    variance,
                )

    @staticmethod
    def _attach_variance(y_df: pd.DataFrame, y_var: Path | pd.DataFrame | str) -> pd.DataFrame:
        """
        Melt the wide variance table and merge a `variance` column onto the (already
        melted) `y_df` by (gene, chrom, tss, individual).

        Variances are kept in the teacher's output space (no log/percentile transform),
        which is consistent with the log-transformed mean targets used for distillation.
        """
        if isinstance(y_var, (Path, str)):
            var_df = pd.read_csv(y_var)
        else:
            var_df = y_var.copy()

        if "individual" not in var_df.columns:
            var_df = var_df.melt(
                id_vars=["gene", "chrom", "tss"],
                var_name="individual",
                value_name="variance",
            )

        merge_keys = ["gene", "chrom", "tss", "individual"]
        tmp_keys   = [f"__merge_{key}" for key in merge_keys]

        y_df = y_df.copy()
        var_df = var_df[merge_keys + ["variance"]].copy()
        for key, tmp_key in zip(merge_keys, tmp_keys):
            y_df[tmp_key] = y_df[key].astype(str)
            var_df[tmp_key] = var_df[key].astype(str)

        merged = y_df.merge(var_df[tmp_keys + ["variance"]], on=tmp_keys, how="left")
        merged = merged.drop(columns=tmp_keys)
        return merged

    @staticmethod
    def to_percentiles(y_df: pd.DataFrame) -> pd.DataFrame:
        """Convert expression values to percentiles separately for each gene, ranking across individuals."""
        y_df = y_df.copy()
        y_df["expression"] = (
            y_df.groupby("gene")["expression"]
            .transform(lambda col: rankdata(col.to_numpy(), method="average") / len(col))
            .astype(float)
        )
        return y_df

    @staticmethod
    def _maf_from_genotypes(X: np.ndarray) -> np.ndarray:
        """
        Compute per-SNP minor allele frequency from a genotype matrix.

        Args:
            X: shape (n_individuals, n_snps), allele dosages in {0, 1, 2}. Missing
               values (bed_reader returns -127 for int8 / NaN for float) are ignored.

        Returns:
            maf: shape (n_snps,) with the minor allele frequency of each SNP.
        """
        Xf = X.astype(np.float64, copy=True)
        Xf[Xf < 0] = np.nan  # bed_reader missing sentinel (-127 for int8)
        with np.errstate(invalid="ignore"):
            p = np.nanmean(Xf, axis=0) / 2.0
        p = np.where(np.isnan(p), 0.0, p)  # all-missing SNP -> MAF 0 (gets filtered out)
        return np.minimum(p, 1.0 - p)

    def _window_maf_mask(self, chrom: str, var_idx: np.ndarray) -> np.ndarray:
        """
        Boolean keep-mask (len == len(var_idx)) for SNPs in a window whose MAF,
        computed across the whole loaded cohort, is >= ``self.maf_threshold``.

        Results are cached per (chrom, first_var, last_var) window.
        """
        key = (chrom, int(var_idx[0]), int(var_idx[-1]))
        cached = self._maf_mask_cache.get(key)
        if cached is not None:
            return cached

        bed_path = os.path.join(self.bim_dir, f"{self.bed_template.format(chrom=chrom)}.bed")
        with open_bed(bed_path) as bed:
            genotypes = bed.read(index=np.s_[:, var_idx], dtype="int8")
        mask = self._maf_from_genotypes(genotypes) >= self.maf_threshold
        self._maf_mask_cache[key] = mask
        return mask

    def split_by_chromosome(self, chroms: list[str]) -> Dataset:
        # filter y and chroms to only include rows with chrom in chroms
        y_filtered       = self.y[self.y["chrom"].astype(str).isin(chroms)].copy()
        bims_filtered    = {chrom: bim for chrom, bim in self.bims.items() if chrom in chroms}
        idx2ind_filtered = {chrom: idx2ind for chrom, idx2ind in self.idx2ind.items() if chrom in chroms}
        return GenotypeDataset(
            bims_filtered,
            idx2ind_filtered,
            y_filtered,
            bim_dir=self.bim_dir,
            window_size=self.window_size,
            select_genes=self.select_genes,
            normalize=self.normalize,
            max_individuals=self.max_individuals,
            bed_template=self.bed_template,
            maf_threshold=self.maf_threshold,
        )
    
    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        row        = self.y.iloc[idx]
        ensid      = row["gene"]
        chrom      = str(row["chrom"])
        tss        = row["tss"]
        individual = row["individual"]
        expression = row["expression"]

        # find window in BIM file for this chrom that contains tss
        bim    = self.bims[chrom]
        start  = tss - self.window_size
        end    = tss + self.window_size
        idx2ind        = self.idx2ind[chrom]
        individual_idx = np.where(idx2ind == individual)[0][0]

        mask = (bim["chrom"] == chrom) & (bim["bp"] >= start) & (bim["bp"] <= end)
        var_idx = np.flatnonzero(mask.to_numpy())

        if self.maf_threshold is not None and var_idx.size > 0:
            var_idx = var_idx[self._window_maf_mask(chrom, var_idx)]

        bed_path = os.path.join(self.bim_dir, f"{self.bed_template.format(chrom=chrom)}.bed")
        with open_bed(bed_path) as bed:
            x = bed.read(index=np.s_[individual_idx, var_idx], dtype="int8")

        x = torch.from_numpy(x).flatten().float() # convert to float for training
        y = torch.tensor(expression, dtype=torch.float32)

        return x, y
    
    def get_gene_matrix(
        self, gene: str, return_variance: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Build the full design matrix for one gene.

        Uses pre-built caches from `_build_caches`:
        - per-chromosome sorted bp + searchsorted for the cis window (O(log M))
        - per-chromosome individual -> BED row dict (O(1) lookup)
        - per-gene (chrom, tss, individuals, expression, variance) tuple (O(1) lookup)

        Args:
            return_variance: if True, also return the per-individual target variances
                (requires the dataset to have been built with `y_var`).

        Returns:
            X       : shape (n_individuals_for_gene, n_snps_in_window)
            y       : shape (n_individuals_for_gene,)
            [y_var] : shape (n_individuals_for_gene,)  (only if return_variance=True)
            snp_ids : shape (n_snps_in_window,)
            chrom   : chromosome name
        """
        meta = self._gene_meta.get(gene)
        if meta is None:
            raise ValueError(f"No data found for gene {gene}")
        chrom, tss, row_individuals, y, y_var = meta

        if return_variance and y_var is None:
            raise ValueError(
                "Variance targets requested but this dataset was built without `y_var`."
            )

        # Cis window via sorted-bp searchsorted (BIM is bp-sorted by chrom in plink output).
        bps      = self._chrom_bps[chrom]
        all_snps = self._chrom_snp_ids[chrom]
        start    = tss - self.window_size
        end      = tss + self.window_size
        left     = int(np.searchsorted(bps, start, side="left"))
        right    = int(np.searchsorted(bps, end,   side="right"))
        var_idx  = np.arange(left, right, dtype=np.int64)
        snp_ids  = all_snps[left:right]

        # MAF filtering (if enabled) happens before subsetting
        if self.maf_threshold is not None and var_idx.size > 0:
            maf_keep = self._window_maf_mask(chrom, var_idx)
            var_idx  = var_idx[maf_keep]
            snp_ids  = snp_ids[maf_keep]

        # Map individuals to BED/FAM row indices via cached dict.
        ind_to_idx = self._ind_to_idx[chrom]
        individual_idx = np.fromiter(
            (ind_to_idx.get(ind, -1) for ind in row_individuals),
            dtype=np.int64,
            count=len(row_individuals),
        )
        keep = individual_idx >= 0
        if not keep.all():
            individual_idx = individual_idx[keep]
            y = y[keep]
            if y_var is not None:
                y_var = y_var[keep]

        bed_path = os.path.join(
            self.bim_dir,
            f"{self.bed_template.format(chrom=chrom)}.bed",
        )
        with open_bed(bed_path) as bed:
            X = bed.read(index=np.s_[individual_idx, var_idx], dtype="int8")

        if return_variance:
            return X, y, y_var, snp_ids, chrom
        return X, y, snp_ids, chrom

if __name__ == "__main__":
    # example usage
    bim = pd.read_csv(
        "chr1.bim",
        sep=r"\s+",
        header=None,
        names=["chrom", "snp", "cm", "bp", "a1", "a2"],
        dtype={"chrom": str, "snp": str, "bp": np.int64},
    )
    bims    = {"chr1": bim}
    
    idx2ind_arr = pd.read_csv("chr1.fam", sep=r"\s+", header=None, usecols=[0, 1], names=["family_id", "individual_id"])
    idx2ind_arr = idx2ind_arr["individual_id"].to_numpy()
    idx2ind = {"chr1": idx2ind_arr}
    
    y_path  = Path("student-target/0.csv")

    dataset = GenotypeDataset(bims=bims, idx2ind=idx2ind, y=y_path, normalize="percentiles")
    print(f"Dataset size: {len(dataset)}")

    print("Splitting dataset by chromosomes 1...")
    new_dataset = dataset.split_by_chromosome(['chr1'])

    print(f"New dataset size (chromosomes 1): {len(new_dataset)}")

    for i in range(min(5, len(new_dataset))):
        input_data, label = new_dataset[i]
        print(f"Input shape at index {i}: {input_data.shape}, Label: {label}")

    for i in range(min(5, len(new_dataset.genes))):
        gene = new_dataset.genes[i]
        X, y, snp_ids, chr = new_dataset.get_gene_matrix(new_dataset.genes[i])
        print(f"Design matrix shape for gene {gene}: {X.shape}, y shape: {y.shape}, snp_ids shape: {snp_ids.shape}, chromosome: {chr}")
    
    print(new_dataset.y.head())