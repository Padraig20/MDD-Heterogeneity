import os
import threading
import weakref

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

# Per-chromosome lookups derived from the `bims`/`idx2ind` inputs. Those inputs are
# read once per run and then handed to one dataset per cell type, so deriving them
# again for every dataset would repeat identical work (and, for the individual ->
# BED row maps of a large cohort, allocate the same millions of dict entries over
# and over). Keyed by object identity, since neither DataFrames nor ndarrays hash.
_BP_CACHE: dict[int, tuple] = {}
_SNP_ID_CACHE: dict[int, tuple] = {}
_IND_INDEX_CACHE: dict[int, tuple] = {}


def _cached_by_identity(cache: dict, owner, build):
    """
    Memoize `build()` against the identity of `owner`.

    A weak reference to `owner` is stored alongside the value so that a recycled
    `id()` (possible once the original object is garbage-collected) can never
    return another object's cached value.
    """
    key   = id(owner)
    entry = cache.get(key)
    if entry is not None:
        ref, value = entry
        if ref() is owner:
            return value
        del cache[key]

    value = build()
    try:
        cache[key] = (weakref.ref(owner), value)
    except TypeError:
        pass  # not weak-referenceable: skip caching rather than risk a stale hit
    return value


class GenotypeDataset(Dataset):

    # Variants per allele-frequency block (see `_block_maf`). Large enough that a
    # 2 Mb cis-window spans only a couple of blocks, small enough that filling one
    # stays a modest read.
    MAF_BLOCK = 262_144

    def __init__(
        self,
        bims: dict[str],
        idx2ind: dict[np.ndarray],
        y: Path | pd.DataFrame,
        bim_dir: str = "",
        window_size=500_000,
        select_genes: Path | None = None,
        normalize: str = "log",
        max_individuals: int | None = None,
        bed_template: str = DEFAULT_BED_TEMPLATE,
        maf_threshold: float | None = None,
        y_aleatoric: Path | pd.DataFrame | None = None,
        y_epistemic: Path | pd.DataFrame | None = None,
        min_detected_frac: float | None = None,
        min_expr_std: float | None = None,
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
            y_aleatoric (Path | pd.DataFrame | None): Optional per-(gene, individual) teacher *aleatoric*
                                          variance targets, in the same wide format as `y`
                                          (gene,chrom,tss,individual1,...), e.g. the 'aleatoric' output of
                                          get_student_data.py. Used together with `y_epistemic` for
                                          probabilistic distillation, where the teacher's total predictive
                                          variance is distilled as two separate targets rather than one
                                          combined number. Variances are taken as-is (no normalization),
                                          since they live in the model's output space, matching the
                                          log-transformed mean targets. Must be provided together with
                                          `y_epistemic` (both or neither).
            y_epistemic (Path | pd.DataFrame | None): Optional per-(gene, individual) teacher *epistemic*
                                          variance targets, same format as `y_aleatoric` (e.g. the
                                          'epistemic' output of get_student_data.py). Must be provided
                                          together with `y_aleatoric` (both or neither).
            min_detected_frac (float | None): Minimum fraction of individuals with nonzero raw expression a gene
                                          must have to be kept (e.g. 0.2 requires >=20% of individuals to have
                                          nonzero expression). Guards against near-all-zero, dropout-dominated
                                          genes, whose Pearson r is fragile (a handful of nonzero points dominate
                                          the covariance) even when their rank/Spearman correlation looks fine,
                                          and which otherwise tend to produce degenerate "perfect" R^2 (both
                                          prediction and truth collapse to 0). Computed on *all* individuals
                                          present in `y`, before `max_individuals` truncation, so the decision of
                                          which genes are usable doesn't change with a training-set-size ablation.
                                          If None, no detection filtering is applied.
            min_expr_std (float | None): Minimum standard deviation of log1p(expression) across individuals a
                                          gene must have to be kept. Filters out near-constant genes (no signal
                                          to predict at all). Computed the same way as `min_detected_frac`
                                          (all individuals, pre-truncation). If None, no std filtering is applied.
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
        self.min_detected_frac = min_detected_frac
        self.min_expr_std = min_expr_std
        if (y_aleatoric is None) != (y_epistemic is None):
            raise ValueError(
                "y_aleatoric and y_epistemic must be provided together (both or neither)."
            )
        if isinstance(y, Path) or isinstance(y, str):
            self.y    = pd.read_csv(y)
        else:
            self.y    = y # assume it's already a DataFrame

        # Two accepted input layouts:
        #   wide (the on-disk format): one row per gene, columns
        #     gene,chrom,tss,individual1,individual2,...
        #   long: one row per gene-individual pair, with `individual`/`expression`
        #     columns (what `split_by_chromosome` used to hand on).
        # Everything downstream consumes the per-gene arrays built in
        # `_build_caches`, so the wide layout is kept as-is rather than melted into
        # n_genes x n_individuals rows: melting a whole cell type (~20k genes x a few
        # hundred individuals) costs seconds and a multi-million-row frame per cell
        # type, and every consumer would immediately group it back up by gene.
        self._long_format = "individual" in self.y.columns

        if not self._long_format:
            metadata_cols   = ["gene", "chrom", "tss"]
            individual_cols = [col for col in self.y.columns if col not in metadata_cols]

            # Expression-based gene filtering happens on *all* individual columns,
            # before any `max_individuals` truncation, so the set of usable genes
            # reflects the full population regardless of training-set-size ablations.
            keep_genes = self._genes_passing_expression_filter(
                self.y, individual_cols=individual_cols
            )
            if keep_genes is not None:
                self.y = self.y[self.y["gene"].isin(keep_genes)]

            selected_genes = self._selected_genes(self.y["gene"].unique())
            if selected_genes is not None:
                self.y = self.y[self.y["gene"].isin(selected_genes)]

            if self.max_individuals is not None:
                individual_cols = individual_cols[:self.max_individuals]
            # One row per gene from here on, so gene ids index straight into the
            # expression matrix built below.
            self.y = self.y.loc[
                ~self.y["gene"].duplicated(), metadata_cols + individual_cols
            ].copy()

            self.genes        = self.y["gene"].to_numpy()
            self._individuals = np.asarray(individual_cols, dtype=str)
            self._expression  = self._normalized_expression(self.y[individual_cols])
            self._aleatoric   = (
                None if y_aleatoric is None
                else self._aligned_matrix(y_aleatoric, "aleatoric", individual_cols)
            )
            self._epistemic   = (
                None if y_epistemic is None
                else self._aligned_matrix(y_epistemic, "epistemic", individual_cols)
            )
        else:
            keep_genes = self._genes_passing_expression_filter(self.y, individual_cols=None)
            if keep_genes is not None:
                self.y = self.y[self.y["gene"].isin(keep_genes)].copy()
            if self.max_individuals is not None:
                selected_individuals = self.y["individual"].drop_duplicates().head(self.max_individuals)
                self.y = self.y[self.y["individual"].isin(selected_individuals)].copy()

            self.genes     = self.y["gene"].unique()
            selected_genes = self._selected_genes(self.genes)
            if selected_genes is not None:
                self.genes = np.array([g for g in self.genes if g in selected_genes])
                self.y     = self.y[self.y["gene"].isin(selected_genes)].copy()

        # Build per-chromosome and per-gene caches so that get_gene_matrix is
        # O(log M + W) instead of repeating O(N) scans / dict builds per call.
        self._build_caches()

    def _selected_genes(self, genes: np.ndarray) -> set | None:
        """
        Resolve `select_genes` into the set of gene ids to keep, or None when no
        selection was requested.
        """
        if self.select_genes is None:
            return None
        if self.select_genes == Path("random"):
            # select 227 genes at random (same number as in MDD gene list) for quick testing
            np.random.seed(42)
            if len(genes) < 227:
                return set(genes)
            return set(np.random.choice(genes, size=227, replace=False))
        return set(pd.read_csv(self.select_genes, sep="\t")["ENSID"])

    def _normalized_expression(self, expr_df: pd.DataFrame) -> np.ndarray:
        """
        Normalize a wide (genes x individuals) expression block, vectorized over the
        whole block rather than per gene-individual row.
        """
        expr = expr_df.to_numpy(dtype=np.float64)
        if self.normalize == "percentiles":
            if expr.shape[1] == 0:
                return expr
            # ranked across individuals, separately per gene (i.e. along the row)
            return rankdata(expr, method="average", axis=1) / expr.shape[1]
        elif self.normalize == "log":
            return np.log1p(expr)
        raise ValueError(f"Invalid normalization method: {self.normalize}")

    def _aligned_matrix(
        self,
        extra: Path | pd.DataFrame | str,
        col_name: str,
        individual_cols: list[str],
    ) -> np.ndarray:
        """
        Load an extra per-(gene, individual) target table (e.g. the teacher's
        aleatoric/epistemic variances) and lay it out on exactly this dataset's gene
        rows and individual columns, so it can be indexed row-wise alongside the
        expression matrix.

        Missing genes/individuals become NaN, as with the left join this replaces;
        the models drop those rows per gene. Values are kept in the teacher's output
        space (no log/percentile transform), consistent with the log-transformed mean
        targets used for distillation.
        """
        if isinstance(extra, (Path, str)):
            extra_df = pd.read_csv(extra)
        else:
            extra_df = extra

        if "individual" in extra_df.columns:
            extra_df = extra_df.pivot_table(
                index="gene", columns="individual", values=col_name
            )
        else:
            extra_df = extra_df.drop(columns=["chrom", "tss"], errors="ignore")
            extra_df = extra_df.set_index("gene")

        extra_df = extra_df[~extra_df.index.duplicated(keep="first")]
        extra_df.index   = extra_df.index.astype(str)
        extra_df.columns = extra_df.columns.astype(str)
        extra_df = extra_df.reindex(
            index=[str(gene) for gene in self.genes],
            columns=[str(col) for col in individual_cols],
        )
        return extra_df.to_numpy(dtype=np.float64)

    def _build_caches(self) -> None:
        # Per-chromosome: sorted bp array (BIM files from plink are bp-sorted by
        # chrom), snp id array, individual->row dict. All three are derived from
        # inputs shared across every dataset of a run, so they are memoized against
        # those inputs' identity instead of rebuilt per dataset.
        self._chrom_bps: dict[str, np.ndarray] = {}
        self._chrom_snp_ids: dict[str, np.ndarray] = {}
        for chrom, bim in self.bims.items():
            self._chrom_bps[chrom] = _cached_by_identity(
                _BP_CACHE, bim, lambda bim=bim: bim["bp"].to_numpy()
            )
            self._chrom_snp_ids[chrom] = _cached_by_identity(
                _SNP_ID_CACHE, bim, lambda bim=bim: bim["snp"].astype(str).to_numpy()
            )

        self._ind_to_idx: dict[str, dict[str, int]] = {}
        for chrom, ind_arr in self.idx2ind.items():
            self._ind_to_idx[chrom] = _cached_by_identity(
                _IND_INDEX_CACHE,
                ind_arr,
                lambda ind_arr=ind_arr: {
                    ind: i for i, ind in enumerate(np.asarray(ind_arr).astype(str))
                },
            )

        # Cached allele frequencies, in blocks of variants, so overlapping cis-windows
        # (and __getitem__ over many individuals of the same gene) don't recompute
        # them. Guarded by a lock because genes are fitted from several threads.
        self._maf_cache: dict[tuple[str, int], np.ndarray] = {}
        self._maf_lock = threading.Lock()
        self._cohort_row_cache: dict[str, np.ndarray] = {}

        # Individual -> target row position, for aligning this dataset's targets onto
        # another dataset's rows (see `gene_targets`).
        self._wide_positions: dict[str, int] | None = None
        self._gene_positions: dict[str, dict[str, int]] = {}

        # Per-gene: metadata + the row-individuals/expression (and optional aleatoric/
        # epistemic uncertainty) vectors.
        self._gene_meta: dict[
            str, tuple[str, int, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]
        ] = {}
        if self._long_format:
            self.has_uncertainty = {"aleatoric", "epistemic"}.issubset(self.y.columns)
            if "gene" in self.y.columns and len(self.y) > 0:
                for gene, group in self.y.groupby("gene", sort=False):
                    aleatoric = (
                        group["aleatoric"].to_numpy() if self.has_uncertainty else None
                    )
                    epistemic = (
                        group["epistemic"].to_numpy() if self.has_uncertainty else None
                    )
                    self._gene_meta[gene] = (
                        str(group["chrom"].iloc[0]),
                        int(group["tss"].iloc[0]),
                        group["individual"].astype(str).to_numpy(),
                        group["expression"].to_numpy(),
                        aleatoric,
                        epistemic,
                    )
        else:
            self.has_uncertainty = self._aleatoric is not None and self._epistemic is not None
            chroms = self.y["chrom"].astype(str).to_numpy()
            tss    = self.y["tss"].to_numpy()
            # Every gene shares one individual array (the wide frame's columns), so
            # it is referenced rather than copied per gene.
            for i, gene in enumerate(self.genes):
                self._gene_meta[gene] = (
                    chroms[i],
                    int(tss[i]),
                    self._individuals,
                    self._expression[i],
                    self._aleatoric[i] if self._aleatoric is not None else None,
                    self._epistemic[i] if self._epistemic is not None else None,
                )

        # Flat (gene, individual) addressing for the torch Dataset interface.
        self._gene_order   = list(self._gene_meta.keys())
        counts             = [len(self._gene_meta[gene][3]) for gene in self._gene_order]
        self._pair_offsets = np.cumsum([0] + counts)

        # The individuals this dataset actually models, used for allele frequencies.
        if self._long_format:
            self._cohort_individuals = (
                self.y["individual"].astype(str).unique()
                if "individual" in self.y.columns else np.empty(0, dtype=str)
            )
        else:
            self._cohort_individuals = self._individuals

    def _genes_passing_expression_filter(
        self,
        df: pd.DataFrame,
        individual_cols: list[str] | None,
    ) -> set | None:
        """
        Compute the set of genes passing `min_detected_frac`/`min_expr_std`, using
        *all* individuals present in `df` (i.e. before any `max_individuals`
        truncation).

        Args:
            df: either the wide-format frame (one column per individual, selected
                via `individual_cols`) or the already-melted long-format frame
                (`gene`/`expression` columns; pass `individual_cols=None`).
            individual_cols: individual column names in `df` (wide format), or
                None to use the melted long format instead.

        Returns:
            The set of gene ids to keep, or None if neither filter is configured
            (i.e. no filtering requested; caller should keep everything).
        """
        if self.min_detected_frac is None and self.min_expr_std is None:
            return None

        if individual_cols is not None:
            expr  = df[individual_cols].to_numpy(dtype=np.float64)
            genes = df["gene"].to_numpy()
            detected_frac = np.mean(expr > 0, axis=1)
            std = np.std(np.log1p(np.clip(expr, 0, None)), axis=1)
        else:
            grouped = df.groupby("gene")["expression"]
            detected_frac = grouped.apply(lambda s: float(np.mean(s.to_numpy() > 0)))
            std           = grouped.apply(lambda s: float(np.std(np.log1p(np.clip(s.to_numpy(), 0, None)))))
            genes         = detected_frac.index.to_numpy()
            detected_frac = detected_frac.to_numpy()
            std           = std.to_numpy()

        keep = np.ones(len(genes), dtype=bool)
        if self.min_detected_frac is not None:
            keep &= detected_frac >= self.min_detected_frac
        if self.min_expr_std is not None:
            keep &= std >= self.min_expr_std

        return set(genes[keep])

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

    def _cohort_rows(self, chrom: str) -> np.ndarray:
        """
        BED row indices of this dataset's individuals on `chrom`, ascending.

        Allele frequencies are a property of the cohort being modelled, so they are
        computed from these rows rather than from every individual in the fileset
        (which, for a fileset far larger than the cohort, also means reading far
        more data than the fit will ever use).
        """
        cached = self._cohort_row_cache.get(chrom)
        if cached is not None:
            return cached

        ind_to_idx = self._ind_to_idx[chrom]
        rows = [
            ind_to_idx[ind]
            for ind in self._cohort_individuals
            if ind in ind_to_idx
        ]
        rows = np.sort(np.asarray(rows, dtype=np.int64))
        self._cohort_row_cache[chrom] = rows
        return rows

    def _block_maf(self, chrom: str, block: int) -> np.ndarray:
        """
        Per-SNP MAF for one fixed-size block of `chrom`'s variants, computed over the
        cohort rows and cached.

        Blocking (rather than caching per cis-window) is what makes MAF filtering
        cheap: neighbouring genes' windows overlap heavily, so a window-keyed cache
        almost never hits and re-reads the same genotypes for every gene, while
        whole-chromosome computation would read far past the windows actually needed.
        """
        key    = (chrom, block)
        cached = self._maf_cache.get(key)
        if cached is not None:
            return cached

        with self._maf_lock:
            # Another thread may have filled this block while we waited.
            cached = self._maf_cache.get(key)
            if cached is not None:
                return cached

            n_variants = len(self._chrom_bps[chrom])
            start      = block * self.MAF_BLOCK
            stop       = min(start + self.MAF_BLOCK, n_variants)
            bed_path   = os.path.join(
                self.bim_dir, f"{self.bed_template.format(chrom=chrom)}.bed"
            )
            rows = self._cohort_rows(chrom)
            with open_bed(bed_path) as bed:
                genotypes = bed.read(index=np.s_[rows, start:stop], dtype="int8")
            maf = self._maf_from_genotypes(genotypes)
            self._maf_cache[key] = maf
            return maf

    def _window_maf_mask(self, chrom: str, var_idx: np.ndarray) -> np.ndarray:
        """
        Boolean keep-mask (len == len(var_idx)) for SNPs in a window whose MAF,
        computed across the loaded cohort, is >= ``self.maf_threshold``.
        """
        maf         = np.empty(var_idx.size, dtype=np.float64)
        first_block = int(var_idx[0]) // self.MAF_BLOCK
        last_block  = int(var_idx[-1]) // self.MAF_BLOCK
        for block in range(first_block, last_block + 1):
            start     = block * self.MAF_BLOCK
            stop      = start + self.MAF_BLOCK
            in_block  = (var_idx >= start) & (var_idx < stop)
            if not in_block.any():
                continue
            maf[in_block] = self._block_maf(chrom, block)[var_idx[in_block] - start]
        return maf >= self.maf_threshold

    def _cis_window(self, chrom: str, tss: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Variant indices and SNP ids of the cis-window around `tss`, MAF-filtered if
        requested. The BIM is bp-sorted per chromosome in plink output, so the window
        bounds come from two binary searches instead of a scan.
        """
        bps      = self._chrom_bps[chrom]
        all_snps = self._chrom_snp_ids[chrom]
        left     = int(np.searchsorted(bps, tss - self.window_size, side="left"))
        right    = int(np.searchsorted(bps, tss + self.window_size, side="right"))
        var_idx  = np.arange(left, right, dtype=np.int64)
        snp_ids  = all_snps[left:right]

        if self.maf_threshold is not None and var_idx.size > 0:
            maf_keep = self._window_maf_mask(chrom, var_idx)
            var_idx  = var_idx[maf_keep]
            snp_ids  = snp_ids[maf_keep]
        return var_idx, snp_ids

    def split_by_chromosome(self, chroms: list[str]) -> Dataset:
        """
        Restrict the dataset to `chroms`.

        `self.y` holds the targets as they were read (normalization lives in the
        derived per-gene arrays), so the sub-dataset re-derives them from raw values
        rather than normalizing already-normalized ones.
        """
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
            min_detected_frac=self.min_detected_frac,
            min_expr_std=self.min_expr_std,
        )
    
    def __len__(self) -> int:
        return int(self._pair_offsets[-1])

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """
        One (genotype window, expression) pair, addressed gene-major: index `idx`
        walks all individuals of the first gene, then all individuals of the next,
        and so on.
        """
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        gene_pos = int(np.searchsorted(self._pair_offsets, idx, side="right") - 1)
        gene     = self._gene_order[gene_pos]
        local    = int(idx - self._pair_offsets[gene_pos])

        chrom, tss, individuals, expression, _, _ = self._gene_meta[gene]
        individual_idx = self._ind_to_idx[chrom][str(individuals[local])]

        var_idx, _ = self._cis_window(chrom, tss)

        bed_path = os.path.join(self.bim_dir, f"{self.bed_template.format(chrom=chrom)}.bed")
        with open_bed(bed_path) as bed:
            x = bed.read(index=np.s_[individual_idx, var_idx], dtype="int8")

        x = torch.from_numpy(x).flatten().float() # convert to float for training
        y = torch.tensor(expression[local], dtype=torch.float32)

        return x, y
    
    def get_gene_matrix(
        self,
        gene: str,
        return_uncertainty: bool = False,
        return_individuals: bool = False,
    ) -> (
        tuple[np.ndarray, np.ndarray, np.ndarray, str]
        | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]
        | tuple[np.ndarray, np.ndarray, np.ndarray, str, np.ndarray]
        | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, np.ndarray]
    ):
        """
        Build the full design matrix for one gene.

        Uses pre-built caches from `_build_caches`:
        - per-chromosome sorted bp + searchsorted for the cis window (O(log M))
        - per-chromosome individual -> BED row dict (O(1) lookup)
        - per-gene (chrom, tss, individuals, expression, aleatoric, epistemic) tuple (O(1) lookup)

        Args:
            return_uncertainty: if True, also return the per-individual target
                aleatoric/epistemic variances (requires the dataset to have been built
                with both `y_aleatoric` and `y_epistemic`).
            return_individuals: if True, also return the row-aligned individual IDs
                (i.e. `individuals[i]` is the donor for row `i` of `X`/`y`). This is
                the same donor subset actually used to build `X` (individuals absent
                from the BED/FAM for this chromosome are already excluded), so it is
                safe to zip directly against `X`'s rows without re-deriving the
                filtering logic above.

        Returns:
            X            : shape (n_individuals_for_gene, n_snps_in_window)
            y            : shape (n_individuals_for_gene,)
            [y_aleatoric]: shape (n_individuals_for_gene,)  (only if return_uncertainty=True)
            [y_epistemic]: shape (n_individuals_for_gene,)  (only if return_uncertainty=True)
            snp_ids      : shape (n_snps_in_window,)
            chrom        : chromosome name
            [individuals]: shape (n_individuals_for_gene,)  (only if return_individuals=True)
        """
        X, snp_ids, chrom, kept_individuals = self.gene_design(gene)
        y, y_aleatoric, y_epistemic = self.gene_targets(gene, kept_individuals)

        if return_uncertainty and (y_aleatoric is None or y_epistemic is None):
            raise ValueError(
                "Uncertainty targets requested but this dataset was built without "
                "`y_aleatoric`/`y_epistemic`."
            )

        if return_uncertainty and return_individuals:
            return X, y, y_aleatoric, y_epistemic, snp_ids, chrom, kept_individuals
        if return_uncertainty:
            return X, y, y_aleatoric, y_epistemic, snp_ids, chrom
        if return_individuals:
            return X, y, snp_ids, chrom, kept_individuals
        return X, y, snp_ids, chrom

    def gene_design(self, gene: str) -> tuple[np.ndarray, np.ndarray, str, np.ndarray]:
        """
        Genotype side of one gene's design matrix: `(X, snp_ids, chrom, individuals)`.

        Split out from `get_gene_matrix` so a single read can be shared by several
        cell types (see `shares_individuals_with`), whose targets differ while their
        genotypes are identical.
        """
        meta = self._gene_meta.get(gene)
        if meta is None:
            raise ValueError(f"No data found for gene {gene}")
        chrom, tss, row_individuals, _, _, _ = meta

        # Cis window (MAF-filtered if enabled) via sorted-bp searchsorted.
        var_idx, snp_ids = self._cis_window(chrom, tss)

        # Map individuals to BED/FAM row indices via cached dict.
        ind_to_idx = self._ind_to_idx[chrom]
        individual_idx = np.fromiter(
            (ind_to_idx.get(ind, -1) for ind in row_individuals),
            dtype=np.int64,
            count=len(row_individuals),
        )
        keep             = individual_idx >= 0
        kept_individuals = row_individuals
        if not keep.all():
            individual_idx   = individual_idx[keep]
            kept_individuals = row_individuals[keep]

        bed_path = os.path.join(
            self.bim_dir,
            f"{self.bed_template.format(chrom=chrom)}.bed",
        )
        with open_bed(bed_path) as bed:
            X = bed.read(index=np.s_[individual_idx, var_idx], dtype="int8")
        return X, snp_ids, chrom, kept_individuals

    def has_gene(self, gene: str) -> bool:
        """Whether this dataset carries targets for `gene`."""
        return gene in self._gene_meta

    def gene_targets(
        self, gene: str, individuals: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """
        Targets for one gene as `(y, y_aleatoric, y_epistemic)`, row-aligned to
        `individuals` (defaults to this dataset's own individuals for the gene).

        Individuals this dataset has no value for become NaN, which the models drop
        per gene.
        """
        meta = self._gene_meta.get(gene)
        if meta is None:
            raise ValueError(f"No data found for gene {gene}")
        _, _, row_individuals, y, y_aleatoric, y_epistemic = meta

        if individuals is None or individuals is row_individuals:
            return y, y_aleatoric, y_epistemic

        positions = self._individual_positions(gene, row_individuals)
        take      = np.fromiter(
            (positions.get(str(ind), -1) for ind in individuals),
            dtype=np.int64,
            count=len(individuals),
        )
        missing = take < 0
        take    = np.where(missing, 0, take)

        def aligned(values):
            if values is None:
                return None
            out = values[take].astype(np.float64, copy=True)
            out[missing] = np.nan
            return out

        return aligned(y), aligned(y_aleatoric), aligned(y_epistemic)

    def _individual_positions(self, gene: str, row_individuals: np.ndarray) -> dict:
        """Individual -> row position within a gene's target vectors."""
        if not self._long_format:
            # Every gene shares the wide frame's columns, so one map serves them all.
            if self._wide_positions is None:
                self._wide_positions = {
                    str(ind): i for i, ind in enumerate(self._individuals)
                }
            return self._wide_positions

        positions = self._gene_positions.get(gene)
        if positions is None:
            positions = {str(ind): i for i, ind in enumerate(row_individuals)}
            self._gene_positions[gene] = positions
        return positions

    def shares_individuals_with(self, other: "GenotypeDataset") -> bool:
        """
        Whether `other` can be fitted against design matrices read from this dataset.

        Requires the exact same individuals in the exact same order: a differing
        order would silently misalign rows, and a differing membership would
        silently train on the wrong individuals.
        """
        if self._long_format or other._long_format:
            return False
        return (
            self.bed_template == other.bed_template
            and self.bim_dir == other.bim_dir
            and self.window_size == other.window_size
            and self.maf_threshold == other.maf_threshold
            and np.array_equal(self._individuals, other._individuals)
        )

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