from __future__ import annotations

import hashlib
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

"""
reference.py

The LD reference panel: the PLINK cohort the covariances are computed from.

The genotype layout and the individual-id conventions mirror
`src/distillation/train.py`, so the reference used for TWAS is the same cohort
the models were distilled on. The constants are duplicated rather than imported
because importing `train.py` pulls in `src/distillation/wandb_logger.py`, which
raises at import time unless WANDB_KEY is set.

Only the SNPs that actually appear in some model are indexed. A full UKB
imputed panel has tens of millions of variants, and a dict over all of them
would cost several GB for no benefit.
"""

ONEK1K_BED_TEMPLATE = "OneK1K.GrCH38_chr{chrom}.biallelic"
UKB_BED_TEMPLATE = "ukb_imp_v3_chr{chrom}.unrelatedbritishqced.maf001geno9.biallelic"

BED_TEMPLATES = {
    "OneK1K": ONEK1K_BED_TEMPLATE,
    "UKB": UKB_BED_TEMPLATE,
}

CHROMOSOMES = [str(c) for c in range(1, 23)]

# bed_reader's missing sentinel for the int8 dtype.
MISSING_INT8 = -127


@dataclass(frozen=True)
class SnpRecord:
    """Where one model SNP lives in the reference, and how it is coded."""

    chrom: str
    var_index: int  # column index into the .bed for this chromosome
    bp: int
    # `effect_allele` is the allele bed_reader counts, i.e. .bim column 5.
    # Verified empirically: a genotype coded homozygous-for-the-first-allele
    # reads back as 2.
    effect_allele: str
    non_effect_allele: str


def impute_to_float(values: np.ndarray) -> np.ndarray:
    """
    Convert a raw int8 dosage block to float64, replacing the missing sentinel
    with each variant's mean.

    Mean imputation matches how `GenotypeDataset` treats missing calls when it
    computes MAF, and keeps a missing call from biasing the covariance.
    """
    dosages = values.astype(np.float64)
    missing = dosages == MISSING_INT8
    if missing.any():
        dosages[missing] = np.nan
        with np.errstate(invalid="ignore"):
            column_means = np.nanmean(dosages, axis=0)
        column_means = np.where(np.isfinite(column_means), column_means, 0.0)
        dosages[missing] = np.take(column_means, np.nonzero(missing)[1])
    return dosages


class Reference:
    """
    A PLINK cohort restricted to a fixed set of individuals and SNPs.

    This is an index, not a reader: it says where each model SNP sits in which
    .bed and which rows of that file the selected individuals occupy, and the
    dosages themselves are read by the covariance workers in
    `src/twas/covariance.py`. Keeping it read-free is what lets it be built
    once in the parent process and used from many.
    """

    def __init__(
        self,
        genotype_dir: Path,
        bed_template: str,
        snp_index: dict[str, SnpRecord],
        row_index: dict[str, np.ndarray],
        individuals: list[str],
        selection: dict,
        counts: dict[str, tuple[int, int]],
        fam_ids: dict[str, np.ndarray],
    ) -> None:
        self.genotype_dir = Path(genotype_dir)
        self.bed_template = bed_template
        self.snp_index = snp_index
        self.row_index = row_index
        self.individuals = individuals
        self.selection = selection
        # chromosome -> (iid_count, sid_count), taken from the .fam/.bim we
        # already parsed. Handing these to open_bed is what makes opening
        # cheap: without them it counts the .bim's lines every single time,
        # which on a UKB chromosome costs far more than the read itself.
        self.counts = counts
        # Kept so `with_individuals` can re-derive the row index without
        # touching the .bim files again, which is the expensive part.
        self.fam_ids = fam_ids

    @property
    def n_individuals(self) -> int:
        return len(self.individuals)

    def bed_path(self, chrom: str) -> Path:
        return bed_path(self.genotype_dir, self.bed_template, chrom)

    def with_individuals(self, individuals: list[str], source: str) -> "Reference":
        """
        The same SNP index restricted to a different cohort.

        Cell types are distilled on whichever donors their target CSV covers,
        which need not be the same across cell types, so the cohort is resolved
        per cell type. Only the row index changes; the SNP index -- the part
        that costs 22 .bim passes -- is shared.

        Individuals absent from the .fam are dropped rather than raising, which
        is what `GenotypeDataset.gene_design` does when it maps target columns
        onto BED rows.
        """
        row_index, kept = _row_index_for(self.fam_ids, individuals)
        missing = len(individuals) - len(kept)
        if missing:
            logging.warning(
                "%d of %d individual(s) from %s are absent from the genotype "
                ".fam and were dropped, leaving %d. `train.py` drops the same "
                "ones, so this matches training.",
                missing, len(individuals), source, len(kept),
            )
        if not kept:
            raise ValueError(
                f"None of the {len(individuals)} individual(s) from {source} are "
                "present in the genotype .fam files. Check that "
                "--genotype-template matches the cohort the models were "
                "distilled on."
            )
        return Reference(
            genotype_dir=self.genotype_dir,
            bed_template=self.bed_template,
            snp_index=self.snp_index,
            row_index=row_index,
            individuals=kept,
            counts=self.counts,
            fam_ids=self.fam_ids,
            selection={
                **self.selection,
                "individuals_from": source,
                "num_individuals": str(len(kept)),
                "sample_seed": None,
                "individual_split": None,
                "n_individuals": len(kept),
                "n_individuals_missing_from_fam": missing,
                "individuals_hash": individuals_hash(kept),
            },
        )


def bed_path(genotype_dir: Path, bed_template: str, chrom: str) -> Path:
    """The .bed of one chromosome. Also used by the covariance workers, which
    carry the directory and template rather than a whole `Reference`."""
    return Path(genotype_dir) / f"{bed_template.format(chrom=chrom)}.bed"


def split_contiguous(samples: list[str], split_idx: int, total_splits: int) -> list[str]:
    """The `split_idx`-th of `total_splits` contiguous, near-equal chunks (1-based)."""
    total = len(samples)
    start = (split_idx - 1) * total // total_splits
    stop = split_idx * total // total_splits
    return samples[start:stop]


def apply_individual_split(samples: list[str], selection: str) -> list[str]:
    """Apply a `K/N` contiguous split, mirroring `get_feats_from_seqs.py`."""
    split_parts = selection.split("/")
    if len(split_parts) != 2:
        raise ValueError("--individual-split syntax must be K/N, e.g. 2/4.")
    try:
        split_idx = int(split_parts[0])
        total_splits = int(split_parts[1])
    except ValueError as exc:
        raise ValueError(
            "--individual-split syntax must use integer values, e.g. 2/4."
        ) from exc
    if total_splits <= 0:
        raise ValueError("--individual-split total N must be greater than 0.")
    if split_idx < 1 or split_idx > total_splits:
        raise ValueError("--individual-split K must satisfy 1 <= K <= N.")

    chosen = split_contiguous(samples, split_idx, total_splits)
    if not chosen:
        raise ValueError("--individual-split selected zero individuals.")
    logging.info(
        "Selected post-sampling split %d/%d with %d of %d individuals.",
        split_idx, total_splits, len(chosen), len(samples),
    )
    return chosen


def select_individual_ids(
    samples: list[str],
    selection: str,
    seed: int,
    individual_split: Optional[str] = None,
) -> list[str]:
    """
    Select reference individuals from the .fam order.

    Same syntax as `--num-individuals` in `get_feats_from_seqs.py`: 'all', an
    integer count (randomly sampled with `seed`), or a 'K/N' contiguous split.
    """
    selection = selection.strip().lower()

    if selection == "all":
        chosen = list(samples)
        logging.info("Using all %d reference individuals.", len(chosen))
    elif "/" in selection:
        split_parts = selection.split("/")
        if len(split_parts) != 2:
            raise ValueError("--num-individuals split syntax must be K/N, e.g. 2/4.")
        try:
            split_idx = int(split_parts[0])
            total_splits = int(split_parts[1])
        except ValueError as exc:
            raise ValueError(
                "--num-individuals split syntax must use integer values, e.g. 2/4."
            ) from exc
        if total_splits <= 0:
            raise ValueError("--num-individuals split total N must be greater than 0.")
        if split_idx < 1 or split_idx > total_splits:
            raise ValueError("--num-individuals split K must satisfy 1 <= K <= N.")
        chosen = split_contiguous(samples, split_idx, total_splits)
        logging.info(
            "Selected split %d/%d with %d of %d reference individuals.",
            split_idx, total_splits, len(chosen), len(samples),
        )
    else:
        try:
            num_individuals = int(selection)
        except ValueError as exc:
            raise ValueError(
                "--num-individuals must be an integer, 'all', or K/N, e.g. 2/4."
            ) from exc
        if num_individuals <= 0:
            raise ValueError("--num-individuals must select at least one individual.")
        if num_individuals > len(samples):
            raise ValueError(
                f"Requested {num_individuals} individuals, but the reference only "
                f"contains {len(samples)} samples."
            )
        chosen = random.Random(seed).sample(samples, num_individuals)
        logging.info(
            "Sampled %d reference individuals using seed %d.", len(chosen), seed
        )

    if not chosen:
        raise ValueError("--num-individuals selected zero individuals.")
    if individual_split is not None:
        chosen = apply_individual_split(chosen, individual_split)
    return chosen


# What `dataset.py` treats as non-individual columns of a target CSV.
TARGET_METADATA_COLUMNS = ("gene", "chrom", "tss")


def individuals_hash(individuals: list[str]) -> str:
    """
    Fingerprint of *which* individuals a covariance was estimated on.

    Order-independent, because a covariance does not depend on the order of the
    rows. Recorded in the covariance metadata so two model directories can be
    checked for having used the same people without comparing the paths the
    lists happened to come from -- the two arms may well read different target
    files while covering an identical cohort.
    """
    digest = hashlib.sha256()
    for individual in sorted(individuals):
        digest.update(individual.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def find_target_csv(targets: Path, cell_type: str) -> Path:
    """
    The target CSV of one cell type, from a file or a `--targets` directory.

    `train.py` names each CSV after the cell type as it is written in the
    single-cell data ('memory B cell.csv') but names the weights JSON with
    underscores ('memory_B_cell.json'), so the two are matched on the folded
    name rather than literally.
    """
    targets = Path(targets)
    if targets.is_file():
        return targets
    if not targets.is_dir():
        raise FileNotFoundError(f"--targets is neither a file nor a directory: {targets}")

    wanted = cell_type.replace(" ", "_").lower()
    for path in sorted(targets.glob("*.csv")):
        if path.stem.replace(" ", "_").lower() == wanted:
            return path
    raise FileNotFoundError(
        f"No target CSV for cell type '{cell_type}' in {targets}. Found: "
        f"{sorted(p.stem for p in targets.glob('*.csv'))[:10]}"
    )


def read_target_individuals(
    path: Path, max_individuals: Optional[int] = None
) -> list[str]:
    """
    The individuals a model was distilled on, read off its target CSV.

    This is the same derivation `GenotypeDataset.__init__` performs, and it is
    the only reliable way to reproduce the training cohort: the individuals
    were chosen by `get_feats_from_seqs.py` sampling the *VCF header* order,
    which a .fam-order sample with the same seed does not reproduce. The
    columns of the target CSV are what actually reached the elastic net, so
    they are taken as the definition.

    Order does not matter for a covariance -- only the set does -- but the
    `max_individuals` truncation does, and it is a prefix of the column order,
    so the column order is preserved here too.
    """
    path = Path(path)
    header = pd.read_csv(path, nrows=0).columns.tolist()

    if "individual" in header:
        # Long format: one row per (gene, individual).
        column = pd.read_csv(path, usecols=["individual"])["individual"]
        individuals = column.astype(str).drop_duplicates().tolist()
    else:
        individuals = [
            str(column) for column in header
            if column not in TARGET_METADATA_COLUMNS
        ]
    if not individuals:
        raise ValueError(f"{path} has no individual columns.")
    if max_individuals is not None:
        individuals = individuals[:max_individuals]
    return individuals


def _read_fam(fam_path: Path, genotype_template: str) -> np.ndarray:
    ids = pd.read_csv(
        fam_path,
        sep=r"\s+",
        header=None,
        usecols=[0, 1],
        names=["family_id", "individual_id"],
    )["individual_id"].to_numpy()
    if genotype_template == "UKB":
        # Same rewrite train.py applies so the ids match the target CSV columns.
        ids = np.array([f"{ind}_{ind}" for ind in ids])
    return ids.astype(str)


def build_reference(
    genotype_dir: Path,
    genotype_template: str,
    wanted_snps: set[str],
    num_individuals: str = "all",
    sample_seed: int = 42,
    individual_split: Optional[str] = None,
    chromosomes: Optional[list[str]] = None,
) -> Reference:
    """
    Index the reference panel, keeping only `wanted_snps`.

    Raises if a chromosome's genotype files are missing, matching how
    `train.py` refuses to train on an incomplete cohort.
    """
    genotype_dir = Path(genotype_dir)
    if genotype_template not in BED_TEMPLATES:
        raise ValueError(
            f"Unknown --genotype-template '{genotype_template}'; "
            f"expected one of {sorted(BED_TEMPLATES)}."
        )
    bed_template = BED_TEMPLATES[genotype_template]
    chromosomes = chromosomes or CHROMOSOMES

    snp_index: dict[str, SnpRecord] = {}
    counts: dict[str, tuple[int, int]] = {}
    all_fam_ids: dict[str, np.ndarray] = {}
    individuals: Optional[list[str]] = None
    duplicates = 0

    for chrom in chromosomes:
        stem = bed_template.format(chrom=chrom)
        bim_path = genotype_dir / f"{stem}.bim"
        fam_path = genotype_dir / f"{stem}.fam"
        bed_path = genotype_dir / f"{stem}.bed"
        for path in (bim_path, fam_path, bed_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing genotype data for chromosome {chrom}: {path}"
                )

        bim = pd.read_csv(
            bim_path,
            sep=r"\s+",
            header=None,
            names=["chrom", "snp", "cm", "bp", "a1", "a2"],
            dtype={"chrom": str, "snp": str, "bp": np.int64},
        )
        keep = bim["snp"].isin(wanted_snps).to_numpy()
        for var_index in np.flatnonzero(keep):
            row = bim.iloc[var_index]
            snp = str(row["snp"])
            if snp in snp_index:
                duplicates += 1
                continue
            snp_index[snp] = SnpRecord(
                chrom=chrom,
                var_index=int(var_index),
                bp=int(row["bp"]),
                effect_allele=str(row["a1"]),
                non_effect_allele=str(row["a2"]),
            )

        fam_ids = _read_fam(fam_path, genotype_template)
        all_fam_ids[chrom] = fam_ids
        if individuals is None:
            individuals = select_individual_ids(
                list(fam_ids), num_individuals, sample_seed, individual_split
            )
        missing = set(individuals) - set(fam_ids.tolist())
        if missing:
            raise ValueError(
                f"{len(missing)} selected individual(s) are absent from {fam_path} "
                f"(first few: {sorted(missing)[:5]}); the cohort must be identical "
                "across chromosomes."
            )
        counts[chrom] = (len(fam_ids), len(bim))
        logging.info(
            "Indexed chromosome %s: %d of %d model SNPs found.",
            chrom, int(keep.sum()), len(wanted_snps),
        )

    if individuals is None:
        raise ValueError("No chromosomes were loaded.")
    if duplicates:
        logging.warning(
            "%d SNP id(s) appear more than once across the reference .bim files; "
            "kept the first occurrence of each.", duplicates,
        )

    found = len(snp_index)
    logging.info(
        "Reference ready: %d of %d model SNPs found, %d individuals.",
        found, len(wanted_snps), len(individuals),
    )
    if found == 0:
        raise ValueError(
            "None of the model SNPs were found in the reference .bim files. Check "
            "that --genotype-template matches the genotype filenames and that the "
            "models were trained against this cohort."
        )

    row_index, kept = _row_index_for(all_fam_ids, individuals)
    return Reference(
        genotype_dir=genotype_dir,
        bed_template=bed_template,
        snp_index=snp_index,
        row_index=row_index,
        individuals=kept,
        counts=counts,
        fam_ids=all_fam_ids,
        selection={
            "genotype_dir": str(genotype_dir),
            "genotype_template": genotype_template,
            "individuals_from": "--num-individuals sample of the .fam order",
            "num_individuals": num_individuals,
            "sample_seed": sample_seed,
            "individual_split": individual_split,
            "n_individuals": len(kept),
            "individuals_hash": individuals_hash(kept),
        },
    )


def _row_index_for(
    fam_ids: dict[str, np.ndarray], individuals: list[str]
) -> tuple[dict[str, np.ndarray], list[str]]:
    """
    Per-chromosome .bed row numbers for the individuals present in every .fam.

    An individual missing from any chromosome is dropped from all of them, so
    every chromosome's covariance is estimated on the same people.
    """
    present = set(individuals)
    for ids in fam_ids.values():
        present &= set(ids.tolist())
    kept = [ind for ind in individuals if ind in present]

    row_index = {}
    for chrom, ids in fam_ids.items():
        positions = {ind: index for index, ind in enumerate(ids)}
        row_index[chrom] = np.array(
            [positions[ind] for ind in kept], dtype=np.int64
        )
    return row_index, kept


def default_genotype_template(genotype_dir: Path) -> Optional[str]:
    """Guess the template from which chromosome-1 files exist, or None."""
    for name, template in BED_TEMPLATES.items():
        if (Path(genotype_dir) / f"{template.format(chrom='1')}.bed").exists():
            return name
    return None


__all__ = [
    "BED_TEMPLATES",
    "CHROMOSOMES",
    "Reference",
    "SnpRecord",
    "TARGET_METADATA_COLUMNS",
    "apply_individual_split",
    "bed_path",
    "build_reference",
    "default_genotype_template",
    "find_target_csv",
    "impute_to_float",
    "read_target_individuals",
    "select_individual_ids",
    "split_contiguous",
]
