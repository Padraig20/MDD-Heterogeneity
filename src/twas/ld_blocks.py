from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

"""
ld_blocks.py

Approximately independent LD blocks, and how far a TWAS hit list spreads across
them.

Counting significant genes alone overstates how many independent signals a TWAS
found: neighbouring genes in one LD block share the same variants and are
routinely all significant off a single causal locus. Berisa and Pickrell (2016,
Bioinformatics 32:283-285, "Approximately independent linkage disequilibrium
blocks in human populations") partitioned the genome into blocks with little LD
across their boundaries, which makes "how many distinct blocks are implicated"
a far better count of independent signals. This is the quantity scPrediXcan
reports as "129 candidate causal genes from 24 different LD blocks among 1,703
pre-defined LD blocks".

Block definitions come from https://bitbucket.org/nygcresearch/ldetect-data as
a three-column BED (`chr start stop`), one file per population:

    EUR/fourier_ls-all.bed   1703 blocks
    AFR/fourier_ls-all.bed   2582 blocks
    ASN/fourier_ls-all.bed   1445 blocks

Genome build
------------
Those files are **hg19**. Gene positions here come from the bp coordinates in
the reference panel's .bim, so the blocks must be in the *reference panel's*
build, not the build of whatever GTF the expression targets were annotated
with. hg19 and hg38 coordinates differ by far more than a typical block is
wide, so mixing them would silently scramble exactly the number this module
exists to compute. `load_ld_blocks` therefore requires the build to be declared
and `require_matching_build` refuses a mismatch outright.

To use these blocks against an hg38 reference panel, lift the BED over first
(the repository already carries `src/preprocessing/liftover_vcf_hg19_to_hg38.sh`
for the equivalent VCF operation).
"""

# Block counts published by Berisa and Pickrell, used only to sanity-check a
# file the user points us at.
KNOWN_BLOCK_COUNTS = {1703: "EUR", 2582: "AFR", 1445: "ASN"}

BUILD_ALIASES = {
    "hg19": "hg19",
    "grch37": "hg19",
    "b37": "hg19",
    "37": "hg19",
    "hg38": "hg38",
    "grch38": "hg38",
    "b38": "hg38",
    "38": "hg38",
}

UNASSIGNED = -1


def normalize_build(build: Optional[str]) -> Optional[str]:
    """Canonical build name, or None when unspecified."""
    if build is None:
        return None
    key = str(build).strip().lower()
    if key in ("", "none", "unknown"):
        return None
    if key not in BUILD_ALIASES:
        raise ValueError(
            f"Unrecognised genome build '{build}'; expected one of "
            f"{sorted(set(BUILD_ALIASES))}."
        )
    return BUILD_ALIASES[key]


def _normalize_chrom(values: Iterable) -> np.ndarray:
    """Strip any 'chr' prefix so block and .bim chromosome names compare equal."""
    return np.array(
        [str(value).strip().lower().removeprefix("chr") for value in values]
    )


@dataclass
class LdBlocks:
    """A partition of the genome into approximately independent LD blocks."""

    name: str
    build: str
    path: Optional[Path]
    frame: pd.DataFrame  # chrom, start, stop, block_index, block

    def __post_init__(self) -> None:
        self._chrom_index: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for chrom, block in self.frame.groupby("chrom", sort=False):
            block = block.sort_values("start")
            self._chrom_index[chrom] = (
                block["start"].to_numpy(dtype=np.int64),
                block["stop"].to_numpy(dtype=np.int64),
                block["block_index"].to_numpy(dtype=np.int64),
            )

    @property
    def n_blocks(self) -> int:
        return int(len(self.frame))

    def assign(self, chrom: str, bp: int) -> int:
        """
        The block containing a position, or `UNASSIGNED`.

        BED intervals are half-open, so a position exactly on a boundary
        belongs to the block that starts there.
        """
        entry = self._chrom_index.get(str(chrom).strip().lower().removeprefix("chr"))
        if entry is None:
            return UNASSIGNED
        starts, stops, indices = entry
        position = int(np.searchsorted(starts, bp, side="right")) - 1
        if position < 0 or bp >= stops[position]:
            return UNASSIGNED
        return int(indices[position])

    def assign_frame(
        self, positions: dict[str, tuple[str, int]], genes: Sequence[str]
    ) -> pd.DataFrame:
        """`gene -> block_index/block label` for the given genes."""
        rows = []
        for gene in genes:
            chrom, bp = positions.get(gene, (None, None))
            if chrom is None:
                rows.append((gene, UNASSIGNED, None))
                continue
            index = self.assign(chrom, int(bp))
            label = None if index == UNASSIGNED else self.frame.loc[index, "block"]
            rows.append((gene, index, label))
        return pd.DataFrame(rows, columns=["gene", "block_index", "block"])


def load_ld_blocks(
    path: Path, build: str, name: Optional[str] = None
) -> LdBlocks:
    """
    Read a three-column BED of LD blocks.

    Accepts the ldetect-data layout directly: an optional `chr start stop`
    header, whitespace separated, chromosome names with or without the 'chr'
    prefix.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LD block file not found: {path}")
    canonical = normalize_build(build)
    if canonical is None:
        raise ValueError(
            "The genome build of the LD block file must be stated explicitly "
            "(--ld-blocks-build); hg19 and hg38 block coordinates are not "
            "interchangeable."
        )

    frame = pd.read_csv(
        path, sep=r"\s+", header=None, names=["chrom", "start", "stop"],
        comment="#", dtype=str, skip_blank_lines=True,
    )
    # Drop the header row if the file carries one.
    if not frame.empty and not str(frame.iloc[0]["start"]).strip().isdigit():
        frame = frame.iloc[1:]
    if frame.empty:
        raise ValueError(f"{path} contains no LD blocks.")

    frame = frame.assign(
        chrom=_normalize_chrom(frame["chrom"]),
        start=frame["start"].astype(np.int64),
        stop=frame["stop"].astype(np.int64),
    )
    invalid = frame["stop"] <= frame["start"]
    if invalid.any():
        raise ValueError(f"{path} has {int(invalid.sum())} block(s) with stop <= start.")

    frame = frame.sort_values(["chrom", "start"]).reset_index(drop=True)
    frame["block_index"] = np.arange(len(frame), dtype=np.int64)
    frame["block"] = (
        "chr" + frame["chrom"] + ":" + frame["start"].astype(str)
        + "-" + frame["stop"].astype(str)
    )

    population = KNOWN_BLOCK_COUNTS.get(len(frame))
    label = name or (
        f"Berisa-Pickrell {population}" if population else path.stem
    )
    logging.info(
        "Loaded %d LD block(s) from %s (%s, build %s).",
        len(frame), path.name, label, canonical,
    )
    return LdBlocks(name=label, build=canonical, path=path, frame=frame)


def require_matching_build(blocks: LdBlocks, reference_build: Optional[str]) -> None:
    """
    Refuse to assign genes to blocks from a different genome build.

    This is a hard error rather than a warning on purpose: the failure is
    silent and produces a plausible-looking but meaningless block count.
    """
    canonical = normalize_build(reference_build)
    if canonical is None:
        raise ValueError(
            "--genotype-build must be given alongside --ld-blocks so the block "
            "coordinates can be checked against the reference panel's. The gene "
            "positions used for block assignment come from the reference .bim, "
            "so it is that panel's build that has to match (UK Biobank imputation "
            "v3 is hg19 unless it was lifted over; OneK1K here is hg38)."
        )
    if canonical != blocks.build:
        raise ValueError(
            f"The LD blocks are {blocks.build} but the reference panel is "
            f"{canonical}. These coordinate systems differ by more than a typical "
            "LD block is wide, so the block assignment would be meaningless. Lift "
            "the block BED over to the reference's build first."
        )


def block_metrics(
    frame: pd.DataFrame,
    blocks: LdBlocks,
    mask: Optional[pd.Series] = None,
    prefix: str = "",
) -> dict:
    """
    How many distinct LD blocks a set of genes covers.

    `genes_per_block` is the headline fine-mapping number: it is how many genes
    the method implicates per independent signal, so driving it towards 1 while
    holding `n_blocks` steady is exactly what a useful fine-mapper does.
    """
    selected = frame if mask is None else frame[mask]
    assigned = selected["block_index"] if "block_index" in selected.columns else pd.Series(dtype=np.int64)
    assigned = assigned[assigned != UNASSIGNED].dropna()

    n_genes = int(len(selected))
    n_blocks = int(assigned.nunique())
    counts = assigned.value_counts()
    return {
        f"{prefix}n_genes": n_genes,
        f"{prefix}n_ld_blocks": n_blocks,
        f"{prefix}n_ld_blocks_total": blocks.n_blocks,
        f"{prefix}frac_ld_blocks": (n_blocks / blocks.n_blocks) if blocks.n_blocks else float("nan"),
        f"{prefix}genes_per_ld_block": (len(assigned) / n_blocks) if n_blocks else float("nan"),
        f"{prefix}max_genes_in_one_ld_block": int(counts.max()) if n_blocks else 0,
        f"{prefix}n_genes_outside_ld_blocks": int(n_genes - len(assigned)),
    }


def agreement_block_curve(
    frame: pd.DataFrame,
    blocks: LdBlocks,
    agreement_column: str = "agreement_bonferroni",
    thresholds: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """
    Gene and LD-block counts as the model-agreement requirement is tightened.

    This is the fine-mapping evidence: if raising the fraction of
    member-bootstrap fits that must call a gene significant drops the gene count
    while leaving the block count intact, the ensemble is discarding redundant
    genes *within* loci rather than losing loci, i.e. it is resolving which gene
    in a block carries the signal.
    """
    if agreement_column not in frame.columns:
        raise KeyError(f"{agreement_column} is not present; run in --model-kind mi.")
    thresholds = list(thresholds) if thresholds is not None else [
        0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0
    ]

    agreement = frame[agreement_column].fillna(0.0)
    baseline = frame[agreement > 0.0]
    baseline_blocks = baseline.loc[
        baseline["block_index"] != UNASSIGNED, "block_index"
    ].nunique()

    rows = []
    for threshold in thresholds:
        # `> 0` means "significant in at least one fit"; every other threshold
        # is inclusive, so 0.8 keeps genes that at least 80% of fits agree on.
        mask = agreement > 0.0 if threshold == 0.0 else agreement >= threshold
        metrics = block_metrics(frame, blocks, mask=mask)
        rows.append({
            "threshold": threshold,
            "n_genes": metrics["n_genes"],
            "n_ld_blocks": metrics["n_ld_blocks"],
            "n_ld_blocks_total": blocks.n_blocks,
            "genes_per_ld_block": metrics["genes_per_ld_block"],
            "max_genes_in_one_ld_block": metrics["max_genes_in_one_ld_block"],
            "frac_genes_retained": (
                metrics["n_genes"] / len(baseline) if len(baseline) else float("nan")
            ),
            "frac_ld_blocks_retained": (
                metrics["n_ld_blocks"] / baseline_blocks
                if baseline_blocks else float("nan")
            ),
        })
    return pd.DataFrame(rows)


__all__ = [
    "KNOWN_BLOCK_COUNTS",
    "LdBlocks",
    "UNASSIGNED",
    "agreement_block_curve",
    "block_metrics",
    "load_ld_blocks",
    "normalize_build",
    "require_matching_build",
]
