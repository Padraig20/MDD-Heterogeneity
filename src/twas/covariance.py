from __future__ import annotations

import gzip
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from tqdm import tqdm

from src.twas.reference import Reference
from src.twas.weights import GeneSnps

"""
covariance.py

Build the reference LD covariance S-PrediXcan consumes, straight from PLINK.

`metaxcan/software/M01_covariances_correlations.py` produces the same artifact
but can only read PrediXcan-format dosage *text* files
(`PrediXcanFormatUtilities.PrediXcanFormatDosageLoader`); it has no PLINK path,
and materialising a UKB-sized cohort as dosage text is not practical. This
module writes the identical format directly from the .bed files.

`metax.MatrixManager` is strict about that format:

  * `_validate` rejects duplicated rows and requires each gene's rows to be
    contiguous, so genes are written one complete block at a time.
  * `_to_matrix` indexes `entries[id_i][id_j]` for every pair of retained SNPs,
    so the diagonal plus one triangle must be complete. `_rows_to_entries`
    mirrors each pair, and only ever adds an id it has seen as RSID1, which the
    diagonal guarantees.

Alongside the covariance we write a sidecar SNP table carrying each SNP's bp,
alleles and dosage standard deviation. The standard deviation is exactly
`sqrt(diag(cov))`, but having it (and the alleles) in a small sidecar means a
pre-built `--ld-dir` is self-sufficient: a later run can build model DBs and
recover raw-dosage weights without touching the genotypes again.
"""

COV_HEADER = "GENE\tRSID1\tRSID2\tVALUE\n"
SNP_HEADER = "GENE\tRSID\tCHR\tBP\tEFFECT_ALLELE\tNON_EFFECT_ALLELE\tSD\n"

COV_SUFFIX = ".cov.txt.gz"
SNP_SUFFIX = ".snps.txt.gz"
META_SUFFIX = ".meta.json"


@dataclass(frozen=True)
class GeneSnpInfo:
    """Per-SNP reference facts needed to build a model DB."""

    snp: str
    chrom: str
    bp: int
    effect_allele: str
    non_effect_allele: str
    sd: float


@dataclass
class LdReference:
    """A built (or previously built) covariance for one cell type."""

    cell_type: str
    cov_path: Path
    snp_path: Path
    meta_path: Path
    meta: dict

    def gene_positions(self) -> dict[str, tuple[str, int]]:
        """`gene -> (chrom, midpoint bp)`, for Manhattan plotting."""
        positions = {}
        for gene, info in self.meta["genes"].items():
            positions[gene] = (str(info["chrom"]), int(info["mid_bp"]))
        return positions

    def load_snp_table(self) -> dict[str, dict[str, GeneSnpInfo]]:
        """`gene -> {snp -> GeneSnpInfo}` read back from the sidecar."""
        table: dict[str, dict[str, GeneSnpInfo]] = {}
        with gzip.open(self.snp_path, "rt") as handle:
            header = handle.readline()
            if header != SNP_HEADER:
                raise ValueError(f"Unexpected header in {self.snp_path}: {header!r}")
            for line in handle:
                gene, snp, chrom, bp, effect, non_effect, sd = line.rstrip("\n").split("\t")
                table.setdefault(gene, {})[snp] = GeneSnpInfo(
                    snp=snp,
                    chrom=chrom,
                    bp=int(bp),
                    effect_allele=effect,
                    non_effect_allele=non_effect,
                    sd=float(sd),
                )
        return table


def snp_set_hash(snp_sets: dict[str, GeneSnps]) -> str:
    """
    Fingerprint of the gene -> SNP mapping a covariance was built for.

    A cached covariance is only reusable for a model whose SNP sets are
    identical, so this is what a cache hit is checked against.
    """
    digest = hashlib.sha256()
    for gene in sorted(snp_sets):
        digest.update(gene.encode())
        digest.update(b"\t")
        digest.update(",".join(snp_sets[gene].snp_ids).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def ld_paths(ld_dir: Path, cell_type: str) -> tuple[Path, Path, Path]:
    ld_dir = Path(ld_dir)
    return (
        ld_dir / f"{cell_type}{COV_SUFFIX}",
        ld_dir / f"{cell_type}{SNP_SUFFIX}",
        ld_dir / f"{cell_type}{META_SUFFIX}",
    )


def load_ld_reference(
    ld_dir: Path,
    cell_type: str,
    expected_hash: Optional[str] = None,
) -> LdReference:
    """Load a previously built covariance, validating its SNP-set fingerprint."""
    cov_path, snp_path, meta_path = ld_paths(ld_dir, cell_type)
    for path in (cov_path, snp_path, meta_path):
        if not path.exists():
            raise FileNotFoundError(
                f"No pre-built LD reference for cell type '{cell_type}': {path} is "
                "missing. Pass --genotypes to build it."
            )
    with meta_path.open() as handle:
        meta = json.load(handle)
    if expected_hash is not None and meta.get("snp_set_hash") != expected_hash:
        raise ValueError(
            f"The cached LD reference at {cov_path} was built for a different set "
            "of model SNPs than the weights JSON currently specifies. Rebuild it "
            "with --genotypes, or point --ld-dir somewhere else."
        )
    return LdReference(
        cell_type=cell_type,
        cov_path=cov_path,
        snp_path=snp_path,
        meta_path=meta_path,
        meta=meta,
    )


def _resolve_gene_snps(
    entry: GeneSnps, reference: Reference
) -> tuple[list[str], np.ndarray, list]:
    """
    The gene's model SNPs that exist in the reference, ordered by position.

    Position order matches how `M01` emits its triangle, and gives the
    covariance block a stable, reproducible layout.
    """
    records = []
    for snp in entry.snp_ids:
        record = reference.snp_index.get(snp)
        if record is not None:
            records.append((snp, record))
    if not records:
        return [], np.empty(0, dtype=np.int64), []

    chroms = {record.chrom for _, record in records}
    if len(chroms) > 1:
        raise ValueError(
            f"Gene {entry.gene} has model SNPs on several chromosomes: {sorted(chroms)}."
        )

    records.sort(key=lambda item: item[1].bp)
    snps = [snp for snp, _ in records]
    var_indices = np.array([record.var_index for _, record in records], dtype=np.int64)
    return snps, var_indices, [record for _, record in records]


def build_covariance(
    cell_type: str,
    snp_sets: dict[str, GeneSnps],
    reference: Reference,
    ld_dir: Path,
    max_snps_in_gene: Optional[int] = None,
    overwrite: bool = False,
    show_progress: bool = True,
) -> LdReference:
    """
    Compute and write the per-gene dosage covariance for one cell type.

    One covariance serves every draw of a model: `EnsembleLR.save_coefficients`
    applies the same PIP mask to the pooled weights and to every
    member-bootstrap replicate, so all draws share a gene's SNP set.
    """
    ld_dir = Path(ld_dir)
    ld_dir.mkdir(parents=True, exist_ok=True)
    cov_path, snp_path, meta_path = ld_paths(ld_dir, cell_type)
    fingerprint = snp_set_hash(snp_sets)

    if not overwrite and meta_path.exists():
        try:
            existing = load_ld_reference(ld_dir, cell_type, expected_hash=fingerprint)
        except (FileNotFoundError, ValueError) as error:
            logging.info("Rebuilding LD reference for '%s': %s", cell_type, error)
        else:
            logging.info(
                "Reusing cached LD reference for '%s' (%s).", cell_type, cov_path.name
            )
            return existing

    genes_meta: dict[str, dict] = {}
    n_dropped_genes = 0
    n_missing_snps = 0
    n_constant_snps = 0
    n_skipped_large = 0

    # Genes are grouped by chromosome so the .bed files are touched in order.
    ordered_genes = sorted(
        snp_sets, key=lambda gene: (_chrom_sort_key(snp_sets[gene].chrom), gene)
    )
    iterator: Iterable[str] = ordered_genes
    if show_progress:
        iterator = tqdm(ordered_genes, desc=f"Covariance ({cell_type})", leave=False)

    tmp_cov = cov_path.with_suffix(cov_path.suffix + ".partial")
    tmp_snp = snp_path.with_suffix(snp_path.suffix + ".partial")
    try:
        with gzip.open(tmp_cov, "wt", newline="") as cov_out, gzip.open(
            tmp_snp, "wt", newline=""
        ) as snp_out:
            cov_out.write(COV_HEADER)
            snp_out.write(SNP_HEADER)

            for gene in iterator:
                entry = snp_sets[gene]
                snps, var_indices, records = _resolve_gene_snps(entry, reference)
                n_missing_snps += len(entry.snp_ids) - len(snps)
                if not snps:
                    n_dropped_genes += 1
                    continue
                if max_snps_in_gene is not None and len(snps) > max_snps_in_gene:
                    logging.debug(
                        "Skipping gene %s: %d SNPs exceeds --max-snps-in-gene.",
                        gene, len(snps),
                    )
                    n_skipped_large += 1
                    n_dropped_genes += 1
                    continue

                chrom = records[0].chrom
                dosages = reference.read_dosages(chrom, var_indices)
                cov = np.cov(dosages, rowvar=False)
                cov = np.atleast_2d(cov)
                sds = np.sqrt(np.clip(np.diag(cov), 0.0, None))

                # A monomorphic SNP has zero variance, so it carries no LD
                # information and its standardized-to-raw rescale would divide
                # by zero. Drop it from the block entirely.
                keep = sds > 0.0
                if not keep.all():
                    n_constant_snps += int((~keep).sum())
                    if not keep.any():
                        n_dropped_genes += 1
                        continue
                    kept = np.flatnonzero(keep)
                    cov = cov[np.ix_(kept, kept)]
                    sds = sds[kept]
                    snps = [snps[i] for i in kept]
                    records = [records[i] for i in kept]

                _write_gene_block(cov_out, gene, snps, cov)
                for snp, record, sd in zip(snps, records, sds):
                    snp_out.write(
                        f"{gene}\t{snp}\t{record.chrom}\t{record.bp}\t"
                        f"{record.effect_allele}\t{record.non_effect_allele}\t{sd:.10g}\n"
                    )

                bps = [record.bp for record in records]
                genes_meta[gene] = {
                    "chrom": chrom,
                    "start_bp": int(min(bps)),
                    "end_bp": int(max(bps)),
                    "mid_bp": int((min(bps) + max(bps)) // 2),
                    "n_snps_in_model": len(entry.snp_ids),
                    "n_snps_in_cov": len(snps),
                }

        meta = {
            "cell_type": cell_type,
            "snp_set_hash": fingerprint,
            "n_genes_requested": len(snp_sets),
            "n_genes_written": len(genes_meta),
            "n_genes_dropped": n_dropped_genes,
            "n_model_snps_missing_from_reference": n_missing_snps,
            "n_monomorphic_snps_dropped": n_constant_snps,
            "n_genes_skipped_too_large": n_skipped_large,
            "max_snps_in_gene": max_snps_in_gene,
            "reference": reference.selection,
            "genes": genes_meta,
        }
        with meta_path.open("w") as handle:
            json.dump(meta, handle, indent=2)
        tmp_cov.replace(cov_path)
        tmp_snp.replace(snp_path)
    finally:
        for path in (tmp_cov, tmp_snp):
            if path.exists():
                path.unlink()

    logging.info(
        "Built LD reference for '%s': %d gene(s) written, %d dropped, %d model SNP(s) "
        "absent from the reference, %d monomorphic SNP(s) dropped.",
        cell_type, len(genes_meta), n_dropped_genes, n_missing_snps, n_constant_snps,
    )
    return LdReference(
        cell_type=cell_type,
        cov_path=cov_path,
        snp_path=snp_path,
        meta_path=meta_path,
        meta=meta,
    )


def _write_gene_block(handle, gene: str, snps: list[str], cov: np.ndarray) -> None:
    """
    One gene's complete covariance block: the diagonal plus the upper triangle
    in position order, with no duplicated (RSID1, RSID2) pair.
    """
    n = len(snps)
    for i in range(n):
        handle.write(f"{gene}\t{snps[i]}\t{snps[i]}\t{cov[i, i]:.10g}\n")
        for j in range(i + 1, n):
            handle.write(f"{gene}\t{snps[i]}\t{snps[j]}\t{cov[i, j]:.10g}\n")


def _chrom_sort_key(chrom: str) -> tuple[int, str]:
    text = str(chrom).removeprefix("chr")
    return (int(text), "") if text.isdigit() else (99, text)


__all__ = [
    "COV_HEADER",
    "GeneSnpInfo",
    "LdReference",
    "SNP_HEADER",
    "build_covariance",
    "ld_paths",
    "load_ld_reference",
    "snp_set_hash",
]
