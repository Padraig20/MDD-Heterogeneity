from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import List, Optional

# Make the repository root and the VariantFormer package importable when this
# script is run directly (e.g. `python src/.../validate_consensus_opt.py`) as
# well as via `-m`. VariantFormer exposes flat top-level packages (utils,
# processors, datasets) through its editable install.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_VF_DIR = _REPO_ROOT / "variantformer"
for _p in (_REPO_ROOT, _VF_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

"""
validate_consensus_opt.py

Equivalence check for the batched CRE sequence extraction optimization in
`variantformer/utils/data_process.py`.

For a handful of genes (and one optional individual VCF) it compares, per CRE
window, the sequence produced by:

  * the per-region ground truth: one `samtools faidx | bcftools consensus` call
    per region (`ExtractSeqFromBed.apply_bcftools_consensus`), and
  * the new batched path: a single `samtools faidx | bcftools consensus` call
    over all of a gene's CRE regions (`ExtractSeqFromBed.process_subject` /
    `_extract_sequences_batched`).

The script exits non-zero if any sequence differs, so it can gate the full run.
Provide `--vcf` to validate the personalized path; omit it to validate the
reference path.
"""

from utils.data_process import ExtractSeqFromBed  # noqa: E402
from utils.functions import multi_try_load_csv  # noqa: E402
from processors.vcfprocessor import VCFProcessor  # noqa: E402

log = logging.getLogger("validate_consensus_opt")


def build_bed_regions(vcf_processor: VCFProcessor, gene_id: str):
    """Build the CRE bed_regions frame for a gene, exactly as VCFDataset does."""
    gene_cre_map_path = vcf_processor.gene_cre_manifest.get_file_path(gene_id)
    if gene_cre_map_path is None:
        return None
    genes_cre_map = multi_try_load_csv(gene_cre_map_path)
    bed_regions = genes_cre_map[["chromosome", "start_cre", "end_cre", "cre_name"]]
    bed_regions = bed_regions.rename(
        columns={
            "chromosome": "chrom",
            "start_cre": "start",
            "end_cre": "end",
            "cre_name": "cCRE",
        }
    )
    return bed_regions


def select_gene_ids(vcf_processor: VCFProcessor, num_genes: int, explicit: Optional[List[str]]) -> List[str]:
    if explicit:
        return explicit
    genes_df = vcf_processor.get_genes()
    return [str(g) for g in genes_df["gene_id"].tolist()[: num_genes * 3]]


def get_gene_info(vcf_processor: VCFProcessor, gene_id: str) -> Optional[dict]:
    """Look up a gene's row from the gencode table (chromosome/start/end/strand)."""
    genes_df = vcf_processor.get_genes()
    rows = genes_df[genes_df["gene_id"] == gene_id]
    if len(rows) == 0:
        return None
    return rows.iloc[0].to_dict()


def benchmark_gene(
    cre_extractor: ExtractSeqFromBed,
    gene_extractor: ExtractSeqFromBed,
    vcf_path: Optional[str],
    variant_type: Optional[str],
    bed_regions,
    gene_info: Optional[dict],
) -> tuple[int, float, float]:
    """Time only the optimized batched path for one gene.

    Returns (num_cre_regions, cre_seconds, gene_window_seconds).
    """
    regions = [region for _, region in bed_regions.iterrows()]
    region_strs = [cre_extractor._region_to_str(region) for region in regions]

    t0 = time.perf_counter()
    cre_extractor._extract_sequences_batched(region_strs, vcf_path, variant_type)
    cre_seconds = time.perf_counter() - t0

    gene_seconds = 0.0
    if gene_info is not None:
        t0 = time.perf_counter()
        gene_extractor.process_gene(gene_info, vcf_path, variant_type=variant_type)
        gene_seconds = time.perf_counter() - t0

    return len(regions), cre_seconds, gene_seconds


def compare_gene(
    extractor: ExtractSeqFromBed,
    vcf_path: Optional[str],
    variant_type: Optional[str],
    bed_regions,
) -> tuple[int, int]:
    """Return (num_compared, num_mismatched) for a single gene."""
    regions = [region for _, region in bed_regions.iterrows()]

    # Ground truth: one bcftools call per region.
    expected = []
    for region in regions:
        seq, _ = extractor.apply_bcftools_consensus(
            region, vcf_path, extractor.ref_fasta, variant_type=variant_type
        )
        expected.append(seq or "")

    # Optimized: one batched bcftools call for all regions.
    region_strs = [extractor._region_to_str(region) for region in regions]
    batched = extractor._extract_sequences_batched(region_strs, vcf_path, variant_type)
    if batched is None:
        raise RuntimeError(
            "Batched extraction returned None (subprocess failure or record "
            "count mismatch); cannot validate."
        )

    mismatched = 0
    for region, exp_seq, got_seq in zip(regions, expected, batched):
        if exp_seq != got_seq:
            mismatched += 1
            log.error(
                "MISMATCH cCRE=%s region=%s len(expected)=%d len(batched)=%d",
                getattr(region, "cCRE", "?"),
                extractor._region_to_str(region),
                len(exp_seq),
                len(got_seq),
            )
    return len(regions), mismatched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate batched vs per-region bcftools consensus extraction."
    )
    parser.add_argument(
        "--vcf",
        type=str,
        default=None,
        help="Path to a single-sample, tabix-indexed VCF (personalized path). "
        "Omit to validate the reference path.",
    )
    parser.add_argument(
        "--num-genes",
        type=int,
        default=20,
        help="Number of genes to validate (default: 20).",
    )
    parser.add_argument(
        "--genes",
        type=str,
        nargs="*",
        default=None,
        help="Explicit gene_ids to validate (overrides --num-genes selection).",
    )
    parser.add_argument(
        "--variant-type",
        type=str,
        default=None,
        choices=[None, "SNP"],
        help="Variant type filter passed to bcftools consensus (default: None).",
    )
    parser.add_argument(
        "--model-class",
        type=str,
        default="v4_ag",
        choices=["v4_pcg", "v4_ag"],
        help="VariantFormer model class (selects gencode/CRE config).",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Time ONLY the optimized batched path (no per-region ground truth) "
        "and project per-individual runtime. Use this to gauge real speed.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=28,
        help="DataLoader worker count assumed for the per-individual projection "
        "in --benchmark mode (default: 28).",
    )
    parser.add_argument(
        "--total-genes",
        type=int,
        default=None,
        help="Total genes per individual assumed for the projection in "
        "--benchmark mode (default: the full gencode gene count).",
    )
    return parser.parse_args()


def run_benchmark(vcf_processor, cre_extractor, fasta_path, candidate_gene_ids, args) -> None:
    """Time the optimized batched path and project per-individual runtime."""
    ds = vcf_processor.model_config.dataset
    gene_extractor = ExtractSeqFromBed(
        neighbour_hood=ds.gene_downstream_neighbour_hood,
        ref_fasta=fasta_path,
        upstream_neighbour_hood=ds.gene_upstream_neighbour_hood,
    )

    per_gene_seconds: List[float] = []
    cre_seconds_all: List[float] = []
    gene_seconds_all: List[float] = []
    region_counts: List[int] = []
    genes_done = 0

    for gene_id in candidate_gene_ids:
        if genes_done >= args.num_genes:
            break
        bed_regions = build_bed_regions(vcf_processor, gene_id)
        if bed_regions is None or len(bed_regions) == 0:
            log.info("Skipping %s (no CRE map / empty)", gene_id)
            continue
        gene_info = get_gene_info(vcf_processor, gene_id)

        n, cre_s, gene_s = benchmark_gene(
            cre_extractor, gene_extractor, args.vcf, args.variant_type, bed_regions, gene_info
        )
        per_gene_seconds.append(cre_s + gene_s)
        cre_seconds_all.append(cre_s)
        gene_seconds_all.append(gene_s)
        region_counts.append(n)
        genes_done += 1
        log.info(
            "Gene %s: %d CREs | CRE batch %.2fs | gene window %.2fs | total %.2fs",
            gene_id, n, cre_s, gene_s, cre_s + gene_s,
        )

    if genes_done == 0:
        log.error("No genes were benchmarked; check gene selection / CRE manifest.")
        sys.exit(2)

    total_genes = args.total_genes or len(vcf_processor.get_genes())
    mean_gene = statistics.mean(per_gene_seconds)
    median_gene = statistics.median(per_gene_seconds)

    log.info("--- Benchmark summary (batched path only, %s) ---",
             "personalized" if args.vcf else "reference")
    log.info("Genes timed: %d | mean CREs/gene: %.0f",
             genes_done, statistics.mean(region_counts))
    log.info("Per gene: mean %.2fs (CRE %.2fs + gene %.2fs), median %.2fs",
             mean_gene, statistics.mean(cre_seconds_all),
             statistics.mean(gene_seconds_all), median_gene)

    serial_h = mean_gene * total_genes / 3600.0
    parallel_h = serial_h / max(1, args.workers)
    log.info(
        "Projection for %d genes/individual: ~%.1f h single-threaded, "
        "~%.1f h with %d workers (CPU side only; GPU overlaps).",
        total_genes, serial_h, parallel_h, args.workers,
    )
    log.info(
        "Note: this ignores GPU/BPE/collate time, which overlap with data "
        "loading; treat it as the data-loading ceiling per individual."
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    vcf_processor = VCFProcessor(model_class=args.model_class)
    fasta_path = vcf_processor.vcf_loader_config.fasta_path
    cre_neighbour_hood = vcf_processor.model_config.dataset.cre_neighbour_hood

    extractor = ExtractSeqFromBed(neighbour_hood=cre_neighbour_hood, ref_fasta=fasta_path)

    candidate_gene_ids = select_gene_ids(vcf_processor, args.num_genes, args.genes)

    if args.benchmark:
        run_benchmark(vcf_processor, extractor, fasta_path, candidate_gene_ids, args)
        return

    total_regions = 0
    total_mismatched = 0
    genes_done = 0

    for gene_id in candidate_gene_ids:
        if not args.genes and genes_done >= args.num_genes:
            break
        bed_regions = build_bed_regions(vcf_processor, gene_id)
        if bed_regions is None or len(bed_regions) == 0:
            log.info("Skipping %s (no CRE map / empty)", gene_id)
            continue

        n, m = compare_gene(extractor, args.vcf, args.variant_type, bed_regions)
        total_regions += n
        total_mismatched += m
        genes_done += 1
        log.info(
            "Gene %s: %d CRE regions compared, %d mismatched", gene_id, n, m
        )

    log.info(
        "Validated %d genes, %d CRE regions, %d mismatches (%s path)",
        genes_done,
        total_regions,
        total_mismatched,
        "personalized" if args.vcf else "reference",
    )

    if genes_done == 0:
        log.error("No genes were validated; check gene selection / CRE manifest.")
        sys.exit(2)
    if total_mismatched != 0:
        log.error("FAILED: batched extraction does not match per-region output.")
        sys.exit(1)
    log.info("PASSED: batched extraction is identical to per-region output.")


if __name__ == "__main__":
    main()
