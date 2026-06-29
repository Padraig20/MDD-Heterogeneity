from __future__ import annotations

import argparse
import logging
import sys
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
    return parser.parse_args()


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
