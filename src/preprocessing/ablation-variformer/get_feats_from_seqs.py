from __future__ import annotations
import argparse
import pickle
import sys
import csv
import logging
import random
import types
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from enformer_pytorch import Enformer
from numpy.lib.format import open_memmap
from tqdm import tqdm

import json
import os

import pysam

sys.path.append(str(Path(__file__).resolve().parent))
from filter_genes import load_gene_subset

# we have very long sequences...
csv.field_size_limit(sys.maxsize)

"""
get_feats_from_seqs.py

Script that takes input from the human reference genome sequences we extracted
earlier (around the TSS) and then extracts features from these sequences using
some sort of gLM model. Here, we use e.g. Enformer via Hugging Face

Optionally personalize each gene window using genotype VCFs, selecting
individuals from the VCF headers and replacing SNPs in the sequence window.

Note: only biallelic SNPs are substituted. For the 'enformer' backbone a
heterozygous genotype is represented by ALT if any ALT allele is present; the
'variformer' backbone instead uses the diploid 0.5/0.5 encoding it was
fine-tuned with (see `apply_dosage_edits`).

Backbones
---------
enformer          Pre-trained Enformer; features are the 5313 human tracks.
variantformer-*   VariantFormer (this repo's submodule), tissue-conditioned.
variformer        Enformer fine-tuned on paired GTEx whole-blood WGS + RNA-seq
                  by Drusinsky et al. (https://github.com/shirondru/enformer_fine_tuning),
                  loaded from a Lightning checkpoint under --variformer-weights.
                  It takes a 49,152 bp TSS-centered window (a center crop of the
                  196,608 bp windows in the input CSV) and yields 3072-dim trunk
                  embeddings plus its own whole-blood expression prediction. The
                  checkpoints were fine-tuned on ~300 genes only, so pass the
                  matching --genes list.

Note: 'variformer' and 'variantformer' are two different models with unhappily
similar names; the former is the fine-tuned Enformer above, the latter is the
tissue-conditioned model in `variantformer/`.

Note: for the 'variantformer' backbone, per-individual VCFs must use "chr"-prefixed
chromosome names (e.g. "chr1"), matching VariantFormer's reference genome/gencode.
Cohorts with bare chromosome names (e.g. OneK1K's "1".."22") must be regenerated
first via `rename_vcf_chr_prefix.sh`; otherwise `bcftools consensus` silently
applies zero variants for every individual (see `vcf_uses_chr_prefix`).

https://huggingface.co/EleutherAI/enformer-official-rough
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mapping = {
    "A": 0, "a": 0,
    "C": 1, "c": 1,
    "G": 2, "g": 2,
    "T": 3, "t": 3,
    "N": 4, "n": 4,
}  # for ACGTN, in that order (-1 for padding)

# Variformer was fine-tuned on 49,152 bp TSS-centered windows and its trunk
# runs with Enformer's cropping disabled, so it emits 49152 / 128 = 384 bins of
# 2 * 1536 = 3072 dimensions each.
VARIFORMER_SEQ_LEN = 49_152

# One-hot rows indexed by the ACGTN codes in `mapping`. N (code 4) becomes an
# all-zero row, matching kipoiseq's `one_hot_dna` as used during fine-tuning.
ONE_HOT_BY_CODE = np.zeros((5, 4), dtype=np.float32)
ONE_HOT_BY_CODE[[0, 1, 2, 3], [0, 1, 2, 3]] = 1.0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare target variables for downstream analysis."
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=None,
        help="Path to input file (*.csv). Required unless --self-test is given."
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help=(
            "Path to write output file (just name, *.npy suffix will be added). "
            "Required unless --self-test is given."
        )
    )
    parser.add_argument(
        "-m", "--model-name",
        type=str,
        default="enformer",
        choices=["enformer", "variantformer-pcg", "variantformer-ag", "variformer"],
        help=(
            "Name of the model to use for feature extraction. 'variformer' is "
            "the whole-blood fine-tuned Enformer from Drusinsky et al., not to "
            "be confused with the tissue-conditioned 'variantformer-*'."
        )
    )
    parser.add_argument(
        "-g", "--genes",
        type=Path,
        default=None,
        help=(
            "Optional gene subset file (e.g. 300_train_genes.tsv); every "
            "ENSEMBL ID found in it is kept and all other rows of the input CSV "
            "are skipped. Required in practice for the 'variformer' backbone, "
            "whose checkpoints only saw those ~300 genes during fine-tuning."
        )
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=1,
        help="Batch size for feature extraction."
    )
    parser.add_argument(
        "-w", "--window-size",
        type=int,
        default=4,
        help="Window size for feature extraction, i.e. number of bins."
    )
    parser.add_argument(
        "--vcf-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory of VCF files used for personalization. The "
            "expected layout depends on the backbone: the 'enformer' backbone "
            "reads chromosome-level multi-sample VCFs (chr1.vcf.gz ... "
            "chr22.vcf.gz), while the 'variantformer' backbone reads one "
            "single-sample, whole-genome VCF per individual named "
            "'<individual>.vcf.gz' (bgzipped + tabix-indexed, e.g. produced by "
            "`bcftools +split`). For the 'variantformer' backbone, the VCFs' "
            "CHROM column must use 'chr'-prefixed contigs (e.g. 'chr1'), matching "
            "VariantFormer's reference genome/gencode; otherwise `bcftools "
            "consensus` silently applies zero variants for every individual "
            "(identical, reference-only embeddings for the whole cohort). Cohorts "
            "with bare chromosome names (e.g. OneK1K's '1'..'22') must first be "
            "regenerated via `src/preprocessing/rename_vcf_chr_prefix.sh`. Omit "
            "--vcf-dir if personalization not needed."
        )
    )
    parser.add_argument(
        "-t", "--tissue",
        type=str,
        default=None,
        help=(
            "Tissue name to condition the embeddings on (required for the "
            "'variantformer' backbone). Must be a tissue present in VariantFormer's "
            "tissue vocabulary, e.g. 'brain - cortex' or 'liver'."
        )
    )
    parser.add_argument(
        "--variantformer-dir",
        type=Path,
        default=None,
        help=(
            "Path to the VariantFormer package directory (the folder containing "
            "the 'processors', 'datasets' and 'configs' subpackages). Defaults to "
            "'<repo-root>/variantformer'."
        )
    )
    parser.add_argument(
        "--variformer-weights",
        type=Path,
        default=Path(__file__).resolve().parent / "model_weights",
        help=(
            "Lightning checkpoint (*.ckpt) for the 'variformer' backbone, or a "
            "directory of them named 'Fold-<n>-*.ckpt' (in which case "
            "--variformer-fold picks one). Defaults to this script's "
            "'model_weights' directory."
        )
    )
    parser.add_argument(
        "--variformer-fold",
        type=int,
        default=0,
        help=(
            "Which fold replicate to use when --variformer-weights is a "
            "directory. The folds share the same ~300 fine-tuning genes and "
            "differ only in their donor splits."
        )
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "Load the 'variformer' checkpoint, run the reference "
            "'dummy_input.pt' through it, compare the center-bin prediction "
            "against '300gene_preds.pt', and exit without extracting features."
        )
    )
    parser.add_argument(
        "--num-individuals",
        type=str,
        default="0",
        help=(
            "Individuals to use from the VCF files when --vcf-dir is provided. "
            "Use an integer to randomly sample that many individuals, 'all' to "
            "use every individual, or 'K/N' to process split K of N contiguous "
            "splits (1-based). Individuals are ordered by VCF header order for "
            "the 'enformer' backbone and by per-individual VCF filename for the "
            "'variantformer' backbone."
        )
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed for integer-based sampling from the available individuals.",
    )
    parser.add_argument(
        "--individual-split",
        type=str,
        default=None,
        help=(
            "Optional K/N contiguous split applied after --num-individuals "
            "selection (1-based). For example, '--num-individuals 1000 "
            "--sample-seed 42 --individual-split 1/20' processes the first 50 "
            "individuals from the seeded random sample of 1000."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=(
            "Override the number of DataLoader workers for the VariantFormer "
            "backbone. Sequence extraction is data-loading bound (one indexed "
            "`bcftools consensus` call per CRE/gene window), so more workers "
            "parallelize this across genes. Defaults to the value in "
            "variantformer/configs/vcfloader.yaml (currently 4). A good starting "
            "point is the number of physical CPU cores."
        ),
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=None,
        help=(
            "Override the DataLoader prefetch_factor (batches preloaded per "
            "worker) for the VariantFormer backbone. Only used when "
            "--num-workers > 0. Defaults to the vcfloader.yaml value (currently 4)."
        ),
    )
    parser.add_argument(
        "--maf-threshold",
        type=float,
        default=None,
        help=(
            "Minimum minor allele frequency for a SNP to be spliced into the "
            "input sequence (e.g. 0.05 for MAF >= 5%%). MAF is computed across "
            "the selected individuals. Applies to the personalized 'enformer' "
            "and 'variformer' backbones, which build their own sequences; "
            "VariantFormer handles variants internally. Defaults to no MAF "
            "filtering."
        ),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity."
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=20000,
        help="Flush and checkpoint every N batches."
    )
    args = parser.parse_args()

    if not args.self_test:
        missing = [flag for flag, value in (("--input", args.input), ("--output", args.output)) if value is None]
        if missing:
            parser.error(f"the following arguments are required: {', '.join(missing)}")
    return args


def setup_logging(verbosity: int) -> None:
    """Configure basic logging based on verbosity level."""
    level = logging.INFO
    if verbosity >= 1:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def save_checkpoint(checkpoint_path: Path, idx: int, ensids: np.ndarray, chroms: np.ndarray, tss: np.ndarray, sample_ids_arr: np.ndarray, output_path: Path):
    np.save(output_path.with_suffix(".ensids.npy"), ensids)
    np.save(output_path.with_suffix(".chroms.npy"), chroms)
    np.save(output_path.with_suffix(".tss.npy"), tss)
    if sample_ids_arr is not None:
        np.save(output_path.with_suffix(".sample_ids.npy"), sample_ids_arr)

    # progress file for tracking checkpointing progress
    tmp_ckpt = checkpoint_path.with_suffix(".tmp")
    with tmp_ckpt.open("w", encoding="utf-8") as f:
        json.dump({"idx": idx}, f)
        f.flush()
        os.fsync(f.fileno())
    tmp_ckpt.replace(checkpoint_path)


def keep_row(row: Dict[str, Any], keep_ensids: Optional[Set[str]]) -> bool:
    """Autosomal rows only, optionally restricted to a gene subset."""
    if not str(row["chrom"]).isdigit():
        return False
    if keep_ensids is None:
        return True
    return str(row["ensid"]).split(".", 1)[0] in keep_ensids


def count_autosomal_rows_csv(input_path: Path, keep_ensids: Optional[Set[str]] = None) -> int:
    with input_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return sum(1 for row in reader if keep_row(row, keep_ensids))


def get_total_output_rows(input_path: Path, personalized: bool, num_individuals: int, keep_ensids: Optional[Set[str]] = None) -> int:
    num_rows = count_autosomal_rows_csv(input_path, keep_ensids)
    if not personalized:
        return num_rows
    return num_rows * num_individuals


def iter_autosomal_rows_csv(input_path: Path, skip_rows: int = 0, keep_ensids: Optional[Set[str]] = None) -> Iterator[Dict[str, Any]]:
    with input_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        kept   = 0
        for row in reader:
            if not keep_row(row, keep_ensids):
                continue
            if kept < skip_rows:
                kept += 1
                continue
            yield row 


def batched_rows(rows: Iterator[Dict[str, Any]], batch_size: int) -> Iterator[List[Dict[str, Any]]]:
    batch: List[Dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def dna_seq_to_array(seq: str) -> np.ndarray:
    """Convert a DNA sequence string to an integer-encoded NumPy array (1D)."""
    return np.fromiter((mapping[nuc] for nuc in seq), dtype=np.uint8)


def dna_seq_to_tensor(seq: np.ndarray) -> torch.Tensor:
    """Convert an integer-encoded NumPy array to a tensor (1D)."""
    return torch.as_tensor(seq, dtype=torch.long, device=device)


def codes_to_one_hot(codes: np.ndarray) -> np.ndarray:
    """One-hot encode ACGTN codes as float32 (L,) -> (L, 4); N becomes all zeros."""
    return ONE_HOT_BY_CODE[codes]


def center_crop(seq: np.ndarray, length: int) -> np.ndarray:
    """Take the central `length` elements of a sequence.

    The input CSV holds 196,608 bp windows centered on the TSS, while Variformer
    was fine-tuned on 49,152 bp. Cropping the center keeps the TSS in the middle,
    which is exactly how the fine-tuning dataset shortened its windows.
    """
    if len(seq) < length:
        raise ValueError(
            f"Sequence of length {len(seq)} is shorter than the required "
            f"{length} bp window; regenerate the input CSV with a larger window."
        )
    start = (len(seq) - length) // 2
    return seq[start: start + length]


def apply_dosage_edits(one_hot: np.ndarray, edits: List[Tuple[int, int, int, float]]) -> None:
    """Splice biallelic SNPs into a one-hot sequence in place, encoding dosage.

    Variformer was fine-tuned on unphased diploid genomes encoded as the average
    of the two haplotypes' one-hot sequences, so a heterozygous site is 0.5/0.5
    across its two alleles and a homozygous ALT site is a clean one-hot ALT. Hard
    calling the ALT allele instead (as the Enformer path does) would present the
    model with an encoding it never saw during fine-tuning.
    """
    for rel_pos, ref_code, alt_code, dosage in edits:
        row = one_hot[rel_pos]
        row[:] = 0.0
        if ref_code < 4:
            row[ref_code] = 1.0 - dosage
        if alt_code < 4:
            row[alt_code] += dosage


def open_variant_files(vcf_dir: Path) -> Dict[str, "pysam.VariantFile"]:
    chrom_to_vcf: Dict[str, pysam.VariantFile] = {}
    for chrom in range(1, 23):
        path = vcf_dir / f"chr{chrom}.vcf.gz"
        logging.debug("Opening VCF for chromosome %s: %s", chrom, path)
        chrom_to_vcf[f"chr{chrom}"] = pysam.VariantFile(path)
    return chrom_to_vcf


def close_variant_files(chrom_to_vcf: Dict[str, "pysam.VariantFile"]) -> None:
    for vf in chrom_to_vcf.values():
        try:
            vf.close()
        except Exception:
            pass


def split_contiguous(items: List[str], split_idx: int, total_splits: int) -> List[str]:
    quotient, remainder = divmod(len(items), total_splits)
    start = (split_idx - 1) * quotient + min(split_idx - 1, remainder)
    stop = start + quotient + (1 if split_idx <= remainder else 0)
    return items[start:stop]


def sample_individual_ids(
    chrom_to_vcf: Dict[str, "pysam.VariantFile"],
    selection: str,
    seed: int,
    individual_split: Optional[str] = None,
) -> List[str]:
    first_vcf = next(iter(chrom_to_vcf.values()))
    samples   = list(first_vcf.header.samples)
    return select_individual_ids(samples, selection, seed, individual_split)


def get_vcf_samples(vcf_path: Path) -> List[str]:
    """Read the sample/individual IDs from a single VCF header."""
    vf = pysam.VariantFile(str(vcf_path))
    try:
        return list(vf.header.samples)
    finally:
        vf.close()


def vcf_uses_chr_prefix(vcf_path: Path) -> bool:
    """Peek the first record's CHROM to detect 'chr1' vs '1' style contig naming.

    VariantFormer's reference genome/gencode coordinates always use "chr"-prefixed
    contigs (e.g. "chr1"), but some per-individual VCFs (e.g. OneK1K, split from
    cohort VCFs with bare "1".."22" contigs) don't. Without a matching contig name,
    `bcftools consensus` silently applies zero variants (exit code 0, no error) when
    VariantFormer pipes it a "chrN:start-end" FASTA region -- personalization becomes
    a silent no-op and every individual ends up with the same reference-only
    embedding. See `rename_vcf_chr_prefix.sh` to fix a mismatched --vcf-dir.
    """
    vf = pysam.VariantFile(str(vcf_path))
    try:
        for rec in vf:
            return str(rec.chrom).startswith("chr")
    finally:
        vf.close()
    return True  # empty VCF; nothing to check


def discover_individual_vcfs(vcf_dir: Path) -> Dict[str, Path]:
    """Map individual ID -> per-individual VCF path for the VariantFormer backbone.

    Expects ``vcf_dir`` to contain one single-sample, whole-genome VCF per
    individual (e.g. produced by ``bcftools +split``), bgzipped and tabix-indexed
    as ``<individual>.vcf(.bgz|.gz)``. The individual ID is the filename with the
    VCF suffix stripped and is assumed to match the sample name inside the file.

    Each VCF's CHROM column must use "chr"-prefixed contigs (e.g. "chr1"), matching
    VariantFormer's reference genome/gencode; see ``vcf_uses_chr_prefix`` and
    ``rename_vcf_chr_prefix.sh`` for cohorts (e.g. OneK1K) that use bare chromosome
    names instead.
    """
    suffixes = (".vcf.gz", ".vcf.bgz")
    mapping: Dict[str, Path] = {}
    for path in sorted(vcf_dir.iterdir()):
        if not path.is_file():
            continue
        name = path.name
        for suffix in suffixes:
            if name.endswith(suffix):
                mapping[name[: -len(suffix)]] = path
                break
    if not mapping:
        raise FileNotFoundError(
            f"No per-individual VCFs (*.vcf.gz / *.vcf.bgz) found in {vcf_dir}. "
            "The VariantFormer backbone expects one single-sample VCF per "
            "individual (e.g. from `bcftools +split`)."
        )
    return mapping


def apply_individual_split(samples: List[str], selection: str) -> List[str]:
    """Apply a 1-based ``K/N`` contiguous split to already-selected samples."""
    split_parts = selection.strip().lower().split("/")
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
        split_idx,
        total_splits,
        len(chosen),
        len(samples),
    )
    return chosen


def select_individual_ids(
    samples: List[str],
    selection: str,
    seed: int,
    individual_split: Optional[str] = None,
) -> List[str]:
    """Select individuals from a list of VCF sample IDs.

    Supports 'all', an integer count (randomly sampled), or a 'K/N' contiguous
    split of the header order (1-based). If ``individual_split`` is provided,
    that contiguous K/N split is applied after this initial selection.
    """
    selection = selection.strip().lower()

    if selection == "all":
        chosen = samples
        logging.info("Selected all %d individuals from VCF headers.", len(chosen))
    elif "/" in selection:
        split_parts = selection.split("/")
        if len(split_parts) != 2:
            raise ValueError("--num-individuals split syntax must be K/N, e.g. 2/4.")
        try:
            split_idx = int(split_parts[0])
            total_splits = int(split_parts[1])
        except ValueError as exc:
            raise ValueError("--num-individuals split syntax must use integer values, e.g. 2/4.") from exc
        if total_splits <= 0:
            raise ValueError("--num-individuals split total N must be greater than 0.")
        if split_idx < 1 or split_idx > total_splits:
            raise ValueError("--num-individuals split K must satisfy 1 <= K <= N.")

        chosen = split_contiguous(samples, split_idx, total_splits)
        logging.info(
            "Selected split %d/%d with %d of %d VCF individuals.",
            split_idx,
            total_splits,
            len(chosen),
            len(samples),
        )
    else:
        try:
            num_individuals = int(selection)
        except ValueError as exc:
            raise ValueError("--num-individuals must be an integer, 'all', or K/N, e.g. 2/4.") from exc
        if num_individuals <= 0:
            raise ValueError("--num-individuals must select at least one individual when --vcf-dir is provided.")
        if num_individuals > len(samples):
            raise ValueError(
                f"Requested {num_individuals} individuals, but VCF header only contains {len(samples)} samples."
            )

        rng    = random.Random(seed)
        chosen = rng.sample(samples, num_individuals)
        logging.info(
            "Sampled %d individuals from VCF headers using seed %d.",
            len(chosen),
            seed,
        )

    if not chosen:
        raise ValueError("--num-individuals selected zero individuals.")
    if individual_split is not None:
        chosen = apply_individual_split(chosen, individual_split)
    logging.debug("Selected individuals: %s", chosen[:5])
    return chosen


def collect_window_genotype_edits(vcf: "pysam.VariantFile", chrom: str, start0: int, end0: int, sampled_ids: List[str], ref_seq: np.ndarray, maf_threshold: Optional[float] = None) -> List[List[Tuple[int, int, int, float]]]:
    """Per individual, the biallelic SNPs to splice into the window.

    Each edit is ``(rel_pos, ref_code, alt_code, alt_dosage)``, where `ref_code`
    is the base actually present in the window (which is what the non-ALT
    haplotype keeps) and `alt_dosage` is the fraction of called alleles that are
    ALT, i.e. 0.5 for a heterozygote and 1.0 for a homozygous ALT. Sites with no
    ALT allele produce no edit at all.
    """
    edits_per_individual = [[] for _ in sampled_ids]
    fetch_chrom          = chrom[3:]

    for rec in vcf.fetch(fetch_chrom, start0, end0):
        ref  = rec.ref
        alts = rec.alts

        if len(ref) != 1 or alts is None or len(alts) != 1 or len(alts[0]) != 1:
            continue

        pos0    = rec.pos - 1
        rel_pos = pos0 - start0
        if rel_pos < 0 or rel_pos >= len(ref_seq):
            continue

        seq_base = int(ref_seq[rel_pos])

        ref = ref.upper()
        alt = alts[0].upper()

        ref_code = mapping[ref]
        alt_code = mapping[alt]

        if seq_base != ref_code:
            logging.debug(
                "Reference mismatch at %s:%d for ENSID window base. Sequence has %s, VCF REF has %s",
                chrom, rec.pos, seq_base, ref
            )

        rec_samples = rec.samples
        gts = [rec_samples[sid]["GT"] for sid in sampled_ids]

        # Optionally drop low-frequency SNPs. MAF is computed across the selected
        # cohort (sampled_ids) from the called alleles of this record.
        if maf_threshold is not None and not _passes_maf(gts, maf_threshold):
            continue

        for i, gt in enumerate(gts):
            if not gt:
                continue
            called = [a for a in gt if a is not None]
            if not called:
                continue
            dosage = sum(1 for a in called if a > 0) / len(called)
            if dosage > 0:
                edits_per_individual[i].append((rel_pos, seq_base, alt_code, dosage))

    return edits_per_individual


def _passes_maf(gts: List[Any], maf_threshold: float) -> bool:
    """Return True if the minor allele frequency across the given genotypes is >= maf_threshold.

    `gts` is a list of per-individual GT tuples (e.g. (0, 1)); None alleles are
    ignored. Records with no called alleles are treated as failing the filter.
    """
    alt_alleles   = 0
    total_alleles = 0
    for gt in gts:
        if not gt:
            continue
        for allele in gt:
            if allele is None:
                continue
            total_alleles += 1
            if allele > 0:
                alt_alleles += 1

    if total_alleles == 0:
        return False

    p   = alt_alleles / total_alleles
    maf = min(p, 1.0 - p)
    return maf >= maf_threshold


def apply_variants_to_sequence(ref_seq: str, variants, individual_idx: int) -> str:
    """
    - missing GT -> keep reference base
    - if any non-zero allele is present -> use ALT
    - otherwise keep reference
    """
    seq_chars = list(ref_seq)

    for rel_pos, ref, alt, gts in variants:
        gt = gts[individual_idx]
        if gt is None:
            continue

        # Example GT values: (0, 0), (0, 1), (1, 1), (None, None)
        called_alleles = [a for a in gt if a is not None]
        if not called_alleles:
            continue

        if any(a > 0 for a in called_alleles):
            seq_chars[rel_pos] = alt

    return "".join(seq_chars)


def iter_personalized_rows(input_path: Path, chrom_to_vcf: Dict[str, "pysam.VariantFile"], sampled_ids: List[str], skip_input_rows: int = 0, skip_within_expanded_row: int = 0, maf_threshold: Optional[float] = None, keep_ensids: Optional[Set[str]] = None, crop_length: Optional[int] = None, diploid_one_hot: bool = False) -> Iterator[Dict[str, Any]]:
    """Expand each input row into gene x sampled-individual rows with SNP-personalized sequences.

    `crop_length` narrows each window to its central `crop_length` bases before
    any VCF lookup, so shorter-context backbones neither read nor splice
    variants they would immediately crop away. With `diploid_one_hot` the yielded
    `sequence_array` is a float32 (L, 4) one-hot with heterozygotes encoded as
    0.5/0.5; otherwise it is the usual (L,) array of ACGTN codes with ALT hard
    called.
    """
    for row_idx, row in enumerate(iter_autosomal_rows_csv(input_path, skip_rows=skip_input_rows, keep_ensids=keep_ensids)):
        ref_seq = dna_seq_to_array(row["sequence"].upper())
        chrom   = row["chrom"]
        start0  = int(row["actual_start"])
        end0    = int(row["actual_end"])

        if crop_length is not None:
            offset  = (len(ref_seq) - crop_length) // 2
            ref_seq = center_crop(ref_seq, crop_length)
            start0 += offset
            end0    = start0 + crop_length

        chrom = f"chr{chrom}" if not chrom.startswith("chr") else chrom

        edits_per_individual = collect_window_genotype_edits(
            vcf=chrom_to_vcf[chrom],
            chrom=chrom,
            start0=start0,
            end0=end0,
            sampled_ids=sampled_ids,
            ref_seq=ref_seq,
            maf_threshold=maf_threshold,
        )

        ref_one_hot = codes_to_one_hot(ref_seq) if diploid_one_hot else None
        start_i     = skip_within_expanded_row if row_idx == 0 else 0

        for i in range(start_i, len(sampled_ids)):
            sample_id = sampled_ids[i]

            if diploid_one_hot:
                seq_arr = ref_one_hot.copy()
                apply_dosage_edits(seq_arr, edits_per_individual[i])
            else:
                seq_arr = ref_seq.copy()
                for rel_pos, _, alt_code, _ in edits_per_individual[i]:
                    seq_arr[rel_pos] = alt_code

            yield {
                "sequence_array": seq_arr,
                "ensid": row["ensid"],
                "chrom": row["chrom"],
                "tss": row["tss"],
                "sample_id": sample_id,
            }


def _row_tss(row: Dict[str, Any]) -> Any:
    """Best-effort extraction of a TSS-like value from a CSV row."""
    for key in ("tss", "tss_start", "actual_start"):
        value = row.get(key)
        if value not in (None, ""):
            return value
    return ""


class DroppedCheckpointObject:
    """Stand-in for a pickled object whose defining module we don't have."""

    def __setstate__(self, state: Any) -> None:
        pass


class TolerantUnpickler(pickle.Unpickler):
    """Unpickler that substitutes a placeholder for unimportable classes.

    Variformer's checkpoints are Lightning checkpoints that pickle their
    training `hyper_parameters`, including a `GTExDataset` instance from the
    original fine-tuning repo. We only want `state_dict`, and vendoring that
    repo (plus its data layout) just to unpickle a dataset object we discard
    would be far worse than dropping it here.
    """

    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            logging.debug("Dropping unavailable checkpoint class %s.%s", module, name)
            return type(name, (DroppedCheckpointObject,), {})


TOLERANT_PICKLE = types.ModuleType("tolerant_pickle")
TOLERANT_PICKLE.__dict__.update(pickle.__dict__)
TOLERANT_PICKLE.Unpickler = TolerantUnpickler


def resolve_variformer_checkpoint(weights: Path, fold: int) -> Path:
    """Resolve --variformer-weights to a single checkpoint file."""
    if weights.is_file():
        return weights
    if not weights.is_dir():
        raise FileNotFoundError(f"--variformer-weights path does not exist: {weights}")

    matches = sorted(weights.glob(f"Fold-{fold}-*.ckpt"))
    if not matches:
        available = sorted(p.name for p in weights.glob("*.ckpt"))
        raise FileNotFoundError(
            f"No 'Fold-{fold}-*.ckpt' in {weights}. Available checkpoints: {available}"
        )
    if len(matches) > 1:
        raise ValueError(f"Several checkpoints match fold {fold} in {weights}: {matches}")
    return matches[0]


def load_variformer(weights_path: Path) -> Tuple[nn.Module, List[str]]:
    """Rebuild the fine-tuned Enformer from a Lightning checkpoint.

    The original repo instantiates the wrapper by downloading pre-trained
    Enformer weights and then overwriting them from the checkpoint. We skip that
    round trip and build the same architecture directly from `enformer_pytorch`,
    which keeps this script offline and free of a dependency on the fine-tuning
    repo. `--self-test` verifies the reconstruction against the reference
    predictions shipped alongside the checkpoints.
    """
    from enformer_pytorch.finetune import HeadAdapterWrapper  # lazy import

    logging.info("Loading Variformer checkpoint %s", weights_path)
    ckpt = torch.load(
        weights_path,
        map_location="cpu",
        weights_only=False,
        pickle_module=TOLERANT_PICKLE,
        mmap=True,
    )

    tissues = list(ckpt.get("hyper_parameters", {}).get("tissues_to_train", []))
    if not tissues:
        raise ValueError(f"{weights_path} does not record any 'tissues_to_train'.")
    logging.info("Checkpoint was fine-tuned on tissue(s): %s", tissues)

    prefix = "model."
    state  = {k[len(prefix):]: v for k, v in ckpt["state_dict"].items() if k.startswith(prefix)}
    if not state:
        raise ValueError(f"{weights_path} holds no 'model.*' weights; is it a Variformer checkpoint?")

    # target_length = -1 disables Enformer's output cropping so the trunk can
    # take the shorter 49,152 bp windows. use_tf_gamma must stay off: the
    # TensorFlow positional-embedding gammas are only tabulated for the 1536-bin
    # sequences of a full 196,608 bp input.
    enformer = Enformer.from_hparams(
        dim=1536,
        depth=11,
        heads=8,
        output_heads=dict(human=5313, mouse=1643),
        target_length=-1,
        use_tf_gamma=False,
    )
    model = HeadAdapterWrapper(
        enformer=enformer,
        num_tracks=len(tissues),
        post_transformer_embed=False,
        output_activation=nn.Identity(),
    )
    model.load_state_dict(state, strict=True)
    del ckpt, state

    return model.to(device).eval(), tissues


def variformer_forward(model: nn.Module, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run one batch of one-hot sequences, returning (embeddings, predictions).

    Splitting the wrapper's forward pass lets us keep the 3072-dim trunk
    embeddings, which the wrapper itself discards after projecting them onto its
    single fine-tuned expression track.
    """
    with torch.no_grad():
        embeddings = model.enformer(batch, return_only_embeddings=True)  # (B, bins, 3072)
        predictions = model.to_tracks(embeddings)                        # (B, bins, tracks)
    return embeddings, predictions


def average_central_bins(binned: np.ndarray, window_size: int) -> np.ndarray:
    """Average the `window_size` bins centered on the sequence midpoint."""
    if window_size < 1:
        raise ValueError("--window-size must be at least 1.")
    center = binned.shape[1] // 2
    start  = center - window_size // 2
    if start < 0 or start + window_size > binned.shape[1]:
        raise ValueError(
            f"--window-size {window_size} does not fit into {binned.shape[1]} output bins."
        )
    return binned[:, start: start + window_size, :].mean(axis=1)


def variformer_self_test(weights_path: Path, tolerance: float = 1e-3) -> None:
    """Check the rebuilt model against the reference predictions in the weights dir.

    `dummy_input.pt` / `300gene_preds.pt` ship with the checkpoints and pin down
    the expected center-bin prediction, so this catches an architecture or
    state-dict mismatch immediately instead of after hours of extraction. The
    reference values were produced on a GPU, so small floating-point drift is
    expected and only a gross mismatch fails.
    """
    weights_dir   = weights_path.parent
    dummy_path    = weights_dir / "dummy_input.pt"
    expected_path = weights_dir / "300gene_preds.pt"
    for path in (dummy_path, expected_path):
        if not path.exists():
            raise FileNotFoundError(f"--self-test needs {path}, which ships with the checkpoints.")

    expected_by_ckpt = torch.load(expected_path, map_location="cpu", weights_only=False)
    if weights_path.name not in expected_by_ckpt:
        raise KeyError(
            f"{expected_path} has no reference prediction for {weights_path.name}; "
            f"it covers {sorted(expected_by_ckpt)}."
        )
    expected = float(expected_by_ckpt[weights_path.name])

    model, _ = load_variformer(weights_path)
    dummy = torch.load(dummy_path, map_location="cpu", weights_only=False).to(device)
    logging.info("Self-test input: %s", tuple(dummy.shape))

    embeddings, predictions = variformer_forward(model, dummy)
    actual = float(predictions[0, predictions.shape[1] // 2, 0])

    logging.info("Embeddings: %s", tuple(embeddings.shape))
    logging.info("Center-bin prediction: %.10f (expected %.10f)", actual, expected)
    difference = abs(actual - expected)
    if difference > tolerance:
        raise ValueError(
            f"Self-test failed: center-bin prediction {actual} differs from the "
            f"expected {expected} by {difference}, above the {tolerance} tolerance."
        )
    logging.info("Self-test passed (absolute difference %.3g).", difference)


def extract_variformer_features(
    data_path: Path,
    output_path: Path,
    weights_path: Path,
    batch_size: int,
    window_size: int,
    vcf_dir: Optional[Path] = None,
    num_individuals: str = "0",
    sample_seed: int = 42,
    individual_split: Optional[str] = None,
    maf_threshold: Optional[float] = None,
    keep_ensids: Optional[Set[str]] = None,
    checkpoint_every: int = 1000,
) -> None:
    """Extract Variformer trunk embeddings (and its own predictions) per sequence.

    Alongside the usual `.features.npy`, this writes a `.preds.npy` holding the
    model's fine-tuned expression prediction, taken from the exact center bin as
    in the original repo's `predict_step`. The embeddings are averaged over
    `--window-size` central bins instead, to match the Enformer feature sets this
    ablation compares against.
    """
    model, tissues = load_variformer(weights_path)
    num_tracks = len(tissues)

    personalized: bool = vcf_dir is not None
    chrom_to_vcf: Dict[str, pysam.VariantFile] = {}
    sampled_ids: List[str] = []

    try:
        if personalized:
            logging.info("Personalized mode enabled using VCF directory: %s", vcf_dir)
            chrom_to_vcf = open_variant_files(vcf_dir)
            sampled_ids  = sample_individual_ids(chrom_to_vcf, num_individuals, sample_seed, individual_split)
        else:
            logging.info("Running in reference-sequence mode.")

        total_rows = get_total_output_rows(
            input_path=data_path,
            personalized=personalized,
            num_individuals=len(sampled_ids) if personalized else 0,
            keep_ensids=keep_ensids,
        )
        if total_rows == 0:
            raise ValueError(
                f"No rows left to process in {data_path}. Autosomal rows are kept "
                "and, when --genes is given, only genes from that subset."
            )
        logging.info("Number of output sequences to process: %d", total_rows)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        feats_mm_path   = output_path.with_suffix(".features.npy")
        preds_mm_path   = output_path.with_suffix(".preds.npy")
        checkpoint_path = output_path.with_suffix(".checkpoint.json")

        emb_dim = model.to_tracks[0].in_features

        if checkpoint_path.exists():
            with checkpoint_path.open("r", encoding="utf-8") as f:
                start_idx = json.load(f).get("idx", 0)
            ensids         = np.load(output_path.with_suffix(".ensids.npy"), allow_pickle=True)
            chroms         = np.load(output_path.with_suffix(".chroms.npy"), allow_pickle=True)
            tss            = np.load(output_path.with_suffix(".tss.npy"), allow_pickle=True)
            sample_ids_arr = np.load(output_path.with_suffix(".sample_ids.npy"), allow_pickle=True) if personalized else None
            feats_mm       = np.load(feats_mm_path, allow_pickle=True, mmap_mode="r+")
            preds_mm       = np.load(preds_mm_path, allow_pickle=True, mmap_mode="r+")
            logging.info("Resuming Variformer extraction from row %d", start_idx)
        else:
            start_idx      = 0
            ensids         = np.empty(total_rows, dtype=object)
            chroms         = np.empty(total_rows, dtype=object)
            tss            = np.empty(total_rows, dtype=object)
            sample_ids_arr = np.empty(total_rows, dtype=object) if personalized else None
            feats_mm       = open_memmap(feats_mm_path, mode="w+", dtype=np.float32, shape=(total_rows, emb_dim))
            preds_mm       = open_memmap(preds_mm_path, mode="w+", dtype=np.float32, shape=(total_rows, num_tracks))

        def save_progress(idx: int) -> None:
            feats_mm.flush()
            preds_mm.flush()
            np.save(output_path.with_suffix(".ensids.npy"), ensids)
            np.save(output_path.with_suffix(".chroms.npy"), chroms)
            np.save(output_path.with_suffix(".tss.npy"), tss)
            if sample_ids_arr is not None:
                np.save(output_path.with_suffix(".sample_ids.npy"), sample_ids_arr)
            tmp_ckpt = checkpoint_path.with_suffix(".tmp")
            with tmp_ckpt.open("w", encoding="utf-8") as f:
                json.dump({"idx": idx}, f)
                f.flush()
                os.fsync(f.fileno())
            tmp_ckpt.replace(checkpoint_path)

        row_iter: Iterator[Dict[str, Any]]
        if personalized:
            row_iter = iter_personalized_rows(
                input_path=data_path,
                chrom_to_vcf=chrom_to_vcf,
                sampled_ids=sampled_ids,
                skip_input_rows=start_idx // len(sampled_ids),
                skip_within_expanded_row=start_idx % len(sampled_ids),
                maf_threshold=maf_threshold,
                keep_ensids=keep_ensids,
                crop_length=VARIFORMER_SEQ_LEN,
                diploid_one_hot=True,
            )
        else:
            row_iter = iter_autosomal_rows_csv(data_path, skip_rows=start_idx, keep_ensids=keep_ensids)

        idx         = start_idx
        bidx        = 0
        num_batches = (total_rows - start_idx + batch_size - 1) // batch_size

        for batch in tqdm(batched_rows(row_iter, batch_size), total=num_batches, desc="Extracting features", unit="batch"):
            if personalized:
                seqs_b       = [b["sequence_array"] for b in batch]
                sample_ids_b = [b["sample_id"] for b in batch]
            else:
                seqs_b = [
                    codes_to_one_hot(center_crop(dna_seq_to_array(b["sequence"].upper()), VARIFORMER_SEQ_LEN))
                    for b in batch
                ]

            batch_tensor = torch.as_tensor(np.stack(seqs_b, axis=0), dtype=torch.float32, device=device)
            embeddings, predictions = variformer_forward(model, batch_tensor)

            features    = average_central_bins(embeddings.cpu().numpy().astype(np.float32), window_size)
            predictions = predictions.cpu().numpy().astype(np.float32)
            predictions = predictions[:, predictions.shape[1] // 2, :]

            actual_bsz = features.shape[0]
            feats_mm[idx: idx + actual_bsz, :] = features
            preds_mm[idx: idx + actual_bsz, :] = predictions
            ensids[idx: idx + actual_bsz]      = [b["ensid"] for b in batch]
            chroms[idx: idx + actual_bsz]      = [b["chrom"] for b in batch]
            tss[idx: idx + actual_bsz]         = [b["tss"] for b in batch]
            if sample_ids_arr is not None:
                sample_ids_arr[idx: idx + actual_bsz] = sample_ids_b

            idx  += actual_bsz
            bidx += 1

            if bidx % checkpoint_every == 0:
                save_progress(idx)
                logging.info("Checkpoint saved at row %d", idx)

        save_progress(idx)
        logging.info("Variformer embeddings successfully extracted to %s", feats_mm_path)
        logging.debug("Sample of extracted features:\n%s", feats_mm[0])

    finally:
        if chrom_to_vcf:
            close_variant_files(chrom_to_vcf)


def extract_variantformer_features(
    data_path: Path,
    output_path: Path,
    tissue: str,
    vf_model_class: str = "v4_pcg",
    vcf_dir: Optional[Path] = None,
    num_individuals: str = "0",
    sample_seed: int = 42,
    num_workers: Optional[int] = None,
    prefetch_factor: Optional[int] = None,
    individual_split: Optional[str] = None,
) -> None:
    if tissue is None:
        raise ValueError("--tissue is required when using the 'variantformer' backbone.")

    from variantformer.processors.vcfprocessor import VCFProcessor # lazy import

    logging.info("Initializing VariantFormer (model class: %s)", vf_model_class)
    vcf_processor = VCFProcessor(model_class=vf_model_class)

    available_tissues = set(vcf_processor.get_tissues())
    if tissue not in available_tissues:
        raise ValueError(
            f"Tissue '{tissue}' is not in VariantFormer's tissue vocabulary. "
            f"Available tissues include e.g.: {sorted(available_tissues)[:10]} ..."
        )

    # variantformer uses versioned ENSIDs, CSV may have unversioned ones...
    genes_df = vcf_processor.get_genes()
    base_to_gene_id: Dict[str, str] = {}
    for gid in genes_df["gene_id"].astype(str):
        base = gid.split(".")[0]
        base_to_gene_id.setdefault(base, gid)

    # all autosomal query genes from input CSV, de-duplicate on gene_id
    gene_meta_by_id: Dict[str, Dict[str, Any]] = {}
    ordered_gene_ids: List[str] = []
    for row in iter_autosomal_rows_csv(data_path):
        ensid = str(row["ensid"])
        gene_id = base_to_gene_id.get(ensid.split(".")[0])
        if gene_id is None:
            logging.debug("ENSID %s not found in VariantFormer gencode; skipping.", ensid)
            continue
        if gene_id in gene_meta_by_id:
            continue
        gene_meta_by_id[gene_id] = {
            "ensid": ensid,
            "chrom": str(row.get("chrom")),
            "tss": _row_tss(row),
        }
        ordered_gene_ids.append(gene_id)

    if not ordered_gene_ids:
        raise ValueError(
            "None of the genes in the input CSV matched VariantFormer's gencode v24."
        )
    logging.info("Matched %d genes to VariantFormer's gencode.", len(ordered_gene_ids))

    num_genes = len(ordered_gene_ids)
    gene_row_offset = {gid: i for i, gid in enumerate(ordered_gene_ids)}

    # One query row per gene, each conditioned on the requested tissue. The
    # VCFDataset filters this against VariantFormer's gencode v24 and tissue
    # vocabulary, and the comma-separated "tissues" string is parsed internally.
    # The same query is reused for every individual.
    query_df = pd.DataFrame(
        {"gene_id": ordered_gene_ids, "tissues": [tissue] * num_genes}
    )

    # Specific individuals to process. In personalized mode --vcf-dir holds one
    # single-sample, whole-genome VCF per individual (named "<individual>.vcf.gz"
    # and tabix-indexed, e.g. produced by `bcftools +split`). Because each file
    # already spans every chromosome, each individual is processed in a single
    # prediction pass over all genes (no per-chromosome loop).
    personalized = vcf_dir is not None
    individual_to_vcf: Dict[str, Path] = {}
    if personalized:
        individual_to_vcf = discover_individual_vcfs(vcf_dir)
        all_individuals = list(individual_to_vcf.keys())
        sample_ids: List[Optional[str]] = list(
            select_individual_ids(
                all_individuals,
                num_individuals,
                sample_seed,
                individual_split,
            )
        )
        logging.info(
            "Personalized mode: %d of %d per-individual VCFs from %s",
            len(sample_ids), len(all_individuals), vcf_dir,
        )

        # Fail fast if the VCFs use bare chromosome names (e.g. "1"): VariantFormer's
        # reference genome/gencode always use "chr"-prefixed contigs (e.g. "chr1"),
        # and bcftools consensus would otherwise silently apply zero variants for
        # every individual -- burning hours of GPU time to produce embeddings that
        # are all identical (reference-only) instead of raising an error.
        probe_vcf = individual_to_vcf[sample_ids[0]]
        if not vcf_uses_chr_prefix(probe_vcf):
            raise ValueError(
                f"VCF '{probe_vcf}' uses bare chromosome names (e.g. '1') but "
                "VariantFormer's reference genome/gencode use 'chr'-prefixed contigs "
                "(e.g. 'chr1'). bcftools consensus would silently apply zero variants "
                "for every individual if run as-is, producing identical embeddings "
                "across the whole cohort. Regenerate --vcf-dir with 'chr'-prefixed "
                "contigs first, e.g. via "
                "src/preprocessing/rename_vcf_chr_prefix.sh <in_dir> <out_dir>."
            )
    else:
        sample_ids = [None]
        logging.info("Reference-sequence mode (no --vcf-dir provided).")

    num_samples = len(sample_ids)

    # Optional DataLoader overrides forwarded to VCFProcessor.create_data (which
    # passes **kwargs straight to torch's DataLoader). Sequence extraction is the
    # bottleneck, so raising num_workers usually gives a near-linear speedup.
    loader_overrides: Dict[str, Any] = {}
    if num_workers is not None:
        loader_overrides["num_workers"] = num_workers
    if prefetch_factor is not None:
        loader_overrides["prefetch_factor"] = prefetch_factor
    if loader_overrides:
        logging.info("Overriding VariantFormer DataLoader config: %s", loader_overrides)

    logging.info("Loading VariantFormer model and checkpoint...")
    model, ckpt_path, trainer = vcf_processor.load_model()

    feats_mm_path = output_path.with_suffix(".features.npy")
    checkpoint_path = output_path.with_suffix(".checkpoint.json")

    # resume if checkpoint exists
    start_sample = 0
    emb_dim: Optional[int] = None
    ensids = chroms = tss = sample_ids_arr = gene_ids_arr = None
    feats_mm = None

    if checkpoint_path.exists():
        with checkpoint_path.open("r", encoding="utf-8") as f:
            cp = json.load(f)
        start_sample = cp.get("sample_idx", 0)
        emb_dim = cp.get("emb_dim")
        ensids = np.load(output_path.with_suffix(".ensids.npy"), allow_pickle=True)
        chroms = np.load(output_path.with_suffix(".chroms.npy"), allow_pickle=True)
        tss = np.load(output_path.with_suffix(".tss.npy"), allow_pickle=True)
        gene_ids_arr = np.load(output_path.with_suffix(".gene_ids.npy"), allow_pickle=True)
        sample_ids_arr = np.load(output_path.with_suffix(".sample_ids.npy"), allow_pickle=True)
        feats_mm = np.load(feats_mm_path, allow_pickle=True, mmap_mode="r+")
        logging.info("Resuming VariantFormer extraction from individual %d/%d", start_sample, num_samples)

    def allocate_outputs(dim: int) -> None:
        nonlocal feats_mm, ensids, chroms, tss, gene_ids_arr, sample_ids_arr
        total_rows = num_genes * num_samples
        logging.info(
            "Allocating output: %d genes x %d individuals = %d rows of dim %d",
            num_genes, num_samples, total_rows, dim,
        )
        ensids = np.empty(total_rows, dtype=object)
        chroms = np.empty(total_rows, dtype=object)
        tss = np.empty(total_rows, dtype=object)
        gene_ids_arr = np.empty(total_rows, dtype=object)
        sample_ids_arr = np.empty(total_rows, dtype=object)
        feats_mm = open_memmap(feats_mm_path, mode="w+", dtype=np.float32, shape=(total_rows, dim))

    def write_predictions(pred_df: "pd.DataFrame", s_idx: int, sample: Optional[str]) -> None:
        nonlocal emb_dim
        if feats_mm is None:
            emb_dim = int(np.asarray(pred_df.iloc[0]["embeddings"], dtype=np.float32).reshape(-1).shape[0])
            allocate_outputs(emb_dim)
        base = s_idx * num_genes
        for _, prow in pred_df.iterrows():
            gene_id = prow["gene_id"]
            meta = gene_meta_by_id.get(gene_id, {})
            idx = base + gene_row_offset[gene_id]
            feats_mm[idx, :] = np.asarray(prow["embeddings"], dtype=np.float32).reshape(-1)
            ensids[idx] = meta.get("ensid", gene_id)
            chroms[idx] = meta.get("chrom")
            tss[idx] = meta.get("tss", "")
            gene_ids_arr[idx] = gene_id
            sample_ids_arr[idx] = sample

    for s_idx in range(start_sample, num_samples):
        sample = sample_ids[s_idx]
        sample_label = sample if sample is not None else "REFERENCE"
        logging.info("Processing individual %d/%d: %s", s_idx + 1, num_samples, sample_label)

        # In personalized mode every individual has exactly one single-sample,
        # whole-genome VCF ("<individual>.vcf.gz", bgzipped + tabix-indexed).
        # We hand that file straight to VariantFormer, which runs
        # `bcftools consensus` to splice the individual's variants into each
        # gene/CRE window (seeking via the tabix index). In reference mode we
        # pass vcf_path=None so VariantFormer uses the unmodified reference
        # genome. A single prediction pass covers all genes and chromosomes for
        # the individual.
        vcf_path = str(individual_to_vcf[sample]) if personalized else None
        vcf_dataset, dataloader = vcf_processor.create_data(vcf_path, query_df, **loader_overrides)
        pred_df = vcf_processor.predict(model, ckpt_path, trainer, dataloader, vcf_dataset)
        write_predictions(pred_df, s_idx, sample)

        # checkpoint after each individual
        feats_mm.flush()
        np.save(output_path.with_suffix(".ensids.npy"), ensids)
        np.save(output_path.with_suffix(".chroms.npy"), chroms)
        np.save(output_path.with_suffix(".tss.npy"), tss)
        np.save(output_path.with_suffix(".gene_ids.npy"), gene_ids_arr)
        np.save(output_path.with_suffix(".sample_ids.npy"), sample_ids_arr)
        tmp_ckpt = checkpoint_path.with_suffix(".tmp")
        with tmp_ckpt.open("w", encoding="utf-8") as f:
            json.dump({"sample_idx": s_idx + 1, "num_genes": num_genes, "emb_dim": emb_dim}, f)
            f.flush()
            os.fsync(f.fileno())
        tmp_ckpt.replace(checkpoint_path)
        logging.info("Checkpoint saved after individual %d/%d", s_idx + 1, num_samples)

    if feats_mm is not None:
        feats_mm.flush()
    logging.info("VariantFormer embeddings successfully extracted to %s", feats_mm_path)


def get_features(
    data_path: Path,
    model_name: str,
    batch_size: int,
    window_size: int,
    output_path: Path,
    vcf_dir: Optional[Path] = None,
    num_individuals: str = "0",
    sample_seed: int = 42,
    checkpoint_every: int = 1000,
    tissue: Optional[str] = None,
    vf_model_class: str = "v4_pcg",
    maf_threshold: Optional[float] = None,
    num_workers: Optional[int] = None,
    prefetch_factor: Optional[int] = None,
    individual_split: Optional[str] = None,
    keep_ensids: Optional[Set[str]] = None,
    variformer_weights: Optional[Path] = None,
) -> None:
    """Extract features from sequences using specified model."""
    logging.info("Extracting features using model: %s", model_name)

    if model_name == "variformer":
        extract_variformer_features(
            data_path=data_path,
            output_path=output_path,
            weights_path=variformer_weights,
            batch_size=batch_size,
            window_size=window_size,
            vcf_dir=vcf_dir,
            num_individuals=num_individuals,
            sample_seed=sample_seed,
            individual_split=individual_split,
            maf_threshold=maf_threshold,
            keep_ensids=keep_ensids,
            checkpoint_every=checkpoint_every,
        )
        return

    if "variantformer" in model_name:
        if maf_threshold is not None:
            logging.warning(
                "--maf-threshold is only applied to backbones that build their own "
                "sequences; ignoring it for VariantFormer, which handles variants internally."
            )
        if keep_ensids is not None:
            logging.warning("--genes is not applied to the VariantFormer backbone; ignoring it.")
        extract_variantformer_features(
            data_path=data_path,
            output_path=output_path,
            tissue=tissue,
            vf_model_class=vf_model_class,
            vcf_dir=vcf_dir,
            num_individuals=num_individuals,
            sample_seed=sample_seed,
            individual_split=individual_split,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
        return

    personalized = vcf_dir is not None
    chrom_to_vcf: Dict[str, pysam.VariantFile] = {}
    sampled_ids: List[str] = []

    try:
        if personalized:
            logging.info("Personalized mode enabled using VCF directory: %s", vcf_dir)
            chrom_to_vcf = open_variant_files(vcf_dir)
            sampled_ids = sample_individual_ids(
                chrom_to_vcf,
                num_individuals,
                sample_seed,
                individual_split,
            )
        else:
            logging.info("Running in reference-sequence mode.")

        if model_name == "enformer":
            model = Enformer.from_pretrained("EleutherAI/enformer-official-rough").to(device)
            model.eval()

            feats_mm_path = output_path.with_suffix(".features.npy")

            total_rows = get_total_output_rows(
                input_path=data_path,
                personalized=personalized,
                num_individuals=len(sampled_ids) if personalized else 0,
                keep_ensids=keep_ensids,
            )
            logging.info("Number of output sequences to process: %d", total_rows)

            # check for existing checkpoint
            cidx = None
            checkpoint_path = output_path.with_suffix(".checkpoint.json")
            if checkpoint_path.exists():
                with checkpoint_path.open("r", encoding="utf-8") as f:
                    checkpoint = json.load(f)
                    cidx        = checkpoint.get("idx", 0)
                logging.info("Resuming from checkpoint at row %d", cidx)
                ensids         = np.load(output_path.with_suffix(".ensids.npy"), allow_pickle=True)
                chroms         = np.load(output_path.with_suffix(".chroms.npy"), allow_pickle=True)
                tss            = np.load(output_path.with_suffix(".tss.npy"), allow_pickle=True)
                sample_ids_arr = np.load(output_path.with_suffix(".sample_ids.npy"), allow_pickle=True) if personalized else None
                feats_mm       = np.load(output_path.with_suffix(".features.npy"), allow_pickle=True, mmap_mode="r+")
                logging.info("Checkpoint loaded. Resuming feature extraction from row %d", cidx)
            else:
                logging.debug("Creating memmap array at %s", feats_mm_path)
                ensids         = np.empty(total_rows, dtype=object)
                chroms         = np.empty(total_rows, dtype=object)
                tss            = np.empty(total_rows, dtype=object)
                sample_ids_arr = np.empty(total_rows, dtype=object) if personalized else None
                feats_mm       = open_memmap(feats_mm_path, mode="w+", dtype=np.float32, shape=(total_rows, 5313))

            start_idx = cidx or 0
            idx       = start_idx
            bidx      = 0
            remaining_rows = total_rows - start_idx
            num_batches = (remaining_rows + batch_size - 1) // batch_size

            row_iter: Iterator[Dict[str, Any]]
            if personalized:
                num_selected_ids = len(sampled_ids)
                skip_input_rows = start_idx // num_selected_ids
                skip_within_expanded_row = start_idx % num_selected_ids
                row_iter = iter_personalized_rows(
                    input_path=data_path,
                    chrom_to_vcf=chrom_to_vcf,
                    sampled_ids=sampled_ids,
                    skip_input_rows=skip_input_rows,
                    skip_within_expanded_row=skip_within_expanded_row,
                    maf_threshold=maf_threshold,
                    keep_ensids=keep_ensids,
                )
            else:
                row_iter = iter_autosomal_rows_csv(data_path, skip_rows=start_idx, keep_ensids=keep_ensids)

            checkpoint_path = output_path.with_suffix(".checkpoint.json")

            for batch in tqdm(batched_rows(row_iter, batch_size), total=num_batches, desc="Extracting features", unit="batch"):
                if personalized:
                    seqs_b       = [b["sequence_array"] for b in batch]
                    sample_ids_b = [b["sample_id"] for b in batch]
                else:
                    seqs_b = [dna_seq_to_array(b["sequence"].upper()) for b in batch]

                ensids_b = [b["ensid"] for b in batch]
                chroms_b = [b["chrom"] for b in batch]
                tss_b    = [b["tss"] for b in batch]

                batch_np     = np.stack(seqs_b, axis=0)
                batch_tensor = torch.as_tensor(batch_np, dtype=torch.long, device=device)

                with torch.no_grad():
                    output = model(batch_tensor)["human"]  # (B, 896, 5313)
                features = output.cpu().numpy().astype(np.float32)  # (B, 896, 5313)

                # select central bin, average over window size
                central_bin      = features.shape[1] // 2
                window_size_half = window_size // 2

                features = features[:, central_bin - window_size_half: central_bin + window_size_half, :]  # (B, W, 5313)
                features = features.mean(axis=1)  # (B, 5313)

                actual_bsz = features.shape[0]

                feats_mm[idx: idx + actual_bsz, :] = features
                ensids[idx: idx + actual_bsz]      = ensids_b
                chroms[idx: idx + actual_bsz]      = chroms_b
                tss[idx: idx + actual_bsz]         = tss_b
                if personalized and sample_ids_arr is not None:
                    sample_ids_arr[idx: idx + actual_bsz] = sample_ids_b

                idx  += actual_bsz
                bidx += 1

                if bidx % checkpoint_every == 0:
                    feats_mm.flush()
                    save_checkpoint(
                        checkpoint_path=checkpoint_path,
                        idx=idx,
                        ensids=ensids,
                        chroms=chroms,
                        tss=tss,
                        sample_ids_arr=sample_ids_arr,
                        output_path=output_path,
                    )
                    logging.info("Checkpoint saved at row %d", idx)

            feats_mm.flush()
            logging.info("Features successfully extracted.")
            logging.debug("Sample of extracted features:\n%s", feats_mm[0])

            np.save(output_path.with_suffix(".ensids.npy"), ensids)
            np.save(output_path.with_suffix(".chroms.npy"), chroms)
            np.save(output_path.with_suffix(".tss.npy"), tss)
            if personalized and sample_ids_arr is not None:
                np.save(output_path.with_suffix(".sample_ids.npy"), sample_ids_arr)

        else:
            raise ValueError(f"Model {model_name} not supported.")

    finally:
        if chrom_to_vcf:
            close_variant_files(chrom_to_vcf)


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    logging.debug("Arguments: %s", args)

    variformer_weights = None
    if args.model_name == "variformer" or args.self_test:
        variformer_weights = resolve_variformer_checkpoint(args.variformer_weights, args.variformer_fold)

    if args.self_test:
        variformer_self_test(variformer_weights)
        return

    keep_ensids = load_gene_subset(args.genes) if args.genes is not None else None
    if args.model_name == "variformer" and keep_ensids is None:
        logging.warning(
            "Running the 'variformer' backbone without --genes: its checkpoints "
            "were fine-tuned on ~300 genes only, so embeddings for other genes "
            "are out of distribution."
        )

    get_features(
        data_path=args.input,
        model_name=args.model_name,
        batch_size=args.batch_size,
        window_size=args.window_size,
        output_path=args.output,
        vcf_dir=args.vcf_dir,
        num_individuals=args.num_individuals,
        sample_seed=args.sample_seed,
        individual_split=args.individual_split,
        checkpoint_every=args.checkpoint_every,
        tissue=args.tissue,
        vf_model_class="v4_pcg" if args.model_name == "variantformer-pcg" else "v4_ag",
        maf_threshold=args.maf_threshold,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        keep_ensids=keep_ensids,
        variformer_weights=variformer_weights,
    )

    logging.info("Done.")


if __name__ == "__main__":
    main()
