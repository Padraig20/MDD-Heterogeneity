from __future__ import annotations
import argparse
import sys
import csv
import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd
import torch
from enformer_pytorch import Enformer
from numpy.lib.format import open_memmap
from tqdm import tqdm

import json
import os

import pysam

# we have very long sequences...
csv.field_size_limit(sys.maxsize)

"""
get_feats_from_seqs.py

Script that takes input from the human reference genome sequences we extracted
earlier (around the TSS) and then extracts features from these sequences using
some sort of gLM model. Here, we use e.g. Enformer via Hugging Face

Optionally personalize each gene window using genotype VCFs, selecting
individuals from the VCF headers and replacing SNPs in the sequence window.

Note: only biallelic SNPs are substituted, and heterozygous genotypes are currently
represented by ALT if any ALT allele is present.

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


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare target variables for downstream analysis."
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        help="Path to input file (*.csv)."
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Path to write output file (just name, *.npy suffix will be added)."
    )
    parser.add_argument(
        "-m", "--model-name",
        type=str,
        default="enformer",
        choices=["enformer", "variantformer-pcg", "variantformer-ag"],
        help="Name of the model to use for feature extraction. List of available models will be expanded in the future."
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
            "`bcftools +split`). Omit if personalization not needed."
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
        help="Random seed for integer-based sampling from the VCF header.",
    )
    parser.add_argument(
        "--maf-threshold",
        type=float,
        default=None,
        help=(
            "Minimum minor allele frequency for a SNP to be substituted into the "
            "Enformer input sequence (e.g. 0.05 for MAF >= 5%%). MAF is computed "
            "across the selected individuals. Only applies to the personalized "
            "'enformer' backbone. Defaults to no MAF filtering."
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
    return parser.parse_args()


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


def count_autosomal_rows_csv(input_path: Path) -> int:
    with input_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return sum(1 for row in reader if str(row["chrom"]).isdigit())


def get_total_output_rows(input_path: Path, personalized: bool, num_individuals: int) -> int:
    num_rows = count_autosomal_rows_csv(input_path)
    if not personalized:
        return num_rows
    return num_rows * num_individuals


def iter_autosomal_rows_csv(input_path: Path, skip_rows: int = 0) -> Iterator[Dict[str, Any]]:
    with input_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        kept   = 0
        for row in reader:
            if not str(row["chrom"]).isdigit():
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


def sample_individual_ids(chrom_to_vcf: Dict[str, "pysam.VariantFile"], selection: str, seed: int) -> List[str]:
    first_vcf = next(iter(chrom_to_vcf.values()))
    samples   = list(first_vcf.header.samples)
    return select_individual_ids(samples, selection, seed)


def get_vcf_samples(vcf_path: Path) -> List[str]:
    """Read the sample/individual IDs from a single VCF header."""
    vf = pysam.VariantFile(str(vcf_path))
    try:
        return list(vf.header.samples)
    finally:
        vf.close()


def discover_individual_vcfs(vcf_dir: Path) -> Dict[str, Path]:
    """Map individual ID -> per-individual VCF path for the VariantFormer backbone.

    Expects ``vcf_dir`` to contain one single-sample, whole-genome VCF per
    individual (e.g. produced by ``bcftools +split``), bgzipped and tabix-indexed
    as ``<individual>.vcf(.bgz|.gz)``. The individual ID is the filename with the
    VCF suffix stripped and is assumed to match the sample name inside the file.
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


def select_individual_ids(samples: List[str], selection: str, seed: int) -> List[str]:
    """Select individuals from a list of VCF sample IDs.

    Supports 'all', an integer count (randomly sampled), or a 'K/N' contiguous
    split of the header order (1-based).
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
        logging.info("Sampled %d individuals from VCF headers.", len(chosen))

    if not chosen:
        raise ValueError("--num-individuals selected zero individuals.")
    logging.debug("Selected individuals: %s", chosen[:5])
    return chosen


def collect_window_alt_edits(vcf: "pysam.VariantFile", chrom: str, start0: int, end0: int, sampled_ids: List[str], ref_seq: np.ndarray, maf_threshold: Optional[float] = None) -> List[List[tuple[int, int]]]:
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
            if gt and any(a is not None and a > 0 for a in gt):
                edits_per_individual[i].append((rel_pos, alt_code))

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


def iter_personalized_rows(input_path: Path, chrom_to_vcf: Dict[str, "pysam.VariantFile"], sampled_ids: List[str], skip_input_rows: int = 0, skip_within_expanded_row: int = 0, maf_threshold: Optional[float] = None) -> Iterator[Dict[str, Any]]:
    """Expand each input row into gene x sampled-individual rows with SNP-personalized sequences."""
    for row_idx, row in enumerate(iter_autosomal_rows_csv(input_path, skip_rows=skip_input_rows)):
        ref_seq = dna_seq_to_array(row["sequence"].upper())
        chrom   = row["chrom"]
        tss     = int(row["tss"])
        start0  = int(row["actual_start"])
        end0    = int(row["actual_end"])

        chrom = f"chr{chrom}" if not chrom.startswith("chr") else chrom

        edits_per_individual = collect_window_alt_edits(
            vcf=chrom_to_vcf[chrom],
            chrom=chrom,
            start0=start0,
            end0=end0,
            sampled_ids=sampled_ids,
            ref_seq=ref_seq,
            maf_threshold=maf_threshold,
        )

        start_i = skip_within_expanded_row if row_idx == 0 else 0

        for i in range(start_i, len(sampled_ids)):
            sample_id = sampled_ids[i]

            seq_arr = ref_seq.copy()
            for rel_pos, alt_code in edits_per_individual[i]:
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


def extract_variantformer_features(
    data_path: Path,
    output_path: Path,
    tissue: str,
    vf_model_class: str = "v4_pcg",
    vcf_dir: Optional[Path] = None,
    num_individuals: str = "0",
    sample_seed: int = 42,
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

    def build_query_df(gene_ids: List[str]) -> "pd.DataFrame":
        return pd.DataFrame(
            {"gene_id": gene_ids, "tissues": [tissue] * len(gene_ids)}
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
            select_individual_ids(all_individuals, num_individuals, sample_seed)
        )
        logging.info(
            "Personalized mode: %d of %d per-individual VCFs from %s",
            len(sample_ids), len(all_individuals), vcf_dir,
        )
    else:
        sample_ids = [None]
        logging.info("Reference-sequence mode (no --vcf-dir provided).")

    num_samples = len(sample_ids)

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

        if personalized:
            # Single prediction pass over all genes/chromosomes for this
            # individual. The VCF is already single-sample and whole-genome, so
            # we feed it directly: no sample subsetting is needed (the file is
            # one individual) and bcftools consensus seeks to each gene/CRE
            # window via the tabix index, so pre-restricting regions is
            # unnecessary. The VCF must be bgzipped and tabix-indexed.
            ind_vcf = individual_to_vcf[sample]
            vcf_dataset, dataloader = vcf_processor.create_data(
                str(ind_vcf),
                build_query_df(ordered_gene_ids),
                sample_ids=None,
            )
            pred_df = vcf_processor.predict(model, ckpt_path, trainer, dataloader, vcf_dataset)
            write_predictions(pred_df, s_idx, sample)
        else:
            vcf_dataset, dataloader = vcf_processor.create_data(
                None, build_query_df(ordered_gene_ids), sample_ids=None
            )
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


def get_features(data_path: Path, model_name: str, batch_size: int, window_size: int, output_path: Path, vcf_dir: Optional[Path] = None, num_individuals: str = "0", sample_seed: int = 42, checkpoint_every: int = 1000, tissue: Optional[str] = None, vf_model_class: str = "v4_pcg", maf_threshold: Optional[float] = None) -> None:
    """Extract features from sequences using specified model."""
    logging.info("Extracting features using model: %s", model_name)

    if "variantformer" in model_name:
        if maf_threshold is not None:
            logging.warning(
                "--maf-threshold is only applied to the 'enformer' backbone; ignoring it "
                "for VariantFormer, which handles variants internally."
            )
        extract_variantformer_features(
            data_path=data_path,
            output_path=output_path,
            tissue=tissue,
            vf_model_class=vf_model_class,
            vcf_dir=vcf_dir,
            num_individuals=num_individuals,
            sample_seed=sample_seed,
        )
        return

    personalized = vcf_dir is not None
    chrom_to_vcf: Dict[str, pysam.VariantFile] = {}
    sampled_ids: List[str] = []

    try:
        if personalized:
            logging.info("Personalized mode enabled using VCF directory: %s", vcf_dir)
            chrom_to_vcf = open_variant_files(vcf_dir)
            sampled_ids  = sample_individual_ids(chrom_to_vcf, num_individuals, sample_seed)
        else:
            logging.info("Running in reference-sequence mode.")

        if model_name == "enformer":
            model = Enformer.from_pretrained("EleutherAI/enformer-official-rough").to(device)
            model.eval()

            feats_mm_path = output_path.with_suffix(".features.npy")

            total_rows = get_total_output_rows(
                input_path=data_path,
                personalized=personalized,
                num_individuals=len(sampled_ids) if personalized else 0
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
                )
            else:
                row_iter = iter_autosomal_rows_csv(data_path, skip_rows=start_idx)

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

    get_features(
        data_path=args.input,
        model_name=args.model_name,
        batch_size=args.batch_size,
        window_size=args.window_size,
        output_path=args.output,
        vcf_dir=args.vcf_dir,
        num_individuals=args.num_individuals,
        sample_seed=args.sample_seed,
        checkpoint_every=args.checkpoint_every,
        tissue=args.tissue,
        vf_model_class="v4_pcg" if args.model_name == "variantformer-pcg" else "v4_ag",
        maf_threshold=args.maf_threshold,
    )

    logging.info("Done.")


if __name__ == "__main__":
    main()