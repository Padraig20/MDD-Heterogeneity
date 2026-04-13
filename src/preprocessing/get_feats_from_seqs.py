from __future__ import annotations
import argparse
import sys
import csv
import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
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

Optionally personalize each gene window using genotype VCFs, sampling N
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
        choices=["enformer"],
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
            "Optional directory containing chromosome-level VCF files "
            "(e.g. chr1.vcf.gz ... chr22.vcf.gz). Omit if personalization not needed."
        )
    )
    parser.add_argument(
        "--num-individuals",
        type=int,
        default=0,
        help=(
            "Number of individuals to randomly sample from the VCF files when "
            "--vcf-dir is provided."
        )
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed for sampling individuals from the VCF header.",
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
    np.save(output_path.with_suffix(".ensids.npy"), ensids[:idx])
    np.save(output_path.with_suffix(".chroms.npy"), chroms[:idx])
    np.save(output_path.with_suffix(".tss.npy"), tss[:idx])
    if sample_ids_arr is not None:
        np.save(output_path.with_suffix(".sample_ids.npy"), sample_ids_arr[:idx])

    # progress file for tracking checkpointing progress
    tmp_ckpt = checkpoint_path.with_suffix(".tmp")
    with tmp_ckpt.open("w", encoding="utf-8") as f:
        json.dump({"idx": idx}, f)
        f.flush()
        os.fsync(f.fileno())
    tmp_ckpt.replace(checkpoint_path)


def count_rows_csv(input_path: Path) -> int:
    with input_path.open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1  # exclude header


def iter_rows_csv(input_path: Path) -> Iterator[Dict[str, Any]]:
    with input_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        yield from reader  # already excludes header


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


def sample_individual_ids(chrom_to_vcf: Dict[str, "pysam.VariantFile"], num_individuals: int, seed: int) -> List[str]:
    first_vcf = next(iter(chrom_to_vcf.values()))
    samples   = list(first_vcf.header.samples)
    rng       = random.Random(seed)
    chosen    = rng.sample(samples, num_individuals)
    logging.info("Sampled %d individuals from VCF headers.", len(chosen))
    logging.debug("Sampled individuals: %s", chosen[:5])
    return chosen


def collect_window_alt_edits(vcf: "pysam.VariantFile", chrom: str, start0: int, end0: int, sampled_ids: List[str], ref_seq: np.ndarray) -> List[List[tuple[int, int]]]:
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
        for i, sid in enumerate(sampled_ids):
            gt = rec_samples[sid]["GT"]
            if gt and any(a is not None and a > 0 for a in gt):
                edits_per_individual[i].append((rel_pos, alt_code))

    return edits_per_individual


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


def iter_personalized_rows(input_path: Path, chrom_to_vcf: Dict[str, "pysam.VariantFile"], sampled_ids: List[str]) -> Iterator[Dict[str, Any]]:
    """Expand each input row into gene x sampled-individual rows with SNP-personalized sequences."""
    for row in iter_rows_csv(input_path):
        ref_seq = dna_seq_to_array(row["sequence"].upper())
        chrom   = row["chrom"]
        start0  = int(row["actual_start"])
        end0    = int(row["actual_end"])

        if not str(chrom).isdigit(): # skip non-autosomal chromosomes
            continue

        chrom = f"chr{chrom}" if not chrom.startswith("chr") else chrom

        edits_per_individual = collect_window_alt_edits(
            vcf=chrom_to_vcf[chrom],
            chrom=chrom,
            start0=start0,
            end0=end0,
            sampled_ids=sampled_ids,
            ref_seq=ref_seq,
        )

        for i, sample_id in enumerate(sampled_ids):
            seq_arr = ref_seq.copy()
            for rel_pos, alt_code in edits_per_individual[i]:
                seq_arr[rel_pos] = alt_code

            yield {
                "sequence_array": seq_arr,
                "ensid": row["ensid"],
                "chrom": row["chrom"],
                "sample_id": sample_id,
            }


def get_total_output_rows(input_path: Path, personalized: bool, num_individuals: int) -> int:
    num_rows = count_rows_csv(input_path)
    if not personalized:
        return num_rows
    return num_rows * num_individuals


def get_features(data_path: Path, model_name: str, batch_size: int, window_size: int, output_path: Path, vcf_dir: Optional[Path] = None, num_individuals: int = 0, sample_seed: int = 42, checkpoint_every: int = 1000) -> None:
    """Extract features from sequences using specified model."""
    logging.info("Extracting features using model: %s", model_name)

    personalized = vcf_dir is not None
    chrom_to_vcf: Dict[str, pysam.VariantFile] = {}
    sampled_ids: List[str] = []

    try:
        if personalized:
            logging.info("Personalized mode enabled using VCF directory: %s", vcf_dir)
            chrom_to_vcf = open_variant_files(vcf_dir)
            sampled_ids = sample_individual_ids(chrom_to_vcf, num_individuals, sample_seed)
        else:
            logging.info("Running in reference-sequence mode.")

        if model_name == "enformer":
            model = Enformer.from_pretrained("EleutherAI/enformer-official-rough", dtype="auto").to(device)
            model.eval()

            feats_mm_path = output_path.with_suffix(".features.npy")

            total_rows = get_total_output_rows(
                input_path=data_path,
                personalized=personalized,
                num_individuals=num_individuals if personalized else 0
            )
            logging.info("Number of output sequences to process: %d", total_rows)

            logging.debug("Creating memmap array at %s", feats_mm_path)
            feats_mm = open_memmap(
                feats_mm_path,
                mode="w+",
                dtype=np.float32,
                shape=(total_rows, 5313),
            )

            ensids         = np.empty(total_rows, dtype=object)
            chroms         = np.empty(total_rows, dtype=object)
            tss            = np.empty(total_rows, dtype=object)
            sample_ids_arr = np.empty(total_rows, dtype=object) if personalized else None

            row_iter: Iterator[Dict[str, Any]]
            if personalized:
                row_iter = iter_personalized_rows(
                    input_path=data_path,
                    chrom_to_vcf=chrom_to_vcf,
                    sampled_ids=sampled_ids,
                )
            else:
                row_iter = iter_rows_csv(data_path)

            idx  = 0
            bidx = 0
            num_batches = (total_rows + batch_size - 1) // batch_size

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

            logging.info("Features successfully extracted.")
            logging.debug("Sample of extracted features:\n%s", feats_mm[0])
            logging.debug("Features saved to %s, now flushing...", feats_mm_path)

            del feats_mm

            np.save(output_path.with_suffix(".ensids.npy"), ensids)
            np.save(output_path.with_suffix(".chroms.npy"), chroms)
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
    )

    logging.info("Done.")


if __name__ == "__main__":
    main()