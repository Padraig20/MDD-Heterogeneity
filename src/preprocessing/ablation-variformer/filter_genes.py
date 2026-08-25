from __future__ import annotations
import argparse
import csv
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

# gene windows are hundreds of kb of sequence per CSV field
csv.field_size_limit(sys.maxsize)

"""
filter_genes.py

Restrict the artefacts of the sequence -> embedding pipeline to the gene subset
used for the VariantFormer ablation study (see 300_train_genes.tsv).

Three kinds of input can be filtered, in any combination:

  1. A gene-window CSV produced by `get_obs_vars.py` (columns: chrom, ensid,
     tss, start, end, sequence, actual_start, actual_end).
  2. A single embedding set produced by `get_feats_from_seqs.py`, addressed by
     its output prefix, i.e. `<prefix>.features.npy` plus the `.ensids.npy`,
     `.chroms.npy`, `.tss.npy`, and (personalized / VariantFormer runs)
     `.sample_ids.npy` and `.gene_ids.npy` sidecars.
  3. A directory of such embedding sets, e.g. one prefix per `--individual-split`
     shard, or one subdirectory per individual holding bare `features.npy` /
     `ensids.npy` / ... files. The directory tree is mirrored into the output
     directory.

ENSEMBL IDs are matched on their unversioned base, so `ENSG00000196557.12` and
`ENSG00000196557` are treated as the same gene.
"""

ENSID_PATTERN = re.compile(r"\bENSG\d+(?:\.\d+)?\b")

FEATURES_SUFFIX = ".features.npy"

# Sidecar arrays written by get_feats_from_seqs.py, all row-aligned with the
# feature matrix. Not every run writes all of them (`sample_ids` only in
# personalized mode, `gene_ids` only for the VariantFormer backbone).
METADATA_FIELDS = ("ensids", "chroms", "tss", "sample_ids", "gene_ids")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Filter gene-window CSVs and embedding sets down to an ablation "
            "gene subset."
        )
    )
    parser.add_argument(
        "-g", "--genes",
        type=Path,
        required=True,
        help=(
            "Path to the gene subset file (e.g. 300_train_genes.tsv). Any "
            "delimited text file works; every ENSEMBL gene ID found in it is "
            "used, with version suffixes ignored."
        )
    )
    parser.add_argument(
        "-i-csv", "--input-csv",
        type=Path,
        default=None,
        help="Path to a gene-window CSV produced by get_obs_vars.py."
    )
    parser.add_argument(
        "-o-csv", "--output-csv",
        type=Path,
        default=None,
        help="Path to write the filtered gene-window CSV. Required with --input-csv."
    )
    parser.add_argument(
        "-i-prefix", "--input-prefix",
        type=Path,
        default=None,
        help=(
            "Prefix of an embedding set produced by get_feats_from_seqs.py, "
            "i.e. the path without the '.features.npy' / '.ensids.npy' suffixes."
        )
    )
    parser.add_argument(
        "-o-prefix", "--output-prefix",
        type=Path,
        default=None,
        help="Prefix to write the filtered embedding set to. Required with --input-prefix."
    )
    parser.add_argument(
        "-i-dir", "--input-dir",
        type=Path,
        default=None,
        help=(
            "Directory holding one or more embedding sets (searched "
            "recursively for '*.features.npy' and 'features.npy')."
        )
    )
    parser.add_argument(
        "-o-dir", "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory to write the filtered embedding sets to, mirroring the "
            "layout of --input-dir. Required with --input-dir."
        )
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=4096,
        help=(
            "Number of feature rows copied per read/write step. Keeps peak "
            "memory bounded for large personalized feature matrices."
        )
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity."
    )
    args = parser.parse_args()

    modes = [
        ("--input-csv", args.input_csv, "--output-csv", args.output_csv),
        ("--input-prefix", args.input_prefix, "--output-prefix", args.output_prefix),
        ("--input-dir", args.input_dir, "--output-dir", args.output_dir),
    ]
    if not any(inp is not None for _, inp, _, _ in modes):
        parser.error(
            "Nothing to do: pass at least one of --input-csv, --input-prefix, "
            "or --input-dir."
        )
    for in_flag, inp, out_flag, out in modes:
        if inp is not None and out is None:
            parser.error(f"{in_flag} requires {out_flag}.")
        if inp is None and out is not None:
            parser.error(f"{out_flag} requires {in_flag}.")
    if args.chunk_rows < 1:
        parser.error("--chunk-rows must be at least 1.")

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


def base_ensid(ensid: object) -> str:
    """Strip the version suffix from an ENSEMBL gene ID ('ENSG1.3' -> 'ENSG1')."""
    return str(ensid).strip().split(".", 1)[0]


def load_gene_subset(genes_path: Path) -> set[str]:
    """Read the unversioned ENSEMBL gene IDs of the ablation subset."""
    logging.info("Loading gene subset from %s", genes_path)
    text = genes_path.read_text(encoding="utf-8")
    wanted = {base_ensid(match) for match in ENSID_PATTERN.findall(text)}
    if not wanted:
        raise ValueError(
            f"No ENSEMBL gene IDs (ENSG...) found in {genes_path}. The gene "
            "subset file must contain ENSEMBL IDs, not just gene symbols."
        )
    logging.info("Gene subset contains %d unique genes.", len(wanted))
    return wanted


def report_missing(found: set[str], wanted: set[str], source: str) -> None:
    """Log how much of the requested subset was actually present in `source`."""
    missing = wanted - found
    logging.info("%s: matched %d of %d subset genes.", source, len(found), len(wanted))
    if missing:
        preview = sorted(missing)
        logging.warning(
            "%s: %d subset genes absent, e.g. %s",
            source, len(missing), preview[:10],
        )


def filter_obs_vars_csv(input_path: Path, output_path: Path, wanted: set[str]) -> None:
    """Copy the rows of a get_obs_vars.py CSV whose `ensid` is in `wanted`.

    Streamed row by row: the `sequence` column holds the full window around the
    TSS, so these files reach tens of GB and must not be loaded at once.
    """
    logging.info("Filtering gene-window CSV %s -> %s", input_path, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    found: set[str] = set()
    kept = total = 0

    with input_path.open(newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None or "ensid" not in reader.fieldnames:
            raise ValueError(
                f"{input_path} has no 'ensid' column; expected a CSV produced "
                "by get_obs_vars.py."
            )
        with output_path.open("w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                total += 1
                ensid = base_ensid(row["ensid"])
                if ensid in wanted:
                    writer.writerow(row)
                    found.add(ensid)
                    kept += 1

    logging.info("Kept %d of %d rows.", kept, total)
    report_missing(found, wanted, str(input_path))


@dataclass(frozen=True)
class EmbeddingSet:
    """One `get_feats_from_seqs.py` output set: a feature matrix plus sidecars.

    `stem` is the output prefix's file name; it is empty for the per-individual
    directory layout where the files are simply `features.npy`, `ensids.npy`, ...
    """
    directory: Path
    stem: str

    @classmethod
    def from_features_path(cls, features_path: Path) -> "EmbeddingSet":
        name = features_path.name
        stem = "" if name == "features.npy" else name[: -len(FEATURES_SUFFIX)]
        return cls(features_path.parent, stem)

    def path(self, field: str, ext: str = ".npy") -> Path:
        name = f"{self.stem}.{field}{ext}" if self.stem else f"{field}{ext}"
        return self.directory / name

    def __str__(self) -> str:
        return str(self.directory / self.stem) if self.stem else f"{self.directory}/"


def discover_embedding_sets(input_dir: Path) -> list[EmbeddingSet]:
    """Find every embedding set below `input_dir`, in both supported layouts."""
    features_paths = [
        path for path in sorted(input_dir.rglob("*features.npy"))
        if path.name == "features.npy" or path.name.endswith(FEATURES_SUFFIX)
    ]
    if not features_paths:
        raise FileNotFoundError(
            f"No embedding sets found under {input_dir}: expected "
            "'<prefix>.features.npy' files, or per-individual subdirectories "
            "containing 'features.npy'."
        )
    return [EmbeddingSet.from_features_path(path) for path in features_paths]


def mirror_embedding_set(source: EmbeddingSet, input_dir: Path, output_dir: Path) -> EmbeddingSet:
    """Map a discovered embedding set onto its place in the output directory."""
    relative = source.directory.relative_to(input_dir)
    return EmbeddingSet(output_dir / relative, source.stem)


def load_row_ensids(source: EmbeddingSet) -> np.ndarray:
    """Load the per-row gene IDs of an embedding set.

    Prefers `.ensids.npy`; VariantFormer runs that only kept the versioned
    `.gene_ids.npy` fall back to that.
    """
    for field in ("ensids", "gene_ids"):
        path = source.path(field)
        if path.exists():
            return np.load(path, allow_pickle=True)
    raise FileNotFoundError(
        f"Embedding set {source} has neither {source.path('ensids')} nor "
        f"{source.path('gene_ids')}; cannot tell which gene each row belongs to."
    )


def copy_selected_features(src_path: Path, dst_path: Path, row_idx: np.ndarray, chunk_rows: int) -> None:
    """Copy the selected feature rows, streaming through a memory map.

    Personalized feature matrices have one row per (gene, individual) pair and
    routinely exceed available RAM, so both sides stay on disk and rows are
    moved in bounded chunks.
    """
    src = np.load(src_path, allow_pickle=True, mmap_mode="r")
    dst = open_memmap(
        dst_path,
        mode="w+",
        dtype=src.dtype,
        shape=(len(row_idx),) + src.shape[1:],
    )
    try:
        for start in range(0, len(row_idx), chunk_rows):
            chunk = row_idx[start: start + chunk_rows]
            dst[start: start + len(chunk)] = src[chunk]
        dst.flush()
    finally:
        del dst
        del src


def filter_embedding_set(
    source: EmbeddingSet,
    destination: EmbeddingSet,
    wanted: set[str],
    chunk_rows: int,
) -> set[str]:
    """Write the rows of `source` belonging to `wanted` into `destination`.

    Returns the subset genes that were present in `source`.
    """
    logging.info("Filtering embedding set %s -> %s", source, destination)

    features_path = source.path("features")
    if not features_path.exists():
        raise FileNotFoundError(f"Missing feature matrix {features_path}.")

    row_ensids = load_row_ensids(source)
    row_bases = np.array([base_ensid(e) for e in row_ensids], dtype=object)
    mask = np.fromiter((base in wanted for base in row_bases), dtype=bool, count=len(row_bases))
    row_idx = np.flatnonzero(mask)

    found = set(row_bases[row_idx].tolist())
    logging.info("Keeping %d of %d rows (%d genes).", len(row_idx), len(row_bases), len(found))

    if len(row_idx) == 0:
        logging.warning("%s: no rows matched the gene subset; skipping.", source)
        return found

    destination.directory.mkdir(parents=True, exist_ok=True)
    copy_selected_features(features_path, destination.path("features"), row_idx, chunk_rows)

    for field in METADATA_FIELDS:
        path = source.path(field)
        if not path.exists():
            logging.debug("%s: no %s sidecar; skipping.", source, field)
            continue
        array = np.load(path, allow_pickle=True)
        if len(array) != len(row_bases):
            raise ValueError(
                f"{path} has {len(array)} rows but the feature matrix has "
                f"{len(row_bases)}; the embedding set is inconsistent."
            )
        np.save(destination.path(field), array[row_idx])

    # A resume checkpoint describes the *unfiltered* run, so carrying it over
    # would make get_feats_from_seqs.py skip work on the filtered output.
    checkpoint = source.path("checkpoint", ".json")
    if checkpoint.exists():
        logging.debug("Not copying resume checkpoint %s.", checkpoint)

    return found


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    logging.debug("Arguments: %s", args)

    wanted = load_gene_subset(args.genes)

    if args.input_csv is not None:
        filter_obs_vars_csv(args.input_csv, args.output_csv, wanted)

    if args.input_prefix is not None:
        source = EmbeddingSet(args.input_prefix.parent, args.input_prefix.name)
        destination = EmbeddingSet(args.output_prefix.parent, args.output_prefix.name)
        found = filter_embedding_set(source, destination, wanted, args.chunk_rows)
        report_missing(found, wanted, str(source))

    if args.input_dir is not None:
        sources = discover_embedding_sets(args.input_dir)
        logging.info("Found %d embedding sets under %s.", len(sources), args.input_dir)
        found: set[str] = set()
        for source in sources:
            destination = mirror_embedding_set(source, args.input_dir, args.output_dir)
            found |= filter_embedding_set(source, destination, wanted, args.chunk_rows)
        report_missing(found, wanted, str(args.input_dir))

    logging.info("Done.")


if __name__ == "__main__":
    main()
