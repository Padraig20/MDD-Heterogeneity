from __future__ import annotations

import argparse
import gzip
import logging
import re
import sys
from pathlib import Path

import pandas as pd

from src.twas.compare import normalize_cell_type

"""
get_shared_genes.py

The gene universe a TWAS comparison is allowed to use.

VariantFormer and Enformer (ctPred) are trained on different gene sets --
17,839 vs 18,229 on the current data -- so a comparison that started from the
distilled weights would mix that annotation difference with the teacher
difference, and would also drop every gene a fit failed to converge on. The
student-prediction CSVs from `src/distillation/get_student_data.py` are the
teachers' gene lists *before* distillation, and that is the intersection
written here.

    python -m src.twas.get_shared_genes \\
        --ours student-preds/variantformer \\
        --ctpred student-preds/ctpred \\
        --gtf gencode.v38.annotation.gtf.gz \\
        --output shared_genes.txt

`src/twas/run.py --shared-genes shared_genes.txt` then keeps only these genes
and uses the GTF symbols for every plot label. A gene on the list that a
distilled model never produced is left missing: that is a failure of the
fit, not a reason to shrink this file.
"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Intersect the genes of two student-prediction directories, "
            "pair each Ensembl id with its GTF gene name, and write "
            "shared_genes.txt."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ours",
        type=Path,
        required=True,
        help=(
            "Student-prediction directory for this study's teacher "
            "(get_student_data.py output: one CSV per cell type, or a preds/ "
            "subdirectory)."
        ),
    )
    parser.add_argument(
        "--ctpred",
        type=Path,
        required=True,
        help="The same, for the ctPred teacher.",
    )
    parser.add_argument(
        "--gtf",
        type=Path,
        required=True,
        help=(
            "GENCODE / Ensembl GTF (plain or .gz). gene_id is paired with "
            "gene_name; every shared Ensembl id that cannot be mapped is "
            "warned and written with an empty symbol."
        ),
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("shared_genes.txt"),
        help="TSV with columns ENSID and Gene, one shared gene per line.",
    )
    parser.add_argument(
        "--cell-types",
        type=str,
        nargs="+",
        metavar="CELL_TYPE",
        default=None,
        help=(
            "Restrict the scan to these cell-type CSVs. Defaults to every "
            "prediction CSV in each directory."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def discover_prediction_csvs(directory: Path) -> list[Path]:
    """
    The mean-prediction CSVs of a get_student_data.py output directory.

    Prefers `preds/*.csv` when that subdirectory exists (the ensemble layout),
    otherwise `*.csv` at the top level. Uncertainty and per-member folders
    are ignored: they carry the same genes as the means.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    search = directory / "preds" if (directory / "preds").is_dir() else directory
    csvs = sorted(path for path in search.glob("*.csv") if path.is_file())
    if csvs:
        return csvs

    members = directory / "members"
    if members.is_dir():
        for member in sorted(members.glob("member_*/preds")):
            csvs = sorted(member.glob("*.csv"))
            if csvs:
                logging.info(
                    "No top-level preds/ in %s; using %s.", directory, member
                )
                return csvs

    raise FileNotFoundError(
        f"No student-prediction CSVs found in {directory}. Expected "
        f"{directory}/*.csv or {directory}/preds/*.csv."
    )


def _select_csvs(csvs: list[Path], requested: list[str] | None) -> list[Path]:
    if requested is None:
        return csvs
    wanted = {normalize_cell_type(name) for name in requested}
    selected = [path for path in csvs if normalize_cell_type(path.stem) in wanted]
    missing = wanted - {normalize_cell_type(path.stem) for path in selected}
    if missing:
        raise FileNotFoundError(
            f"No prediction CSV for cell type(s) {sorted(missing)}. "
            f"Available: {sorted(path.stem for path in csvs)}"
        )
    return selected


def read_prediction_genes(path: Path) -> dict[str, str]:
    """
    `{versionless key -> id as written}` from one prediction CSV.

    Only the gene column is read. The original id is kept so the shared file
    can prefer this study's spelling when the two teachers version a gene
    differently.
    """
    header = pd.read_csv(path, nrows=0)
    column = next(
        (name for name in ("gene", "ENSID", "ensid") if name in header.columns),
        header.columns[0],
    )
    values = pd.read_csv(path, usecols=[column])[column].astype(str)
    mapping: dict[str, str] = {}
    for raw in values:
        key = str(raw).split(".")[0].strip().upper()
        if key and key != "NAN":
            mapping.setdefault(key, str(raw).strip())
    if not mapping:
        raise ValueError(f"{path} has no gene identifiers in column {column!r}.")
    return mapping


def genes_in_directory(
    directory: Path, cell_types: list[str] | None
) -> dict[str, str]:
    """Union of genes across the prediction CSVs of one teacher."""
    csvs = _select_csvs(discover_prediction_csvs(directory), cell_types)
    logging.info("Reading %d prediction CSV(s) from %s.", len(csvs), directory)
    combined: dict[str, str] = {}
    sizes: list[int] = []
    for path in csvs:
        mapping = read_prediction_genes(path)
        sizes.append(len(mapping))
        for key, raw in mapping.items():
            combined.setdefault(key, raw)
        logging.info("  %s: %d gene(s).", path.name, len(mapping))
    if sizes and min(sizes) != max(sizes):
        logging.warning(
            "Cell-type CSVs in %s do not all have the same gene count "
            "(%d–%d); the union (%d) is used.",
            directory, min(sizes), max(sizes), len(combined),
        )
    return combined


# GTF / GFF3 attribute keys. GENCODE quotes them; GFF3 uses ID=/Name=.
_GTF_GENE_ID = re.compile(r'(?:gene_id|ID)\s*[="]\s*([^";]+)')
_GTF_GENE_NAME = re.compile(
    r'(?:gene_name|gene_symbol|Name)\s*[="]\s*([^";]+)'
)


def _open_text(path: Path):
    path = Path(path)
    if path.suffix == ".gz" or path.name.endswith(".gtf.gz"):
        return gzip.open(path, "rt")
    return path.open("r")


def _gtf_attributes(attrs: str) -> tuple[str | None, str | None]:
    gene_id = _GTF_GENE_ID.search(attrs)
    gene_name = _GTF_GENE_NAME.search(attrs)
    identifier = gene_id.group(1).strip() if gene_id else None
    symbol = gene_name.group(1).strip() if gene_name else None
    return identifier, symbol


def load_gtf_gene_names(path: Path) -> dict[str, str]:
    """
    `{versionless Ensembl id -> gene symbol}` from a GTF.

    Prefers `gene` features. If the file has no gene rows (exon-only GTFs),
    any feature that carries both `gene_id` and `gene_name` is used instead.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"GTF not found: {path}")

    from_gene: dict[str, str] = {}
    from_any: dict[str, str] = {}
    with _open_text(path) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 9:
                continue
            identifier, symbol = _gtf_attributes(parts[8])
            if not identifier or not symbol:
                continue
            key = identifier.split(".")[0].strip().upper()
            if not key:
                continue
            if parts[2] == "gene":
                from_gene.setdefault(key, symbol)
            else:
                from_any.setdefault(key, symbol)

    mapping = from_gene or from_any
    if not mapping:
        raise ValueError(
            f"{path} has no gene_id/gene_name pairs. Check that it is a "
            "GENCODE or Ensembl gene annotation GTF."
        )
    logging.info(
        "Read %d gene name(s) from %s.", len(mapping), path
    )
    return mapping


def map_gene_names(ids: list[str], gtf_names: dict[str, str]) -> dict[str, str]:
    """
    Pair each Ensembl id with its GTF symbol.

    Unmapped genes stay out of the returned dict and are logged one WARNING
    each; the caller writes them with an empty Gene column.
    """
    names: dict[str, str] = {}
    missing: list[str] = []
    for gene in ids:
        key = gene.split(".")[0].strip().upper()
        symbol = gtf_names.get(key)
        if symbol:
            names[key] = symbol
        else:
            missing.append(gene)
    for gene in missing:
        logging.warning("No GTF gene name for %s.", gene)
    if missing:
        logging.warning(
            "%d of %d shared gene(s) have no GTF name; plots will fall back "
            "to the Ensembl id.",
            len(missing), len(ids),
        )
    else:
        logging.info("Paired all %d shared gene(s) with a GTF name.", len(ids))
    return names


def write_gene_list(
    path: Path, ids: list[str], names: dict[str, str] | None = None
) -> None:
    """TSV with columns ENSID and Gene, matching `data/mdd_genes.tsv`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = names or {}
    lines = ["ENSID\tGene\n"]
    for gene in ids:
        key = gene.split(".")[0].strip().upper()
        lines.append(f"{gene}\t{names.get(key, '')}\n")
    path.write_text("".join(lines))


def intersect_prediction_genes(
    ours_dir: Path,
    ctpred_dir: Path,
    cell_types: list[str] | None = None,
) -> tuple[list[str], dict]:
    """
    Shared genes of two student-prediction directories.

    Returns the identifiers (preferring this study's spelling) and a small
    count dict for the log line.
    """
    ours = genes_in_directory(ours_dir, cell_types)
    theirs = genes_in_directory(ctpred_dir, cell_types)
    shared_keys = sorted(set(ours) & set(theirs))
    if not shared_keys:
        raise ValueError(
            "The two student-prediction directories share no gene. Check that "
            "both came from get_student_data.py against the same annotation."
        )
    ids = [ours.get(key, theirs[key]) for key in shared_keys]
    stats = {
        "n_ours": len(ours),
        "n_ctpred": len(theirs),
        "n_shared": len(ids),
        "n_ours_only": len(ours) - len(shared_keys),
        "n_ctpred_only": len(theirs) - len(shared_keys),
    }
    return ids, stats


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        ids, stats = intersect_prediction_genes(
            args.ours, args.ctpred, args.cell_types
        )
        gtf_names = load_gtf_gene_names(args.gtf)
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        logging.error("%s", error)
        sys.exit(1)

    names = map_gene_names(ids, gtf_names)
    write_gene_list(args.output, ids, names)
    logging.info(
        "Wrote %d shared gene(s) (%d with a GTF name) to %s (ours %d, "
        "ctPred %d, ours-only %d, ctPred-only %d).",
        stats["n_shared"], len(names), args.output, stats["n_ours"],
        stats["n_ctpred"], stats["n_ours_only"], stats["n_ctpred_only"],
    )
    logging.info(
        "Pass this file to TWAS as `--shared-genes %s`.", args.output
    )


if __name__ == "__main__":
    main()
