from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

"""
sprediXcan.py

Run `metaxcan/software/SPrediXcan.py` on one model DB.

It is invoked as a subprocess rather than imported: `SPrediXcan.py` does bare
`import M03_betas` / `import M04_zscores`, which only resolve when its own
directory is the working directory, and `metax.Logging` reconfigures the root
logger process-wide.

`--gwas_h2` / `--gwas_N` are deliberately not exposed. Passing them sends
`M04_zscores` into `correct_inf_phi`, which runs `SELECT gene, phi FROM extra`
and then inner-joins on the result. The model DBs written by `model_db.py`
follow the standard PredictXcan schema, which has no `phi` column, so that path
would either raise or silently discard every gene.
"""

SCRIPT_NAME = "SPrediXcan.py"

# GWAS options forwarded verbatim to SPrediXcan.py, as
# `attribute name -> command line flag`.
GWAS_PASSTHROUGH = {
    "gwas_file": "--gwas_file",
    "gwas_folder": "--gwas_folder",
    "gwas_file_pattern": "--gwas_file_pattern",
    "snp_column": "--snp_column",
    "effect_allele_column": "--effect_allele_column",
    "non_effect_allele_column": "--non_effect_allele_column",
    "chromosome_column": "--chromosome_column",
    "position_column": "--position_column",
    "freq_column": "--freq_column",
    "beta_column": "--beta_column",
    "beta_sign_column": "--beta_sign_column",
    "or_column": "--or_column",
    "se_column": "--se_column",
    "zscore_column": "--zscore_column",
    "pvalue_column": "--pvalue_column",
    "separator": "--separator",
    "skip_until_header": "--skip_until_header",
    "snp_map_file": "--snp_map_file",
}

GWAS_FLAGS = {
    "keep_non_rsid": "--keep_non_rsid",
    "handle_empty_columns": "--handle_empty_columns",
}


@dataclass
class GwasOptions:
    """How to read the GWAS summary statistics."""

    gwas_file: Optional[str] = None
    gwas_folder: Optional[str] = None
    gwas_file_pattern: Optional[str] = None
    snp_column: str = "SNP"
    effect_allele_column: str = "A1"
    non_effect_allele_column: str = "A2"
    chromosome_column: Optional[str] = None
    position_column: Optional[str] = None
    freq_column: Optional[str] = None
    beta_column: Optional[str] = None
    beta_sign_column: Optional[str] = None
    or_column: Optional[str] = None
    se_column: Optional[str] = None
    zscore_column: Optional[str] = None
    pvalue_column: Optional[str] = None
    separator: Optional[str] = None
    skip_until_header: Optional[str] = None
    snp_map_file: Optional[str] = None
    keep_non_rsid: bool = False
    handle_empty_columns: bool = False

    def validate(self) -> None:
        if not self.gwas_file and not self.gwas_folder:
            raise ValueError("Provide --gwas-file or --gwas-folder.")
        if self.gwas_file and self.gwas_folder:
            raise ValueError("Provide only one of --gwas-file and --gwas-folder.")
        has_effect = any(
            (self.zscore_column, self.beta_column, self.or_column, self.beta_sign_column)
        )
        if not has_effect:
            raise ValueError(
                "The GWAS needs an effect column: pass at least one of "
                "--zscore-column, --beta-column, --or-column or --beta-sign-column."
            )
        if not self.zscore_column and not (self.se_column or self.pvalue_column):
            raise ValueError(
                "Without --zscore-column, S-PrediXcan needs --se-column or "
                "--pvalue-column to derive the SNP z-scores."
            )

    def to_args(self) -> list[str]:
        args: list[str] = []
        for attribute, flag in GWAS_PASSTHROUGH.items():
            value = getattr(self, attribute)
            if value is not None and value != "":
                args.extend([flag, str(value)])
        for attribute, flag in GWAS_FLAGS.items():
            if getattr(self, attribute):
                args.append(flag)
        return args


@dataclass
class SPrediXcanResult:
    output_path: Path
    command: list[str] = field(default_factory=list)


def run_sprediXcan(
    metaxcan_dir: Path,
    model_db_path: Path,
    covariance_path: Path,
    output_path: Path,
    gwas: GwasOptions,
    additional_output: bool = True,
    verbosity: int = 40,
    python_executable: Optional[str] = None,
) -> SPrediXcanResult:
    """Run S-PrediXcan and return where it wrote its results."""
    # Resolved because the subprocess runs with `metaxcan_dir` as its working
    # directory, which would otherwise re-anchor a relative path.
    metaxcan_dir = Path(metaxcan_dir).resolve()
    script = metaxcan_dir / SCRIPT_NAME
    if not script.exists():
        raise FileNotFoundError(
            f"{script} not found. Point --metaxcan-dir at metaxcan/software "
            "(is the submodule checked out?)."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        python_executable or sys.executable,
        str(script),
        "--model_db_path", str(Path(model_db_path).resolve()),
        "--covariance", str(Path(covariance_path).resolve()),
        "--output_file", str(output_path.resolve()),
        "--overwrite",
        "--throw",
        "--verbosity", str(verbosity),
    ]
    if additional_output:
        command.append("--additional_output")
    command.extend(gwas.to_args())

    logging.debug("Running: %s", " ".join(command))
    completed = subprocess.run(
        command,
        cwd=str(metaxcan_dir),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"S-PrediXcan failed for {Path(model_db_path).name} "
            f"(exit {completed.returncode}).\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    if not output_path.exists():
        raise RuntimeError(
            f"S-PrediXcan reported success but wrote no results for "
            f"{Path(model_db_path).name}. This usually means no model SNP matched "
            f"the GWAS.\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return SPrediXcanResult(output_path=output_path, command=command)


NUMERIC_COLUMNS = (
    "zscore",
    "effect_size",
    "pvalue",
    "var_g",
    "pred_perf_r2",
    "pred_perf_pval",
    "pred_perf_qval",
    "n_snps_used",
    "n_snps_in_cov",
    "n_snps_in_model",
    "best_gwas_p",
    "largest_weight",
)


def read_results(path: Path) -> pd.DataFrame:
    """
    Read a S-PrediXcan result CSV.

    `MetaxcanUtilities.format_output` runs `fillna("NA")` before writing, so
    every numeric column can come back as an object column of strings.
    """
    frame = pd.read_csv(path, na_values=["NA"], keep_default_na=True)
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


__all__ = [
    "GwasOptions",
    "SPrediXcanResult",
    "read_results",
    "run_sprediXcan",
]
