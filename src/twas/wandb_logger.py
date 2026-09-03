from __future__ import annotations

import logging
import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

"""
wandb_logger.py

WandB logging for TWAS, one run per cell type.

Unlike `src/distillation/wandb_logger.py`, the project and entity come from the
CLI rather than module constants, and `wandb` is imported lazily so the whole
pipeline stays usable with `--wandb-project` omitted (that module raises at
import time when WANDB_KEY is unset).
"""

# Keep large per-gene tables from being uploaded in full.
MAX_TABLE_ROWS = 20_000


class TwasWandBLogger:
    """Logs one WandB run per cell type, named after the cell type."""

    def __init__(
        self,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        config: Optional[dict] = None,
    ) -> None:
        self.project = project
        self.entity = entity
        self.config = config or {}
        self.enabled = project is not None
        self._wandb = None
        self._run = None

        if not self.enabled:
            return

        import wandb

        self._wandb = wandb
        key = os.getenv("WANDB_KEY")
        if key:
            wandb.login(key=key)
        else:
            logging.info(
                "WANDB_KEY is not set; relying on an existing wandb login or "
                "WANDB_MODE for authentication."
            )

    def start(self, cell_type: str, config: Optional[dict] = None) -> None:
        """Open a fresh run for one cell type."""
        if not self.enabled:
            return
        run_config = {**self.config, **(config or {}), "cell_type": cell_type}
        self._run = self._wandb.init(
            project=self.project,
            entity=self.entity,
            name=cell_type,
            config=run_config,
            reinit=True,
        )

    def log_results(
        self,
        results: pd.DataFrame,
        summary: dict,
        figures: dict[str, Optional[plt.Figure]],
        tables: Optional[dict[str, pd.DataFrame]] = None,
    ) -> None:
        """Log one cell type's final table, scalar statistics and figures."""
        if not self.enabled or self._run is None:
            return

        payload: dict = dict(summary)
        payload["results_table"] = self._table(results)
        for name, table in (tables or {}).items():
            payload[name] = self._table(table)
        for name, figure in figures.items():
            if figure is not None:
                payload[name] = self._wandb.Image(figure)

        self._wandb.log(payload)
        self._run.summary.update(summary)

    def _table(self, frame: pd.DataFrame):
        if len(frame) > MAX_TABLE_ROWS:
            logging.info(
                "Truncating a %d-row table to the %d most significant rows for WandB.",
                len(frame), MAX_TABLE_ROWS,
            )
            frame = (
                frame.nsmallest(MAX_TABLE_ROWS, "pvalue")
                if "pvalue" in frame.columns
                else frame.head(MAX_TABLE_ROWS)
            )
        # WandB cannot serialise pandas' nullable/extension dtypes reliably.
        return self._wandb.Table(dataframe=frame.reset_index(drop=True).astype(object).where(frame.notna(), None))

    def finish(self) -> None:
        """Close the current cell type's run."""
        if not self.enabled or self._run is None:
            return
        self._wandb.finish()
        self._run = None


__all__ = ["TwasWandBLogger"]
