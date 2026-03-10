import wandb
import os
import matplotlib.pyplot as plt
from typing import Optional
import pandas as pd

ENTITY  = "your-entity"
PROJECT = "your-project"

if "WANDB_KEY" not in os.environ:
    raise EnvironmentError("You should *really* set the WANDB_KEY environment variable!!!")

wandb.login(key=os.getenv("WANDB_KEY"))
class WandBLogger:

    def __init__(self, enabled=True, 
                 run_name: str=None) -> None:
        
        self.enabled = enabled

        if self.enabled:
            wandb.init(entity=ENTITY,
                       project=PROJECT)
            if run_name is None:
                wandb.run.name = wandb.run.id    
            else:
                wandb.run.name = run_name  
            
    def log_celltype_diagnostics(
        self,
        df: pd.DataFrame,
        cell_type: str,
        step: Optional[int] = None,
    ) -> None:
        
        if not self.enabled:
            return

        # statistics
        mean_r2    = float(df["r2"].mean())
        median_r2  = float(df["r2"].median())
        mean_nnz   = float(df["nonzero_weights"].mean())
        median_nnz = float(df["nonzero_weights"].median())

        # plot R2 vs rank (i.e. sorted by R2 ascending)
        fig_r2, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(df["rank"], df["r2"], s=4, alpha=0.5)
        ax.set_title(f"{cell_type} — Elastic Net R²")
        ax.set_xlabel("Gene rank (sorted by R² ascending)")
        ax.set_ylabel("R²")
        ax.text(
            0.03, 0.95,
            f"mean_R² = {mean_r2:.3f}\nmedian_R² = {median_r2:.3f}",
            transform=ax.transAxes,
            va="top"
        )
        fig_r2.tight_layout()

        # plot nonzero weights vs same rank
        # is model complexity (nonzero weights) correlated with performance (R2)?
        fig_nnz, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(df["rank"], df["nonzero_weights"], s=4, alpha=0.5)
        ax.set_title(f"{cell_type} — Nonzero weights")
        ax.set_xlabel("Gene rank (sorted by R² ascending)")
        ax.set_ylabel("Number of nonzero weights")
        ax.text(
            0.03, 0.95,
            f"mean_nnz = {mean_nnz:.2f}\nmedian_nnz = {median_nnz:.2f}",
            transform=ax.transAxes,
            va="top"
        )
        fig_nnz.tight_layout()

        # table for wandb inspection
        table = wandb.Table(dataframe=df)

        log_data = {
            f"{cell_type}/summary_table": table,
            f"{cell_type}/r2_plot": wandb.Image(fig_r2),
            f"{cell_type}/nonzero_plot": wandb.Image(fig_nnz),
            f"{cell_type}/mean_r2": mean_r2,
            f"{cell_type}/median_r2": median_r2,
            f"{cell_type}/mean_nonzero_weights": mean_nnz,
            f"{cell_type}/median_nonzero_weights": median_nnz,
        }

        if step is not None:
            wandb.log(log_data, step=step)
        else:
            wandb.log(log_data)

        plt.close(fig_r2)
        plt.close(fig_nnz)
 
    def finish(self):
        if self.enabled:
            wandb.finish()