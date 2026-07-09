import wandb
import os
import matplotlib.pyplot as plt
from typing import Optional
import numpy as np
import pandas as pd

from src.distillation.utils import safe_pearson

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
        model_label: str = "Elastic Net",
    ) -> None:
        
        if not self.enabled:
            return

        # Guard against an empty summary (e.g. every gene for this cell type failed to
        # fit): there are no columns to read
        if df.empty or "r2" not in df.columns:
            wandb.log({f"{cell_type}/n_genes": int(len(df))})
            return

        # statistics. .mean()/.median() skip NaNs, so diverged/failed per-gene fits
        # (marked NaN upstream) don't pollute the aggregates or the plots.
        mean_r2    = float(df["r2"].mean())
        median_r2  = float(df["r2"].median())
        mean_nnz   = float(df["nonzero_weights"].mean())
        median_nnz = float(df["nonzero_weights"].median())

        # in-sample (train fold of the same held-out split), for the overfitting gap
        has_insample  = "insample_r2" in df.columns
        mean_insample = float(df["insample_r2"].mean()) if has_insample else float("nan")

        has_pearson = "pearson_r" in df.columns
        if has_pearson:
            mean_pearson_r   = float(df["pearson_r"].mean())
            median_pearson_r = float(df["pearson_r"].median())
            has_insample_pearson  = "insample_pearson_r" in df.columns
            mean_insample_pearson = (
                float(df["insample_pearson_r"].mean()) if has_insample_pearson else float("nan")
            )

        # We report R² and Pearson r as two fully independent pairs of plots
        df_by_r2 = df.dropna(subset=["r2"])
        df_by_r2 = df_by_r2[df_by_r2["r2"] >= 0].sort_values("r2", ascending=True).reset_index(drop=True)
        n_dropped_r2 = len(df) - len(df_by_r2)

        fig_r2 = None
        fig_nnz_r2 = None
        if not df_by_r2.empty:
            rank_r2 = np.arange(1, len(df_by_r2) + 1)

            fig_r2, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(rank_r2, df_by_r2["r2"], s=4, alpha=0.5)
            ax.set_title(f"{cell_type} — {model_label} held-out R² (per-gene)")
            ax.set_xlabel("Gene rank (sorted by R² ascending)")
            ax.set_ylabel("Held-out R²")
            annot = f"mean_R² = {mean_r2:.3f}\nmedian_R² = {median_r2:.3f}"
            if has_insample:
                annot += f"\nmean in-sample_R² = {mean_insample:.3f}"
            if n_dropped_r2:
                annot += f"\n({n_dropped_r2} gene(s) with undefined/negative R² excluded)"
            ax.text(0.03, 0.95, annot, transform=ax.transAxes, va="top")
            fig_r2.tight_layout()

            mean_nnz_r2   = float(df_by_r2["nonzero_weights"].mean())
            median_nnz_r2 = float(df_by_r2["nonzero_weights"].median())
            fig_nnz_r2, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(rank_r2, df_by_r2["nonzero_weights"], s=4, alpha=0.5)
            ax.set_title(f"{cell_type} — Nonzero weights (ranked by R²)")
            ax.set_xlabel("Gene rank (sorted by R² ascending)")
            ax.set_ylabel("Number of nonzero weights")
            ax.text(
                0.03, 0.95,
                f"mean_nnz = {mean_nnz_r2:.2f}\nmedian_nnz = {median_nnz_r2:.2f}",
                transform=ax.transAxes,
                va="top",
            )
            fig_nnz_r2.tight_layout()

        fig_pearson = None
        fig_nnz_pearson = None
        n_dropped_pearson = 0
        if has_pearson:
            df_by_pearson = df.dropna(subset=["pearson_r"]).sort_values("pearson_r", ascending=True).reset_index(drop=True)
            n_dropped_pearson = len(df) - len(df_by_pearson)

            if not df_by_pearson.empty:
                rank_pearson = np.arange(1, len(df_by_pearson) + 1)

                fig_pearson, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(rank_pearson, df_by_pearson["pearson_r"], s=4, alpha=0.5)
                ax.axhline(0.0, color="grey", linestyle=":", linewidth=1)
                ax.set_ylim(-1.05, 1.05)
                ax.set_title(f"{cell_type} — {model_label} held-out Pearson r (per-gene)")
                ax.set_xlabel("Gene rank (sorted by Pearson r ascending)")
                ax.set_ylabel("Held-out Pearson r")
                annot = f"mean_r = {mean_pearson_r:.3f}\nmedian_r = {median_pearson_r:.3f}"
                if has_insample_pearson:
                    annot += f"\nmean in-sample_r = {mean_insample_pearson:.3f}"
                if n_dropped_pearson:
                    annot += f"\n({n_dropped_pearson} gene(s) with undefined r excluded)"
                ax.text(0.03, 0.95, annot, transform=ax.transAxes, va="top")
                fig_pearson.tight_layout()

                mean_nnz_pearson   = float(df_by_pearson["nonzero_weights"].mean())
                median_nnz_pearson = float(df_by_pearson["nonzero_weights"].median())
                fig_nnz_pearson, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(rank_pearson, df_by_pearson["nonzero_weights"], s=4, alpha=0.5)
                ax.set_title(f"{cell_type} — Nonzero weights (ranked by Pearson r)")
                ax.set_xlabel("Gene rank (sorted by Pearson r ascending)")
                ax.set_ylabel("Number of nonzero weights")
                ax.text(
                    0.03, 0.95,
                    f"mean_nnz = {mean_nnz_pearson:.2f}\nmedian_nnz = {median_nnz_pearson:.2f}",
                    transform=ax.transAxes,
                    va="top",
                )
                fig_nnz_pearson.tight_layout()

        # table for wandb inspection
        table = wandb.Table(dataframe=df)

        log_data = {
            f"{cell_type}/summary_table": table,
            f"{cell_type}/mean_r2": mean_r2,
            f"{cell_type}/median_r2": median_r2,
            f"{cell_type}/mean_nonzero_weights": mean_nnz,
            f"{cell_type}/median_nonzero_weights": median_nnz,
            f"{cell_type}/n_genes_excluded_r2": n_dropped_r2,
        }
        if fig_r2 is not None:
            log_data[f"{cell_type}/r2_plot"] = wandb.Image(fig_r2)
        if fig_nnz_r2 is not None:
            log_data[f"{cell_type}/nonzero_plot_by_r2"] = wandb.Image(fig_nnz_r2)

        # Held-out split bookkeeping: the overfitting gap (in-sample minus held-out)
        # and how many individuals / genes actually got a held-out fold.
        if has_insample:
            log_data[f"{cell_type}/mean_insample_r2"]   = mean_insample
            log_data[f"{cell_type}/mean_overfit_gap"]   = mean_insample - mean_r2
        if "n_test" in df.columns:
            log_data[f"{cell_type}/mean_n_train"]       = float(df["n_train"].mean())
            log_data[f"{cell_type}/mean_n_test"]        = float(df["n_test"].mean())
            log_data[f"{cell_type}/n_genes_no_holdout"] = int((df["n_test"] == 0).sum())

        if has_pearson:
            log_data[f"{cell_type}/mean_pearson_r"]            = mean_pearson_r
            log_data[f"{cell_type}/median_pearson_r"]          = median_pearson_r
            log_data[f"{cell_type}/n_genes_undefined_pearson_r"] = n_dropped_pearson
            if has_insample_pearson:
                log_data[f"{cell_type}/mean_insample_pearson_r"]  = mean_insample_pearson
                log_data[f"{cell_type}/mean_pearson_overfit_gap"] = mean_insample_pearson - mean_pearson_r
            if fig_pearson is not None:
                log_data[f"{cell_type}/pearson_plot"] = wandb.Image(fig_pearson)
            if fig_nnz_pearson is not None:
                log_data[f"{cell_type}/nonzero_plot_by_pearson"] = wandb.Image(fig_nnz_pearson)

        # We produce two separate std-correlation plots (they measure different things):
        #   * across-gene: per-gene mean predicted vs target std (calibration scatter)
        #   * within-gene: distribution of the per-gene across-individual correlations
        fig_across = None
        fig_within = None
        if "std_w2" in df.columns:
            log_data.update(
                self._variance_diagnostics(df, cell_type, model_label)
            )
            fig_across = self._across_gene_std_fig(df, cell_type, model_label)
            if fig_across is not None:
                log_data[f"{cell_type}/across_gene_std_plot"] = wandb.Image(fig_across)
            fig_within = self._within_gene_std_fig(df, cell_type, model_label)
            if fig_within is not None:
                log_data[f"{cell_type}/within_gene_std_plot"] = wandb.Image(fig_within)

        if step is not None:
            wandb.log(log_data, step=step)
        else:
            wandb.log(log_data)

        for fig in (fig_r2, fig_nnz_r2, fig_pearson, fig_nnz_pearson, fig_across, fig_within):
            if fig is not None:
                plt.close(fig)

    def _variance_diagnostics(self, df: pd.DataFrame, cell_type: str, model_label: str) -> dict:
        """
        Scalar metrics summarising how well the predicted variances match the teacher's
        target variances. Two very different correlations are reported, because they
        answer different questions:
          - std_corr (within-gene): corr(pred std, target std) *across individuals*,
            averaged over genes. Measures whether the model tracks which individuals are
            more/less certain for a given gene.
          - across_gene_std_corr: corr of the *per-gene mean* pred vs target std across
            genes.
        Plus:
          - std_w2:    variance-matching part of the Wasserstein distance (lower is better)
          - std_ratio: mean(pred std) / mean(target std) (calibration; ~1 is ideal)
        Aggregated across genes (NaNs from failed fits are skipped).
        """
        diverged = int(df["diverged"].sum()) if "diverged" in df.columns else 0
        across_gene_corr = safe_pearson(df["pred_std"], df["target_std"])
        return {
            f"{cell_type}/mean_std_w2": float(df["std_w2"].mean()),
            f"{cell_type}/median_std_w2": float(df["std_w2"].median()),
            f"{cell_type}/mean_wasserstein": float(df["wasserstein"].mean()),
            f"{cell_type}/median_wasserstein": float(df["wasserstein"].median()),
            f"{cell_type}/mean_within_gene_std_corr": float(df["std_corr"].mean()),
            f"{cell_type}/across_gene_std_corr": across_gene_corr,
            f"{cell_type}/mean_std_ratio": float(df["std_ratio"].mean()),
            f"{cell_type}/median_std_ratio": float(df["std_ratio"].median()),
            f"{cell_type}/n_diverged": diverged,
            f"{cell_type}/n_genes": int(len(df)),
        }

    def _across_gene_std_fig(self, df: pd.DataFrame, cell_type: str, model_label: str):
        """
        Across-gene std correlation: per-gene mean predicted vs target std (one point
        per gene). This is the calibration scatter; its Pearson r is `across_gene_std_corr`.
        """
        sub = df[["pred_std", "target_std"]].dropna()
        if sub.empty:
            return None

        across_corr = safe_pearson(sub["pred_std"], sub["target_std"])

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(sub["target_std"], sub["pred_std"], s=4, alpha=0.4)
        lo = float(min(sub["target_std"].min(), sub["pred_std"].min()))
        hi = float(max(sub["target_std"].max(), sub["pred_std"].max()))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x (perfect)")
        ax.set_title(f"{cell_type} — {model_label} across-gene std correlation")
        ax.set_xlabel("Teacher target std (per-gene mean, native log-expr space)")
        ax.set_ylabel("Predicted std (per-gene mean, native log-expr space)")
        ax.legend(loc="upper left")
        mean_w2 = float(df["std_w2"].mean())
        ax.text(
            0.03, 0.90,
            f"across-gene r = {across_corr:.3f}\n"
            f"across-gene r² = {across_corr ** 2:.3f}\n"
            f"mean_std_W² = {mean_w2:.3f}",
            transform=ax.transAxes,
            va="top",
        )
        fig.tight_layout()
        return fig

    def _within_gene_std_fig(self, df: pd.DataFrame, cell_type: str, model_label: str):
        """
        Within-gene std correlation: distribution of the per-gene Pearson r between
        predicted and target std *across individuals* (one r per gene).
        """
        if "std_corr" not in df.columns:
            return None
        vals = df["std_corr"].dropna().to_numpy(dtype=float)
        if vals.size == 0:
            return None

        mean_r   = float(np.mean(vals))
        median_r = float(np.median(vals))

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.boxplot(
            vals,
            vert=True,
            showmeans=True,
            meanline=True,
            widths=0.5,
            flierprops=dict(marker=".", markersize=3, alpha=0.3),
        )
        ax.axhline(0.0, color="grey", linestyle=":", linewidth=1)
        ax.set_ylim(-1, 1)
        ax.set_xticks([1])
        ax.set_xticklabels(["all genes"])
        ax.set_title(f"{cell_type} — {model_label} within-gene std correlation")
        ax.set_ylabel("Per-gene Pearson r of std across individuals")
        ax.text(
            0.03, 0.97,
            f"mean r = {mean_r:.3f}\nmedian r = {median_r:.3f}\nn_genes = {vals.size}",
            transform=ax.transAxes,
            va="top",
        )
        fig.tight_layout()
        return fig
 
    def finish(self):
        if self.enabled:
            wandb.finish()