import wandb
import os
import matplotlib.pyplot as plt
from typing import Optional
import numpy as np
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

        # plot R2 vs rank (i.e. sorted by R2 ascending). For the *illustration* we cap
        # R² at 0: a handful of badly-fit genes have hugely negative R².
        # The logged mean_r2 / median_r2 scalars below remain the true, uncapped values.
        r2_capped = df["r2"].clip(lower=0.0)
        mean_r2_capped   = float(r2_capped.mean())
        median_r2_capped = float(r2_capped.median())

        # in-sample R² (train fold of the same held-out split), for the overfitting gap
        has_insample  = "insample_r2" in df.columns
        mean_insample = float(df["insample_r2"].mean()) if has_insample else float("nan")

        fig_r2, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(df["rank"], r2_capped, s=4, alpha=0.5)
        ax.set_title(f"{cell_type} — {model_label} held-out R² (per-gene 20% individuals)")
        ax.set_xlabel("Gene rank (sorted by R² ascending)")
        ax.set_ylabel("Held-out R² (capped at 0 for display)")
        annot = f"mean_R² = {mean_r2_capped:.3f}\nmedian_R² = {median_r2_capped:.3f}"
        if has_insample:
            annot += f"\nmean in-sample_R² = {mean_insample:.3f}"
        ax.text(0.03, 0.95, annot, transform=ax.transAxes, va="top")
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

        # Held-out split bookkeeping: the overfitting gap (in-sample minus held-out)
        # and how many individuals / genes actually got a held-out fold.
        if has_insample:
            log_data[f"{cell_type}/mean_insample_r2"]   = mean_insample
            log_data[f"{cell_type}/mean_overfit_gap"]   = mean_insample - mean_r2
        if "n_test" in df.columns:
            log_data[f"{cell_type}/mean_n_train"]       = float(df["n_train"].mean())
            log_data[f"{cell_type}/mean_n_test"]        = float(df["n_test"].mean())
            log_data[f"{cell_type}/n_genes_no_holdout"] = int((df["n_test"] == 0).sum())

        if "pearson_r" in df.columns:
            log_data[f"{cell_type}/mean_pearson_r"] = float(df["pearson_r"].mean())
            log_data[f"{cell_type}/median_pearson_r"] = float(df["pearson_r"].median())

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

        plt.close(fig_r2)
        plt.close(fig_nnz)
        if fig_across is not None:
            plt.close(fig_across)
        if fig_within is not None:
            plt.close(fig_within)

    @staticmethod
    def _pearson(a: pd.Series, b: pd.Series) -> float:
        """NaN-robust Pearson correlation without numpy divide-by-zero warnings."""
        m = a.notna() & b.notna()
        if int(m.sum()) < 2:
            return float("nan")
        av = a[m].to_numpy(dtype=float); bv = b[m].to_numpy(dtype=float)
        av = av - av.mean(); bv = bv - bv.mean()
        denom = float(np.sqrt((av @ av) * (bv @ bv)))
        return float((av @ bv) / denom) if denom > 1e-12 else float("nan")

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
        across_gene_corr = self._pearson(df["pred_std"], df["target_std"])
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

        across_corr = self._pearson(sub["pred_std"], sub["target_std"])

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