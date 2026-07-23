import wandb
import os
import matplotlib.pyplot as plt
from typing import Optional
import numpy as np
import pandas as pd

from src.distillation.utils import pearson_pvalue, safe_pearson

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

            # Two-sided p-value of the held-out Pearson r, from the per-gene held-out
            # sample size (n_test). Added as a column so it shows up in the summary
            # table too; NaN wherever pearson_r or n_test is unavailable/undefined.
            if "n_test" in df.columns:
                df["pearson_pvalue"] = pearson_pvalue(df["pearson_r"].to_numpy(), df["n_test"].to_numpy())
            else:
                df["pearson_pvalue"] = float("nan")

            pvalue_vals = df["pearson_pvalue"].dropna().to_numpy(dtype=float)
            has_pearson_pvalue = pvalue_vals.size > 0
            if has_pearson_pvalue:
                mean_pearson_pvalue   = float(np.mean(pvalue_vals))
                median_pearson_pvalue = float(np.median(pvalue_vals))
                n_sig_pearson_pvalue  = int(np.sum(pvalue_vals < 0.05))

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
        fig_pearson_pvalue = None
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

            if has_pearson_pvalue:
                fig_pearson_pvalue = self._pearson_pvalue_fig(pvalue_vals, cell_type, model_label)

        # Spearman correlation of mean predictions vs targets (per gene), logged as a
        # histogram matching the style used for teacher-prediction diagnostics: x-axis
        # Spearman r, y-axis number of genes, with mean/median annotated on the figure.
        # Per-gene values already live in `df["spearman_r"]` (from summarize_models) and
        # therefore also appear in the summary table below.
        has_spearman = "spearman_r" in df.columns
        fig_spearman = None
        n_dropped_spearman = 0
        mean_spearman_r = float("nan")
        median_spearman_r = float("nan")
        has_insample_spearman = False
        mean_insample_spearman = float("nan")
        if has_spearman:
            mean_spearman_r   = float(df["spearman_r"].mean())
            median_spearman_r = float(df["spearman_r"].median())
            has_insample_spearman = "insample_spearman_r" in df.columns
            mean_insample_spearman = (
                float(df["insample_spearman_r"].mean()) if has_insample_spearman else float("nan")
            )
            spearman_vals = df["spearman_r"].dropna().to_numpy(dtype=float)
            n_dropped_spearman = len(df) - spearman_vals.size
            if spearman_vals.size > 0:
                fig_spearman = self._spearman_hist_fig(
                    spearman_vals, cell_type, model_label,
                    mean_r=mean_spearman_r, median_r=median_spearman_r,
                    mean_insample_r=mean_insample_spearman if has_insample_spearman else None,
                    n_dropped=n_dropped_spearman,
                )

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

            # Two-sided significance test of the held-out Pearson r (t-test on r, n_test).
            if has_pearson_pvalue:
                log_data[f"{cell_type}/mean_pearson_pvalue"]          = mean_pearson_pvalue
                log_data[f"{cell_type}/median_pearson_pvalue"]        = median_pearson_pvalue
                log_data[f"{cell_type}/n_genes_pearson_pvalue_lt_0.05"] = n_sig_pearson_pvalue
                log_data[f"{cell_type}/frac_genes_pearson_pvalue_lt_0.05"] = (
                    n_sig_pearson_pvalue / pvalue_vals.size
                )
                if fig_pearson_pvalue is not None:
                    log_data[f"{cell_type}/pearson_pvalue_plot"] = wandb.Image(fig_pearson_pvalue)

        if has_spearman:
            log_data[f"{cell_type}/mean_spearman_r"]   = mean_spearman_r
            log_data[f"{cell_type}/median_spearman_r"] = median_spearman_r
            log_data[f"{cell_type}/n_genes_undefined_spearman_r"] = n_dropped_spearman
            if has_insample_spearman:
                log_data[f"{cell_type}/mean_insample_spearman_r"]  = mean_insample_spearman
                log_data[f"{cell_type}/mean_spearman_overfit_gap"] = (
                    mean_insample_spearman - mean_spearman_r
                )
            if fig_spearman is not None:
                log_data[f"{cell_type}/spearman_plot"] = wandb.Image(fig_spearman)

        # We produce two separate std-correlation plots per uncertainty source (they
        # measure different things):
        #   * across-gene: per-gene mean predicted vs target std (calibration scatter)
        #   * within-gene: distribution of the per-gene across-individual correlations
        # "total" is the combined (aleatoric + epistemic) predictive std, reported
        # under the original (unprefixed) keys for backwards compatibility. When the
        # model distills aleatoric/epistemic as separate targets (ProbabilisticLR),
        # the same two plots + metrics are additionally reported for each source
        # individually (under an `{kind}_`-prefixed key), so each can be judged on
        # its own rather than only via their sum.
        variance_figs = []
        if "std_w2" in df.columns:
            log_data.update(
                self._variance_diagnostics(
                    df, cell_type, model_label, kind="total",
                    pred_col="pred_std", target_col="target_std", w2_col="std_w2",
                    corr_col="std_corr", ratio_col="std_ratio", wasserstein_col="wasserstein",
                )
            )
            fig = self._across_gene_std_fig(
                df, cell_type, model_label,
                pred_col="pred_std", target_col="target_std", w2_col="std_w2",
            )
            if fig is not None:
                log_data[f"{cell_type}/across_gene_std_plot"] = wandb.Image(fig)
                variance_figs.append(fig)
            fig = self._within_gene_std_fig(df, cell_type, model_label, corr_col="std_corr")
            if fig is not None:
                log_data[f"{cell_type}/within_gene_std_plot"] = wandb.Image(fig)
                variance_figs.append(fig)

            for kind in ("aleatoric", "epistemic"):
                pred_col, target_col = f"{kind}_pred_std", f"{kind}_target_std"
                w2_col, corr_col, ratio_col = f"{kind}_w2", f"{kind}_std_corr", f"{kind}_std_ratio"
                if pred_col not in df.columns or target_col not in df.columns:
                    continue
                log_data.update(
                    self._variance_diagnostics(
                        df, cell_type, model_label, kind=kind,
                        pred_col=pred_col, target_col=target_col, w2_col=w2_col,
                        corr_col=corr_col, ratio_col=ratio_col,
                    )
                )
                fig = self._across_gene_std_fig(
                    df, cell_type, model_label,
                    pred_col=pred_col, target_col=target_col, w2_col=w2_col, label=kind,
                )
                if fig is not None:
                    log_data[f"{cell_type}/across_gene_std_plot_{kind}"] = wandb.Image(fig)
                    variance_figs.append(fig)
                fig = self._within_gene_std_fig(df, cell_type, model_label, corr_col=corr_col, label=kind)
                if fig is not None:
                    log_data[f"{cell_type}/within_gene_std_plot_{kind}"] = wandb.Image(fig)
                    variance_figs.append(fig)

        if step is not None:
            wandb.log(log_data, step=step)
        else:
            wandb.log(log_data)

        for fig in (fig_r2, fig_nnz_r2, fig_pearson, fig_nnz_pearson, fig_pearson_pvalue, fig_spearman, *variance_figs):
            if fig is not None:
                plt.close(fig)

    def _spearman_hist_fig(
        self,
        values: np.ndarray,
        cell_type: str,
        model_label: str,
        mean_r: float,
        median_r: float,
        mean_insample_r: Optional[float] = None,
        n_dropped: int = 0,
        n_bins: int = 50,
    ):
        """
        Histogram of the per-gene held-out Spearman correlation between predicted and
        target mean expression. Style matches the teacher-prediction Spearman
        diagnostic (x = Spearman r, y = number of genes), with mean/median annotated
        on the figure.
        """
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None

        # Cover the full Spearman range (predictions can anti-correlate); bins of
        # width ~0.04 on [-1, 1] (~50 bins) keep the histogram readable at both
        # the dense high-correlation peak and the long low-correlation tail.
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(vals, bins=n_bins, range=(-1.0, 1.0), edgecolor="white", color="0.45")
        ax.set_xlim(-1.0, 1.0)
        ax.set_title(f"{cell_type} — {model_label} held-out Spearman r (per-gene)")
        ax.set_xlabel("Spearman correlation")
        ax.set_ylabel("number of genes")
        annot = f"mean = {mean_r:.3f}\nmedian = {median_r:.3f}\nn_genes = {vals.size}"
        if mean_insample_r is not None and np.isfinite(mean_insample_r):
            annot += f"\nmean in-sample = {mean_insample_r:.3f}"
        if n_dropped:
            annot += f"\n({n_dropped} gene(s) with undefined r excluded)"
        ax.text(0.03, 0.97, annot, transform=ax.transAxes, va="top")
        fig.tight_layout()
        return fig

    def _pearson_pvalue_fig(
        self,
        pvalues: np.ndarray,
        cell_type: str,
        model_label: str,
        n_bins: int = 20,
    ):
        """
        Histogram of the per-gene p-values from the two-sided significance test of
        the held-out Pearson r (t-test on r and n_test). Under the null of no true
        correlation, this should look roughly uniform on [0, 1]; an excess of small
        p-values indicates genes with a genuinely predictive model.
        """
        vals = np.asarray(pvalues, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None

        n = vals.size
        n_sig = int(np.sum(vals < 0.05))
        pct_sig = 100.0 * n_sig / n

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(vals, bins=n_bins, range=(0.0, 1.0), edgecolor="white")
        ax.set_xlim(0.0, 1.0)
        ax.set_title(f"{cell_type} — {model_label} held-out Pearson r p-value (per-gene)")
        ax.set_xlabel("p-value")
        ax.set_ylabel("Number of genes")
        ax.text(
            0.97, 0.95,
            f"{n:,} genes\np < 0.05: {n_sig} ({pct_sig:.1f}%)",
            transform=ax.transAxes,
            ha="right", va="top",
        )
        fig.tight_layout()
        return fig

    def _variance_diagnostics(
        self,
        df: pd.DataFrame,
        cell_type: str,
        model_label: str,
        kind: str,
        pred_col: str,
        target_col: str,
        w2_col: str,
        corr_col: str,
        ratio_col: str,
        wasserstein_col: Optional[str] = None,
    ) -> dict:
        """
        Scalar metrics summarising how well a predicted uncertainty source matches the
        teacher's target. `kind` is one of "total" (the combined aleatoric + epistemic
        predictive std -- reported under the original, unprefixed keys for backwards
        compatibility) or "aleatoric"/"epistemic" (reported under an `{kind}_`-prefixed
        key, when the model distills each as a separate target).

        Two very different correlations are reported, because they answer different
        questions:
          - {corr_col} (within-gene): corr(pred std, target std) *across individuals*,
            averaged over genes. Measures whether the model tracks which individuals are
            more/less certain for a given gene.
          - across_gene_std_corr: corr of the *per-gene mean* pred vs target std across
            genes.
        Plus:
          - {w2_col}:    variance-matching error for this source (lower is better)
          - {ratio_col}: mean(pred std) / mean(target std) (calibration; ~1 is ideal)
        Aggregated across genes (NaNs from failed fits are skipped).
        """
        prefix = f"{cell_type}/" if kind == "total" else f"{cell_type}/{kind}_"
        across_gene_corr = safe_pearson(df[pred_col], df[target_col])
        out = {
            f"{prefix}mean_std_w2": float(df[w2_col].mean()),
            f"{prefix}median_std_w2": float(df[w2_col].median()),
            f"{prefix}mean_within_gene_std_corr": float(df[corr_col].mean()),
            f"{prefix}across_gene_std_corr": across_gene_corr,
            f"{prefix}mean_std_ratio": float(df[ratio_col].mean()),
            f"{prefix}median_std_ratio": float(df[ratio_col].median()),
        }
        if wasserstein_col is not None:
            out[f"{prefix}mean_wasserstein"] = float(df[wasserstein_col].mean())
            out[f"{prefix}median_wasserstein"] = float(df[wasserstein_col].median())
        if kind == "total":
            diverged = int(df["diverged"].sum()) if "diverged" in df.columns else 0
            out[f"{prefix}n_diverged"] = diverged
            out[f"{prefix}n_genes"] = int(len(df))
        return out

    def _across_gene_std_fig(
        self,
        df: pd.DataFrame,
        cell_type: str,
        model_label: str,
        pred_col: str = "pred_std",
        target_col: str = "target_std",
        w2_col: str = "std_w2",
        label: str = "",
    ):
        """
        Across-gene std correlation: per-gene mean predicted vs target std (one point
        per gene) for one uncertainty source (`label` is "" for the combined total, or
        "aleatoric"/"epistemic" for the individual heads). This is the calibration
        scatter; its Pearson r is `across_gene_std_corr`.
        """
        sub = df[[pred_col, target_col]].dropna()
        if sub.empty:
            return None

        across_corr = safe_pearson(sub[pred_col], sub[target_col])
        qualifier = f"{label} " if label else ""

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(sub[target_col], sub[pred_col], s=4, alpha=0.4)
        lo = float(min(sub[target_col].min(), sub[pred_col].min()))
        hi = float(max(sub[target_col].max(), sub[pred_col].max()))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x (perfect)")
        ax.set_title(f"{cell_type} — {model_label} across-gene {qualifier}std correlation")
        ax.set_xlabel(f"Teacher target {qualifier}std (per-gene mean, native log-expr space)")
        ax.set_ylabel(f"Predicted {qualifier}std (per-gene mean, native log-expr space)")
        ax.legend(loc="upper left")
        mean_w2 = float(df[w2_col].mean())
        ax.text(
            0.03, 0.90,
            f"across-gene r = {across_corr:.3f}\n"
            f"across-gene r² = {across_corr ** 2:.3f}\n"
            f"mean_{qualifier}std_W² = {mean_w2:.3f}",
            transform=ax.transAxes,
            va="top",
        )
        fig.tight_layout()
        return fig

    def _within_gene_std_fig(
        self,
        df: pd.DataFrame,
        cell_type: str,
        model_label: str,
        corr_col: str = "std_corr",
        label: str = "",
    ):
        """
        Within-gene std correlation: distribution of the per-gene Pearson r between
        predicted and target std *across individuals* (one r per gene), for one
        uncertainty source (`label` is "" for the combined total, or
        "aleatoric"/"epistemic" for the individual heads).
        """
        if corr_col not in df.columns:
            return None
        vals = df[corr_col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None

        mean_r   = float(np.mean(vals))
        median_r = float(np.median(vals))
        qualifier = f"{label} " if label else ""

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.boxplot(
            vals,
            vert=True,
            showmeans=True,
            meanline=True,
            widths=0.5,
            tick_labels=["all genes"],
            flierprops=dict(marker=".", markersize=3, alpha=0.3),
        )
        ax.axhline(0.0, color="grey", linestyle=":", linewidth=1)
        ax.set_ylim(-1, 1)
        ax.set_title(f"{cell_type} — {model_label} within-gene {qualifier}std correlation")
        ax.set_ylabel(f"Per-gene Pearson r of {qualifier}std across individuals")
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
