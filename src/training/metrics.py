"""
Custom Metrics to get the Pearson Correlation Coefficient per cell and per gene.
Idea adapted from seq2cells:
https://github.com/GSK-AI/seq2cells/blob/main/seq2cells/metrics_and_losses/metrics.py

Furthermore, we keep track of the mean predicted uncertainty. We also keep track of
the Expected Normalized Calibration Error (ENCE) and Spearman rank correlation between
predicted uncertainty and observed squared error. We also produce a boxplot of
MSE vs predicted uncertainty quantiles.
Idea(s) adapted from:
"Evaluating and Calibrating Uncertainty Prediction in Regression Tasks" (2020)
"""

import matplotlib
matplotlib.use("Agg")  # headless backend; we only render figures to log them
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import Tensor
from torchmetrics import Metric


def _pearson_from_sums(
    sum_xy: Tensor,
    sum_x: Tensor,
    sum_y: Tensor,
    sum_x2: Tensor,
    sum_y2: Tensor,
    n: float | Tensor,
) -> Tensor:
    """Compute Pearson correlations from float64 sufficient statistics.

    The textbook one-pass variance expression can be slightly negative due to
    cancellation when values have very little spread.  Mathematically these
    variances are non-negative, so clamp round-off below zero and mark
    genuinely constant/non-finite inputs as undefined.
    """
    cov = sum_xy - (sum_x * sum_y) / n
    var_x = (sum_x2 - (sum_x * sum_x) / n).clamp_min(0.0)
    var_y = (sum_y2 - (sum_y * sum_y) / n).clamp_min(0.0)
    denom = torch.sqrt(var_x * var_y)

    valid = (
        denom.gt(0.0)
        & torch.isfinite(denom)
        & torch.isfinite(cov)
    )
    correlation = cov / denom
    correlation = correlation.clamp(min=-1.0, max=1.0)
    return torch.where(valid, correlation, torch.nan)


class MeanCellPearson(Metric):
    """Pearson per cell (correlate across genes), then mean over cells."""

    is_differentiable = False
    higher_is_better  = True
    full_state_update = False

    def __init__(self, n_cells: int, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # Float32 raw moments lose too much precision when predictions or
        # targets are nearly constant (common for percentile targets).
        zeros = torch.zeros(n_cells, dtype=torch.float64)
        self.add_state("sum_xy", default=zeros.clone(), dist_reduce_fx="sum")
        self.add_state("sum_x",  default=zeros.clone(), dist_reduce_fx="sum")
        self.add_state("sum_y",  default=zeros.clone(), dist_reduce_fx="sum")
        self.add_state("sum_x2", default=zeros.clone(), dist_reduce_fx="sum")
        self.add_state("sum_y2", default=zeros.clone(), dist_reduce_fx="sum")
        self.add_state("n",      default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")

    def update(self, pred: Tensor, target: Tensor) -> None:
        # pred/target: [B, n_cells]
        assert pred.shape == target.shape

        pred = pred.to(dtype=torch.float64)
        target = target.to(dtype=torch.float64)
        self.sum_xy += torch.sum(pred * target, dim=0)
        self.sum_x  += torch.sum(pred, dim=0)
        self.sum_y  += torch.sum(target, dim=0)
        self.sum_x2 += torch.sum(pred * pred, dim=0)
        self.sum_y2 += torch.sum(target * target, dim=0)
        self.n      += pred.shape[0]

    def compute(self) -> Tensor:
        assert self.n > 0.0, "No samples to compute metric!"

        return _pearson_from_sums(
            self.sum_xy,
            self.sum_x,
            self.sum_y,
            self.sum_x2,
            self.sum_y2,
            self.n,
        )

class MeanGenePearson(Metric):
    """Pearson per gene (across cells), then mean over genes."""
    is_differentiable = False
    higher_is_better  = True
    full_state_update = False

    def __init__(self, n_genes: int, n_cells: int, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.n_cells = float(n_cells)
        self.n_genes = n_genes

        zeros = torch.zeros(n_genes, dtype=torch.float64)
        self.add_state("sum_xy", default=zeros.clone(), dist_reduce_fx="sum")
        self.add_state("sum_x",  default=zeros.clone(), dist_reduce_fx="sum")
        self.add_state("sum_y",  default=zeros.clone(), dist_reduce_fx="sum")
        self.add_state("sum_x2", default=zeros.clone(), dist_reduce_fx="sum")
        self.add_state("sum_y2", default=zeros.clone(), dist_reduce_fx="sum")
        self.add_state("n",      default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")

    def update(self, pred: Tensor, target: Tensor) -> None:
        # pred/target: [B, n_cells]
        assert pred.shape == target.shape
        
        B = pred.shape[0]

        start = int(self.n.item())
        end   = min(start + B, self.n_genes)

        pred = pred.to(dtype=torch.float64)
        target = target.to(dtype=torch.float64)
        # Each row contains all cells for one gene, so it can be centered
        # directly.  This avoids subtracting two nearly equal raw moments.
        pred = pred - pred.mean(dim=1, keepdim=True)
        target = target - target.mean(dim=1, keepdim=True)
        self.sum_xy[start:end] += (pred * target).sum(dim=1)
        self.sum_x2[start:end] += (pred * pred).sum(dim=1)
        self.sum_y2[start:end] += (target * target).sum(dim=1)

        self.n += B

    def compute(self) -> Tensor:
        n = int(self.n.item())
        assert n > 0, "No samples to compute metric!"

        sum_x  = self.sum_x[:n]
        sum_y  = self.sum_y[:n]
        sum_xy = self.sum_xy[:n]
        sum_x2 = self.sum_x2[:n]
        sum_y2 = self.sum_y2[:n]

        nc = self.n_cells  # num cells per gene

        return _pearson_from_sums(
            sum_xy,
            sum_x,
            sum_y,
            sum_x2,
            sum_y2,
            nc,
        )


def _spearman(a: Tensor, b: Tensor) -> float:
    """Spearman rank correlation == Pearson correlation computed on ranks."""
    # ordinal ranks via double argsort (ties broken arbitrarily, fine for continuous data)
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = ra.norm() * rb.norm()
    if denom <= 0:
        return float("nan")
    return float((ra @ rb) / denom)


class UncertaintyCalibration:
    """
    Accumulates per-element predicted uncertainty and squared error during evaluation,
    then computes calibration metrics and a MSE-vs-uncertainty boxplot.

    Uncertainty is the total predictive variance, and error is the squared error.
    A perfectly calibrated regressor's predicted variance would match the observed
    squared error in expectation.
    """

    def __init__(self, n_ence_bins: int = 15, n_box_bins: int = 6, max_samples: int = 5_000_000):
        self.n_ence_bins = n_ence_bins
        self.n_box_bins = n_box_bins
        self.max_samples = max_samples  # cap to keep memory bounded
        self._unc_chunks: list[Tensor] = []
        self._err_chunks: list[Tensor] = []
        self._n = 0

    def update(self, prediction: Tensor, total_uncertainty: Tensor, targets: Tensor) -> None:
        """All tensors shaped [B, n_cells]; flattened and stored on CPU."""
        if self._n >= self.max_samples:
            return
        unc = total_uncertainty.detach().reshape(-1).float().cpu()
        err = (prediction.detach() - targets.detach()).reshape(-1).float().cpu() ** 2
        self._unc_chunks.append(unc)
        self._err_chunks.append(err)
        self._n += unc.numel()

    def _gather(self) -> tuple[Tensor, Tensor]:
        unc = torch.cat(self._unc_chunks)
        err = torch.cat(self._err_chunks)
        # drop non-finite entries (e.g. nan targets) to keep metrics well-defined
        mask = torch.isfinite(unc) & torch.isfinite(err)
        return unc[mask], err[mask]

    def compute_ence(self) -> float:
        unc, err = self._gather()
        n = unc.numel()
        if n < self.n_ence_bins:
            return float("nan")

        order = torch.argsort(unc)
        unc_s, err_s = unc[order], err[order]

        ence = 0.0
        valid_bins = 0
        for idx in torch.chunk(torch.arange(n), self.n_ence_bins):
            if idx.numel() == 0:
                continue
            rmv = torch.sqrt(unc_s[idx].mean().clamp_min(1e-12))
            rmse = torch.sqrt(err_s[idx].mean().clamp_min(0.0))
            ence += float((rmv - rmse).abs() / rmv.clamp_min(1e-12))
            valid_bins += 1
        return ence / max(valid_bins, 1)

    def compute_spearman(self) -> float:
        unc, err = self._gather()
        if unc.numel() < 2:
            return float("nan")
        return _spearman(unc, err)

    def make_boxplot(self, title: str = "Uncertainty"):
        """Boxplot of squared error (MSE per element) across equal-count (quantile) bins of total predictive uncertainty."""
        unc, err = self._gather()
        n = unc.numel()
        if n < self.n_box_bins:
            return None

        order = torch.argsort(unc)
        err_s = err[order].numpy()

        bins = np.array_split(err_s, self.n_box_bins)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.boxplot(
            bins,
            positions=range(self.n_box_bins),
            widths=0.6,
            showfliers=False,
            patch_artist=True,
            boxprops=dict(facecolor="#b9d9e0", edgecolor="#5a7d8c"),
            medianprops=dict(color="#5a7d8c"),
            whiskerprops=dict(color="#5a7d8c"),
            capprops=dict(color="#5a7d8c"),
        )
        ax.set_title(title)
        ax.set_xlabel("Total Uncertainty (quantile bin)")
        ax.set_ylabel("MSE")
        ax.set_xticks(range(self.n_box_bins))
        ax.set_xticklabels(range(self.n_box_bins))
        fig.tight_layout()
        return fig


def _numpy_pearson(first: np.ndarray, second: np.ndarray) -> float:
    """Return Pearson's r for finite, nonconstant one-dimensional arrays."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    finite = np.isfinite(first) & np.isfinite(second)
    if finite.sum() < 2:
        return float("nan")
    first = first[finite]
    second = second[finite]
    first -= first.mean()
    second -= second.mean()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    return float(np.clip((first @ second) / denominator, -1.0, 1.0))


class PopulationVarianceEvaluation:
    """Evaluate aleatoric variance against empirical donor variance.

    This diagnostic is intended for unseen-gene evaluation of a
    ``ReferencePopulationMddDataset``.  The empirical target must have shape
    ``[n_genes, n_cell_types]`` and be expressed on the same target scale as
    the model.  Predictions can be supplied one evaluation batch at a time.

    The model's stored post-hoc temperature calibrates *total* predictive
    variance.  We consequently report magnitude metrics for both the deployed
    (scaled) aleatoric variance and its uncalibrated value.  Pearson correlation
    is invariant to that positive scalar, so it is reported only once.
    """

    def __init__(
        self,
        target_variance: np.ndarray,
        cell_types: list[str],
        donor_counts: np.ndarray,
        variance_scale: float,
    ) -> None:
        target = np.asarray(target_variance, dtype=np.float64)
        if target.ndim != 2:
            raise ValueError(
                "target_variance must have shape [n_genes, n_cell_types]."
            )
        if target.shape[1] != len(cell_types):
            raise ValueError(
                f"Target has {target.shape[1]} cell types but received "
                f"{len(cell_types)} labels."
            )
        counts = np.asarray(donor_counts, dtype=np.int64)
        if counts.shape != (len(cell_types),):
            raise ValueError(
                "donor_counts must contain one value per cell type."
            )
        if not np.isfinite(variance_scale) or variance_scale <= 0.0:
            raise ValueError("variance_scale must be finite and positive.")

        self.target_variance = target
        self.cell_types = [str(cell_type) for cell_type in cell_types]
        self.donor_counts = counts
        self.variance_scale = float(variance_scale)
        self._prediction_chunks: list[np.ndarray] = []

    def update(self, aleatoric_variance: Tensor) -> None:
        prediction = (
            aleatoric_variance.detach().to(dtype=torch.float64).cpu().numpy()
        )
        if prediction.ndim != 2 or prediction.shape[1] != len(self.cell_types):
            raise ValueError(
                "aleatoric_variance must have shape [batch, n_cell_types]."
            )
        self._prediction_chunks.append(prediction)

    def _predictions(self) -> np.ndarray:
        if not self._prediction_chunks:
            raise RuntimeError("No aleatoric predictions were accumulated.")
        prediction = np.concatenate(self._prediction_chunks, axis=0)
        if prediction.shape != self.target_variance.shape:
            raise RuntimeError(
                "Aleatoric prediction/target shape mismatch: "
                f"{prediction.shape} versus {self.target_variance.shape}. "
                "Population-variance evaluation requires a sequential loader "
                "with exactly one population-mean row per unseen gene."
            )
        return prediction

    @staticmethod
    def _magnitude_metrics(
        prediction: np.ndarray,
        target: np.ndarray,
    ) -> dict[str, float]:
        finite = (
            np.isfinite(prediction)
            & np.isfinite(target)
            & (prediction >= 0.0)
            & (target >= 0.0)
        )
        prediction = prediction[finite]
        target = target[finite]
        if prediction.size == 0:
            return {
                "variance_ratio": float("nan"),
                "variance_r2": float("nan"),
                "std_rmse": float("nan"),
                "std_nrmse": float("nan"),
            }

        target_sum = float(target.sum())
        variance_ratio = (
            float(prediction.sum() / target_sum)
            if target_sum > 0.0
            else float("nan")
        )
        centered_target = target - target.mean()
        total_sum_squares = float(centered_target @ centered_target)
        residual = prediction - target
        variance_r2 = (
            1.0 - float(residual @ residual) / total_sum_squares
            if total_sum_squares > 0.0
            else float("nan")
        )
        prediction_std = np.sqrt(prediction)
        target_std = np.sqrt(target)
        std_rmse = float(
            np.sqrt(np.mean(np.square(prediction_std - target_std)))
        )
        mean_target_std = float(target_std.mean())
        std_nrmse = (
            std_rmse / mean_target_std
            if mean_target_std > 0.0
            else float("nan")
        )
        return {
            "variance_ratio": variance_ratio,
            "variance_r2": variance_r2,
            "std_rmse": std_rmse,
            "std_nrmse": std_nrmse,
        }

    def compute(self) -> tuple[dict[str, float], dict[str, float]]:
        prediction = self._predictions()
        target = self.target_variance
        per_cell_type = {
            cell_type: _numpy_pearson(
                prediction[:, index],
                target[:, index],
            )
            for index, cell_type in enumerate(self.cell_types)
        }
        finite_correlations = np.asarray(
            [value for value in per_cell_type.values() if np.isfinite(value)],
            dtype=np.float64,
        )
        calibrated = self._magnitude_metrics(prediction, target)
        uncalibrated = self._magnitude_metrics(
            prediction / self.variance_scale,
            target,
        )
        usable_donor_counts = self.donor_counts[self.donor_counts >= 2]
        scalars = {
            "pearson_macro": (
                float(finite_correlations.mean())
                if finite_correlations.size
                else float("nan")
            ),
            "pearson_pooled": _numpy_pearson(prediction.ravel(), target.ravel()),
            "n_genes": float(target.shape[0]),
            "n_gene_cell_type_pairs": float(
                np.count_nonzero(np.isfinite(prediction) & np.isfinite(target))
            ),
            "donors_min": (
                float(usable_donor_counts.min())
                if usable_donor_counts.size
                else float("nan")
            ),
            "donors_max": (
                float(usable_donor_counts.max())
                if usable_donor_counts.size
                else float("nan")
            ),
            **calibrated,
            **{
                f"uncalibrated_{name}": value
                for name, value in uncalibrated.items()
            },
        }
        return scalars, per_cell_type

    def make_figure(self, title: str = "Population aleatoric variance"):
        prediction = self._predictions()
        target = self.target_variance
        finite = (
            np.isfinite(prediction)
            & np.isfinite(target)
            & (prediction >= 0.0)
            & (target >= 0.0)
        )
        if finite.sum() < 2:
            return None

        predicted_std = np.sqrt(prediction[finite])
        target_std = np.sqrt(target[finite])
        positive = np.concatenate(
            [predicted_std[predicted_std > 0.0], target_std[target_std > 0.0]]
        )
        floor = (
            max(float(positive.min()) * 0.1, np.finfo(np.float64).tiny)
            if positive.size
            else np.finfo(np.float64).tiny
        )
        x = np.log10(np.maximum(target_std, floor))
        y = np.log10(np.maximum(predicted_std, floor))
        max_points = 100_000
        if x.size > max_points:
            indices = np.linspace(0, x.size - 1, max_points, dtype=np.int64)
            x = x[indices]
            y = y[indices]

        _, per_cell_type = self.compute()
        figure_height = max(5.0, 0.3 * len(self.cell_types) + 1.8)
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(12.0, figure_height),
            gridspec_kw={"width_ratios": (1.1, 1.0)},
        )
        axes[0].hexbin(x, y, gridsize=55, mincnt=1, cmap="viridis")
        lower = float(min(x.min(), y.min()))
        upper = float(max(x.max(), y.max()))
        axes[0].plot([lower, upper], [lower, upper], "--", color="0.4", linewidth=1)
        axes[0].set_xlabel("log10 empirical across-donor SD")
        axes[0].set_ylabel("log10 predicted aleatoric SD")
        axes[0].set_title("Magnitude calibration")

        correlations = np.asarray(
            [per_cell_type[cell_type] for cell_type in self.cell_types],
            dtype=np.float64,
        )
        y_positions = np.arange(len(self.cell_types))
        colors = np.where(correlations >= 0.0, "#4C78A8", "#F58518")
        axes[1].barh(y_positions, correlations, color=colors)
        axes[1].axvline(0.0, color="0.4", linewidth=0.8)
        axes[1].set_yticks(y_positions)
        axes[1].set_yticklabels(self.cell_types, fontsize=8)
        axes[1].invert_yaxis()
        axes[1].set_xlim(-1.0, 1.0)
        axes[1].set_xlabel("Pearson r across unseen genes")
        axes[1].set_title("Per-cell-type ranking")
        axes[1].grid(axis="x", alpha=0.2)

        figure.suptitle(title)
        figure.tight_layout()
        return figure
