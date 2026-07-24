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
