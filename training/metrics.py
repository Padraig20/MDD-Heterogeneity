"""
Custom Metrics to get the Pearson Correlation Coefficient per cell and per gene.
Idea adapted from seq2cells:
https://github.com/GSK-AI/seq2cells/blob/main/seq2cells/metrics_and_losses/metrics.py
"""

import torch
from torch import Tensor
from torchmetrics import Metric

class MeanCellPearson(Metric):
    """Pearson per cell (correlate across genes), then mean over cells."""

    is_differentiable = False
    higher_is_better  = True
    full_state_update = False

    def __init__(self, n_cells: int, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        zeros = torch.zeros(n_cells, dtype=torch.float32)
        self.add_state("sum_xy", default=zeros.clone(), dist_reduce_fx="sum")
        self.add_state("sum_x",  default=zeros.clone(), dist_reduce_fx="sum")
        self.add_state("sum_y",  default=zeros.clone(), dist_reduce_fx="sum")
        self.add_state("sum_x2", default=zeros.clone(), dist_reduce_fx="sum")
        self.add_state("sum_y2", default=zeros.clone(), dist_reduce_fx="sum")
        self.add_state("n",      default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, pred: Tensor, target: Tensor) -> None:
        # pred/target: [B, n_cells]
        assert pred.shape == target.shape

        self.sum_xy += torch.sum(pred * target, dim=0)
        self.sum_x  += torch.sum(pred, dim=0)
        self.sum_y  += torch.sum(target, dim=0)
        self.sum_x2 += torch.sum(pred * pred, dim=0)
        self.sum_y2 += torch.sum(target * target, dim=0)
        self.n      += pred.shape[0]

    def compute(self) -> Tensor:
        assert self.n > 0.0, "No samples to compute metric!"

        cov   = self.sum_xy - (self.sum_x * self.sum_y) / self.n
        var_x = self.sum_x2 - (self.sum_x * self.sum_x) / self.n
        var_y = self.sum_y2 - (self.sum_y * self.sum_y) / self.n

        assert var_x.ge(0).all() and var_y.ge(0).all(), "Negative variance encountered in Pearson computation!"

        denom = torch.sqrt(var_x) * torch.sqrt(var_y)

        assert denom.ge(0).all(), "Non-positive denominator encountered in Pearson computation!"

        return cov / (denom + 1e-12)  # [n]; account for possible 0 division

class MeanGenePearson(Metric):
    """Pearson per gene (across cells), then mean over genes."""
    is_differentiable = False
    higher_is_better  = True
    full_state_update = False

    def __init__(self, n_genes: int, n_cells: int, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.n_cells = float(n_cells)
        self.n_genes = n_genes

        zeros = torch.zeros(n_genes, dtype=torch.float32)
        self.add_state("sum_xy", default=zeros.clone(), dist_reduce_fx="sum")
        self.add_state("sum_x",  default=zeros.clone(), dist_reduce_fx="sum")
        self.add_state("sum_y",  default=zeros.clone(), dist_reduce_fx="sum")
        self.add_state("sum_x2", default=zeros.clone(), dist_reduce_fx="sum")
        self.add_state("sum_y2", default=zeros.clone(), dist_reduce_fx="sum")
        self.add_state("n",      default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, pred: Tensor, target: Tensor) -> None:
        # pred/target: [B, n_cells]
        assert pred.shape == target.shape
        
        B = pred.shape[0]

        start = int(self.n.item())
        end   = min(start + B, self.n_genes)

        self.sum_x[start:end]  += pred.sum(dim=1)
        self.sum_y[start:end]  += target.sum(dim=1)
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

        cov   = sum_xy - (sum_x * sum_y) / nc
        var_x = sum_x2 - (sum_x * sum_x) / nc
        var_y = sum_y2 - (sum_y * sum_y) / nc
        
        assert var_x.ge(0).all() and var_y.ge(0).all(), "Negative variance encountered in Pearson computation!"

        denom = torch.sqrt(var_x) * torch.sqrt(var_y)

        assert denom.ge(0).all(), "Non-positive denominator encountered in Pearson computation!"

        return cov / (denom + 1e-12) # [n]; account for possible 0 division
