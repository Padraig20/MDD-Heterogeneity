""""Loss is taken from Seq2Cells paper."""

import torch
import torch.nn as nn
from typing import Literal

def log(t, eps=1e-20):
    """Custom log function clamped to minimum epsilon."""
    return torch.log(t.clamp(min=eps))

def poisson_loss(pred: torch.Tensor, target: torch.Tensor):
    """Poisson loss"""
    return (pred - target * log(pred)).mean()

def nonzero_median(tensor: torch.Tensor, axis: int, keepdim: bool) -> torch.Tensor:
    """Compute the median across non-zero float elements.

    Notes
    -----
    Modifies the tensor in place to avoid making a copy.
    """
    tensor = torch.where(tensor != 0.0, tensor.double(), float("nan"))

    # returns values and indices - we only want the value(s)
    medians = torch.nanmedian(tensor, dim=axis, keepdim=keepdim)[0]

    medians = medians.nan_to_num(0)

    return medians

class Seq2CellsLoss(nn.Module):
    """Loss function as defined in the Seq2Cells paper."""

    def __init__(
        self,
        rel_weight_gene: float = 1.0,
        rel_weight_cell: float = 1.0,
        norm_by: Literal["mean", "nonzero_median"] = "mean",
        eps: float = 1e-8,
    ):
        """Initialise Seq2CellsLoss.

        Parameter
        ---------
        rel_weight_gene: float = 1.0
            The relative weight to put on the across gene/tss correlation.
        rel_weight_cell: float = 1.0
            The relative weight to put on the across cells correlation.
        norm_by:  Literal['mean', 'nonzero_median'] = 'nonzero_median'
            What to use as across gene / cell average to subtract from the
            signal to normalise it. Mean or the Median of the non zero entries.
        eps: float 1e-8
            epsilon
        """
        super().__init__()
        self.eps = eps
        self.norm_by = norm_by
        self.rel_weight_gene = rel_weight_gene
        self.rel_weight_cell = rel_weight_cell
    
    def forward(self, predictions, targets):
        """
        Compute the Seq2Cells loss.
        
        Uses cosine-similarity-based loss to equally emphasize across-cell 
        and across-gene correlations, independent of cell count growth.

        Implementation adapted from:
        https://github.com/GSK-AI/seq2cells/blob/main/seq2cells/metrics_and_losses/losses.py

        Formula:
            loss = (1 - cos_sim_genes(p - p̄, t - t̄, ε)) + 
               (1 - cos_sim_cells(p - p̄, t - t̄, ε))
            p... predictions
            t... targets
            p̄... mean of predictions across genes/cells
            t̄... mean of targets across genes/cells
            ε... small constant to avoid division by zero (set to 1e-8)
        
        Args:
            predictions: Model predictions (p)
            targets: Target values (t)
        
        Returns:
            Computed loss value
        """
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
            )

        # Ensure float dtype
        p = predictions.float()
        t = targets.float()

        # ===== Across-genes term (correlation over genes) =====
        # Center across genes (last dim)
        p_g = p - p.mean(dim=-1, keepdim=True)
        t_g = t - t.mean(dim=-1, keepdim=True)

        # Flatten all but gene-dim: (..., cells, genes) -> (N, genes)
        p_g_flat = p_g.reshape(-1, p_g.shape[-1])
        t_g_flat = t_g.reshape(-1, t_g.shape[-1])

        dot_g = (p_g_flat * t_g_flat).sum(dim=-1)
        norm_p_g = p_g_flat.norm(dim=-1)
        norm_t_g = t_g_flat.norm(dim=-1)

        cos_genes = dot_g / (norm_p_g * norm_t_g + self.eps)
        cos_genes_mean = cos_genes.mean()

        # ===== Across-cells term (correlation over cells) =====
        # Center across cells (second-to-last dim)
        p_c = p - p.mean(dim=-2, keepdim=True)
        t_c = t - t.mean(dim=-2, keepdim=True)

        # Move cells dim to last: (..., cells, genes) -> (..., genes, cells)
        p_c = p_c.movedim(-2, -1)
        t_c = t_c.movedim(-2, -1)

        # Flatten all but cell-dim: (..., genes, cells) -> (M, cells)
        p_c_flat = p_c.reshape(-1, p_c.shape[-1])
        t_c_flat = t_c.reshape(-1, t_c.shape[-1])

        dot_c = (p_c_flat * t_c_flat).sum(dim=-1)
        norm_p_c = p_c_flat.norm(dim=-1)
        norm_t_c = t_c_flat.norm(dim=-1)

        cos_cells = dot_c / (norm_p_c * norm_t_c + self.eps)
        cos_cells_mean = cos_cells.mean()

        # Final loss
        loss = (1.0 - cos_genes_mean) + (1.0 - cos_cells_mean)
        return loss
    
if __name__ == "__main__":
    # example usage
    loss_fn = Seq2CellsLoss()
    preds = torch.randn(4, 10, 5) # (batch_size, n_cells, n_genes)
    targets = torch.randn(4, 10, 5)
    loss_value = loss_fn(preds, targets)
    print(f"Seq2Cells Loss: {loss_value.item()}")