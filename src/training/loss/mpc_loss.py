"""Mutual Pearson Correlation Loss."""

import torch
import torch.nn as nn

def pearson_corr_dim(x: torch.Tensor, y: torch.Tensor, dim: int = -1, eps: float = 1e-8):
    """
    Pearson correlation along a given dimension.

    Returns a tensor with that dimension reduced (one correlation per slice).
    """
    x_centered = x - x.mean(dim=dim, keepdim=True)
    y_centered = y - y.mean(dim=dim, keepdim=True)

    num    = (x_centered * y_centered).sum(dim=dim)
    x_norm = torch.linalg.norm(x_centered, dim=dim)
    y_norm = torch.linalg.norm(y_centered, dim=dim)
    denom  = (x_norm * y_norm).clamp_min(eps)

    return num / denom

class MPCLoss(nn.Module):
    """Mutual Pearson Correlation Loss."""

    def __init__(
        self,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.eps = eps
    
    def forward(self, predictions, targets):
        """
        Compute the Mutual Pearson Correlation loss.

        Args:
            predictions: Model predictions (p)
            targets: Target values (t)
        
        Returns:
            Computed loss value: L_mpc
        """
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
            )

        p = predictions.float()
        t = targets.float()

        # ----- Mutual Pearson correlation (MPC) -----
        # p, t: (..., n_cells, n_genes)

        # across genes: Pearson over gene-dim, mean over cell(types)
        g_corr_per_cell = pearson_corr_dim(t, p, dim=-1, eps=self.eps)  # shape: (..., n_cells)
        g_corr = g_corr_per_cell.mean()

        # across cells: Pearson over cell(type)-dim, mean over genes
        c_corr_per_gene = pearson_corr_dim(t, p, dim=-2, eps=self.eps)  # shape: (..., n_genes)
        c_corr = c_corr_per_gene.mean()

        L_mpc = -(g_corr + c_corr)

        return L_mpc

if __name__ == "__main__":
    # example usage
    loss_fn = MPCLoss()
    preds = torch.randn(4, 10, 5)
    targets = torch.randn(4, 10, 5)
    loss_value = loss_fn(preds, targets)
    print(f"Mutual Pearson Correlation Loss: {loss_value.item()}")