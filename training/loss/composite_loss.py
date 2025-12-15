"""Composite loss made of Pearson Correlation, Poisson NLL, and MSE. Idea adapted from UNICORN paper."""

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

class CompositeLoss(nn.Module):
    """Loss function similarly defined in the UNICORN paper."""

    def __init__(
        self,
        lambda_mpc: float = 1.0,
        lambda_pnll: float = 1.0,
        lambda_mse: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.lambda_mpc = lambda_mpc
        self.lambda_pnll = lambda_pnll
        self.lambda_mse = lambda_mse
        self.eps = eps
    
    def forward(self, predictions, targets):
        """
        Compute the Composite loss.
        
        Essentially just a weighted combination of Pearson correlation,
        Poisson negative log likelihood and MSE.

        Implemented what is described in:
        https://www.nature.com/articles/s41467-025-64506-8

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

        # ----- Poisson NLL -----
        p_pos = p.clamp_min(self.eps)
        poisson_nll = torch.mean(p_pos - t * torch.log(p_pos))

        # ----- MSE -----
        mse_loss = torch.mean((p - t) ** 2)

        # ----- Total -----
        loss = (
            self.lambda_mpc * L_mpc
            + self.lambda_pnll * poisson_nll
            + self.lambda_mse * mse_loss
        )
        return loss

if __name__ == "__main__":
    # example usage
    loss_fn = CompositeLoss()
    preds = torch.randn(4, 10, 5).abs()  # positive for Poisson
    targets = torch.randn(4, 10, 5)
    loss_value = loss_fn(preds, targets)
    print(f"Composite Loss: {loss_value.item()}")