"""Poisson NLL Loss."""

import torch
import torch.nn as nn

class PNLLLoss(nn.Module):
    """Poisson Negative Log Likelihood Loss."""

    def __init__(
        self,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.eps = eps
    
    def forward(self, predictions, targets):
        """
        Compute the PNLL loss.

        Args:
            predictions: Model predictions (p)
            targets: Target values (t)
        
        Returns:
            Computed loss value: poisson_nll
        """

        if predictions.shape[-1] == 2: # (mu, var) from deep ensemble
            predictions = predictions[..., 0]

        if predictions.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
            )

        p = predictions.float()
        t = targets.float()

        # ----- Poisson NLL -----
        p_pos = p.clamp_min(self.eps)
        poisson_nll = torch.mean(p_pos - t * torch.log(p_pos))
        
        return poisson_nll

if __name__ == "__main__":
    # example usage
    loss_fn = PNLLLoss()
    preds = torch.randn(4, 10, 5).abs()  # positive for Poisson
    targets = torch.randn(4, 10, 5)
    loss_value = loss_fn(preds, targets)
    print(f"Poisson NLL Loss: {loss_value.item()}")