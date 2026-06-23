"""MSE loss"""

import torch
import torch.nn as nn

class MSELoss(nn.Module):
    """MSE Loss"""

    def __init__(
        self
    ):
        super().__init__()
    
    def forward(self, predictions, targets):
        """
        Compute the MSE loss.
        
        Args:
            predictions: Model predictions (p)
            targets: Target values (t)
        
        Returns:
            Computed loss value: mse_loss
        """

        if predictions.shape[-1] == 2: # (mu, var) from deep ensemble
            predictions = predictions[..., 0]

        if predictions.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
            )

        p = predictions.float()
        t = targets.float()

        # ----- MSE -----
        mse_loss = torch.mean((p - t) ** 2)

        return mse_loss

if __name__ == "__main__":
    # example usage
    loss_fn = MSELoss()
    preds = torch.randn(4, 10, 5)
    targets = torch.randn(4, 10, 5)
    loss_value = loss_fn(preds, targets)
    print(f"MSE Loss: {loss_value.item()}")