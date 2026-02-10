"""
Gaussian NLL Loss (as in 'Reliable Neural Networks for Regression Uncertainty Estimation').

We use this loss function only for uncertainty estimation! The model outputs both a mean and
a variance, and the loss encourages the model to learn a mean that fits the data, and a variance
that reflects the uncertainty in the predictions.

Implements the maximum-likelihood NLL for a Gaussian with *heteroscedastic* variance:
  L = mean( (y - mu)^2 / (2 * sigma2) + 0.5 * log(sigma2) )
"""

import torch
import torch.nn as nn

class GaussianNLLLoss(nn.Module):
    """Gaussian Negative Log-Likelihood for regression with learned mean and variance."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, predictions, targets):
        if predictions.shape[:-1] != targets.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}. "
                f"Expected predictions[..., 2] and targets[...]"
            )
        if predictions.shape[-1] != 2:
            raise ValueError(
                f"Expected predictions last dim == 2 (mu, raw_var), got {predictions.shape[-1]}"
            )

        preds = predictions.float()
        t     = targets.float()

        mu      = preds[..., 0]
        raw_var = preds[..., 1]

        sigma2 = torch.sigmoid(raw_var).clamp_min(self.eps) # bound variance to be positive and > 0
        resid2 = (t - mu) ** 2

        # negative log-likelihood up to additive constant:
        # 0.5*log(sigma2) + 0.5*resid^2/sigma2  (+ const)
        nll = 0.5 * torch.log(sigma2) + 0.5 * resid2 / sigma2

        return torch.mean(nll)

if __name__ == "__main__":
    # example usage
    loss_fn = GaussianNLLLoss()
    preds = torch.randn(4, 10, 2) # raw outputs
    targets = torch.randn(4, 10)
    loss_value = loss_fn(preds, targets)
    print(f"Gaussian NLL Loss: {loss_value.item():.6f}")
