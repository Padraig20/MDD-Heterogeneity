"""Loss idea taken from Likelihood Annealing paper."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


class LikaLoss(nn.Module):
    """Likelihood Annealing loss for heteroscedastic regression.

    Expects predictions to provide both (all positive):
      - mu: mean prediction
      - sigma: predicted scale (std)

    Formula:
      NLL(mu, sigma; y) + T2 * (|mu - y|²) + T3 * (|sigma - |mu - y||²)
    """

    def __init__(
        self,
        eps: float = 1e-8,
        T2: float = 100.0, # start with "high" temperature values
        T3: float = 100.0,
        decay: float = 0.9,
    ):
        """
        Parameters
        ----------
        eps : float
            Numerical stability epsilon.
        T2, T3 : float
            Likelihood Annealing temperatures (anneal toward 0 during training).
        decay : float
            Decay rate for temperatures per epoch.
        """
        super().__init__()
        self.eps = eps
        self.T2 = float(T2)
        self.T3 = float(T3)
        self.decay = decay

    def set_temperatures(self) -> None:
        """Update temperatures."""
        self.T2 = self.T2 * self.decay
        self.T3 = self.T3 * self.decay

    def forward(
        self,
        predictions_mu: torch.Tensor,
        predictions_sigma: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        mu, sigma = predictions_mu, predictions_sigma

        res = mu - targets
        abs_res = res.abs()

        # Gaussian NLL
        nll = torch.log(sigma) + 0.5 * (res / sigma)**2

        # LikA regularizers
        reg2 = abs_res**2
        reg3 = (sigma - abs_res)**2

        loss = nll + self.T2 * reg2 + self.T3 * reg3

        return loss.mean()


if __name__ == "__main__":
    # example usage
    loss_fn = LikaLoss()
    preds = torch.randn(4, 10, 5) # (batch_size, n_cells, n_genes)
    stds  = torch.rand(4, 10, 5) + 0.1 # ensure positive std
    targets = torch.randn(4, 10, 5)
    loss_value = loss_fn(preds, stds, targets)
    print(f"LikA Loss: {loss_value.item()}")
    loss_value.backward()