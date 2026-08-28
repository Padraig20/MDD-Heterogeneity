"""ctPred: the single-cell-type expression predictor from scPrediXcan.

Reimplements ``ctPred`` from the scPrediXcan framework (Zhang et al.):
https://github.com/hakyimlab/scPrediXcan

Unlike ``models.mlp.MLPPredictor`` (which tapers the hidden layer sizes from
``input_dim`` down to ``output_dim``), ctPred uses a fixed hidden dimension
across all of its hidden layers and a distinctive output activation
(ReLU followed by a soft upper clip), matching the reference implementation
in ``Scripts/ctPred/ctPred_utils.py`` of the scPrediXcan repository.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CtPredMLP(nn.Module):
    """Fixed-hidden-dimension MLP with the ctPred output activation."""

    def __init__(
        self,
        input_dim: int,
        n_layers: int,
        output_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1 for ctPred, got {n_layers}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        # input_dim -> hidden_dim, then repeatedly apply one shared
        # hidden_dim -> hidden_dim block.  The reference implementation creates
        # `hidden_layer` once and inserts those same module objects at every
        # hidden position, so its repeated Linear transformations share weights.
        # Preserve that behavior here for an exact architectural match.
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        hidden_layer = [
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        for _ in range(n_layers - 1):
            layers.extend(hidden_layer)
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = F.relu(x)
        # Soft upper clip: saturates towards 1 for large x while staying
        # non-negative, matching targets that are rank/percentile-normalized
        # into [0, 1] (see the `--norm-targets percentiles` option).
        return x / (1.0 + F.softplus(x - 1.0))


class CtPredPredictor(nn.Module):
    """Wraps ``CtPredMLP`` with the same public interface as ``mlp.MLPPredictor``."""

    def __init__(
        self,
        input_dim: int,
        n_layers: int,
        output_dim: int,
        hidden_dim: int = 64,
        layer_norm: bool = False,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.mlp = CtPredMLP(
            input_dim,
            n_layers,
            output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.layer_norm = nn.LayerNorm(input_dim) if layer_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layer_norm:
            x = self.layer_norm(x)
        return self.mlp(x)


if __name__ == "__main__":
    model = CtPredPredictor(input_dim=5313, n_layers=4, output_dim=1)
    sample_input = torch.randn(4, 5313)
    output = model(sample_input)
    print(f"Output shape: {output.shape}")
