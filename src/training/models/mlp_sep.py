"""Independent per-cell-type predictors for cell type-specific expression.

This module keeps the public interface of ``models.mlp.MLPPredictor`` while
representing independent scalar-output models in a single PyTorch module for
convenient inference and checkpointing. ``SeparateModelPredictor`` owns the
shared per-cell-type bookkeeping (module list, indexing, concatenated
``forward``); ``MLPPredictor`` and ``CtPredPredictor`` only supply the
single-cell-type architecture to replicate per cell type.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from src.training.models.mlp import MLPPredictor as SingleMLPPredictor
from src.training.models.ctpred import CtPredPredictor as SingleCtPredPredictor


class SeparateModelPredictor(nn.Module):
    """A collection of independent, single-cell-type models.

    Parameters are intentionally not shared between cell types. ``forward``
    concatenates the scalar prediction from each model so callers see the
    same output shape as the shared multi-output MLP: ``(..., output_dim)``.

    Optimizers are deliberately created by the training entry point rather
    than stored on this module. ``cell_type_models`` exposes the ownership
    boundary needed to construct one optimizer per model.
    """

    def __init__(
        self,
        output_dim: int,
        single_model_factory: Callable[[], nn.Module],
        input_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if output_dim < 1:
            raise ValueError(f"output_dim must be positive, got {output_dim}.")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mlps = nn.ModuleList(
            [single_model_factory() for _ in range(output_dim)]
        )
        self.dropout_rate = dropout

    @property
    def cell_type_models(self) -> nn.ModuleList:
        """Return the independently optimized model for every cell type."""
        return self.mlps

    def forward_cell_type(
        self,
        x: torch.Tensor,
        cell_type_index: int,
    ) -> torch.Tensor:
        """Predict one cell type, retaining a final dimension of size one."""
        if not 0 <= cell_type_index < self.output_dim:
            raise IndexError(
                f"cell_type_index must be in [0, {self.output_dim}), "
                f"got {cell_type_index}."
            )
        return self.mlps[cell_type_index](x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                self.forward_cell_type(x, cell_type_index)
                for cell_type_index in range(self.output_dim)
            ],
            dim=-1,
        )


class MLPPredictor(SeparateModelPredictor):
    """Independent, single-cell-type MLPs with tapering hidden layer sizes."""

    def __init__(
        self,
        input_dim: int,
        n_layers: int,
        output_dim: int,
        layer_norm: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            output_dim=output_dim,
            input_dim=input_dim,
            dropout=dropout,
            single_model_factory=lambda: SingleMLPPredictor(
                input_dim=input_dim,
                n_layers=n_layers,
                output_dim=1,
                layer_norm=layer_norm,
                dropout=dropout,
            ),
        )


class CtPredPredictor(SeparateModelPredictor):
    """Independent, single-cell-type ctPred MLPs, as used in scPrediXcan.

    Reimplements ctPred (Zhang et al., scPrediXcan
    https://github.com/hakyimlab/scPrediXcan): a fixed-hidden-dimension MLP
    with ReLU/Dropout hidden blocks, trained separately per cell type, and
    ending in a ReLU + soft-upper-clip output activation.
    """

    def __init__(
        self,
        input_dim: int,
        n_layers: int,
        output_dim: int,
        hidden_dim: int = 64,
        layer_norm: bool = False,
        dropout: float = 0.05,
    ) -> None:
        super().__init__(
            output_dim=output_dim,
            input_dim=input_dim,
            dropout=dropout,
            single_model_factory=lambda: SingleCtPredPredictor(
                input_dim=input_dim,
                n_layers=n_layers,
                output_dim=1,
                hidden_dim=hidden_dim,
                layer_norm=layer_norm,
                dropout=dropout,
            ),
        )
        self.hidden_dim = hidden_dim


if __name__ == "__main__":
    model = MLPPredictor(input_dim=20, n_layers=3, output_dim=5)
    sample_input = torch.randn(4, 20)
    output = model(sample_input)
    print(f"MLP-sep output shape: {output.shape}")

    ctpred_model = CtPredPredictor(input_dim=20, n_layers=4, output_dim=5)
    output = ctpred_model(sample_input)
    print(f"CtPred output shape: {output.shape}")
