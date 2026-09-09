"""Independent per-cell-type deep ensembles.

Each cell type owns a full ``MLPEnsemble`` with ``output_dim=1``. Parameters,
member training, and post-hoc variance scales are not shared across cell types.
Combined evaluation concatenates the per-cell-type aggregates so callers see
the same ``(prediction, aleatoric, epistemic)`` shapes as the shared ensemble.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.training.models.mlp_deep_ensemble import MLPEnsemble


class SeparateMLPEnsemble(nn.Module):
    """One deep ensemble per cell type.

    Training must go through ``forward_cell_type`` so each ensemble can be
    optimized independently. Combined ``forward`` is evaluation-only.
    """

    def __init__(
        self,
        n_models: int,
        input_dim: int,
        n_layers: int,
        output_dim: int,
        layer_norm: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if output_dim < 1:
            raise ValueError(f"output_dim must be positive, got {output_dim}.")
        if n_models < 1:
            raise ValueError(f"n_models must be positive, got {n_models}.")

        self.n_models = n_models
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout
        self.mlps = nn.ModuleList(
            [
                MLPEnsemble(
                    n_models=n_models,
                    input_dim=input_dim,
                    n_layers=n_layers,
                    output_dim=1,
                    layer_norm=layer_norm,
                    dropout=dropout,
                )
                for _ in range(output_dim)
            ]
        )

    @property
    def cell_type_models(self) -> nn.ModuleList:
        """Return the independently optimized ensemble for every cell type."""
        return self.mlps

    @property
    def variance_scale(self) -> torch.Tensor:
        """Mean of the per-cell-type post-hoc variance scales."""
        return torch.stack(
            [ensemble.variance_scale for ensemble in self.mlps]
        ).mean()

    @property
    def variance_scales(self) -> list[float]:
        """Post-hoc variance scale fitted for each cell-type ensemble."""
        return [
            float(ensemble.variance_scale.item()) for ensemble in self.mlps
        ]

    def set_variance_scale(self, scale: float) -> None:
        """Broadcast one variance scale to every cell-type ensemble."""
        for ensemble in self.mlps:
            ensemble.set_variance_scale(scale)

    def _check_cell_type_index(self, cell_type_index: int) -> None:
        if not 0 <= cell_type_index < self.output_dim:
            raise IndexError(
                f"cell_type_index must be in [0, {self.output_dim}), "
                f"got {cell_type_index}."
            )

    def forward_cell_type(
        self,
        x: torch.Tensor,
        cell_type_index: int,
        return_members: bool = False,
    ):
        """Run the ensemble for one cell type."""
        self._check_cell_type_index(cell_type_index)
        return self.mlps[cell_type_index](x, return_members=return_members)

    def forward_uncalibrated(self, x: torch.Tensor):
        """Concatenate uncalibrated per-cell-type aggregates."""
        if self.training:
            raise RuntimeError(
                "forward_uncalibrated() requires model.eval()."
            )
        predictions = []
        aleatorics = []
        epistemics = []
        for ensemble in self.mlps:
            prediction, aleatoric, epistemic = ensemble.forward_uncalibrated(x)
            predictions.append(prediction)
            aleatorics.append(aleatoric)
            epistemics.append(epistemic)
        return (
            torch.cat(predictions, dim=-1),
            torch.cat(aleatorics, dim=-1),
            torch.cat(epistemics, dim=-1),
        )

    def forward(self, x: torch.Tensor, return_members: bool = False):
        """Concatenate per-cell-type ensemble outputs in evaluation mode.

        Combined training through this method is rejected so callers cannot
        jointly backpropagate every cell-type ensemble.
        """
        if self.training:
            raise RuntimeError(
                "SeparateMLPEnsemble.forward() is only for evaluation. "
                "Train through forward_cell_type() so each cell type is "
                "optimized independently."
            )

        predictions = []
        aleatorics = []
        epistemics = []
        member_means = []
        member_sigmas = []
        for ensemble in self.mlps:
            outputs = ensemble(x, return_members=return_members)
            prediction, aleatoric, epistemic = outputs[:3]
            predictions.append(prediction)
            aleatorics.append(aleatoric)
            epistemics.append(epistemic)
            if return_members:
                member_means.append(outputs[3])
                member_sigmas.append(outputs[4])

        aggregate = (
            torch.cat(predictions, dim=-1),
            torch.cat(aleatorics, dim=-1),
            torch.cat(epistemics, dim=-1),
        )
        if not return_members:
            return aggregate
        return (
            *aggregate,
            torch.cat(member_means, dim=-1),
            torch.cat(member_sigmas, dim=-1),
        )


if __name__ == "__main__":
    model = SeparateMLPEnsemble(
        n_models=5,
        input_dim=20,
        n_layers=3,
        output_dim=4,
    )
    sample_input = torch.randn(3, 20)

    model.train()
    means, variances = model.forward_cell_type(sample_input, 0)
    print(
        "Train cell-type 0: "
        f"{len(means)} members, "
        f"mean {means[0].shape}, variance {variances[0].shape}"
    )

    model.eval()
    prediction, aleatoric, epistemic = model(sample_input)
    print(
        f"Eval aggregate: pred {prediction.shape}, "
        f"aleatoric {aleatoric.shape}, epistemic {epistemic.shape}"
    )
    *_, member_means, member_sigmas = model(sample_input, return_members=True)
    print(
        f"Eval members: means {member_means.shape}, "
        f"sigmas {member_sigmas.shape}"
    )
