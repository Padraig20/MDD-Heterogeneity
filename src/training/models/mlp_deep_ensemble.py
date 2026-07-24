import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, n_layers, output_dim, dropout=0.0):
        super(MLP, self).__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
        layer_sizes = np.linspace(input_dim, output_dim, n_layers+2) # input, hidden..., output

        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes)-1):
            self.layers.append(nn.Linear(int(layer_sizes[i]), int(layer_sizes[i+1])))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.relu(x)
            x = self.dropout(x)
        # raw (linear) logits; per-head activations are applied in MLPPredictor so that
        # the mean and the variance can be parameterized differently.
        x = self.layers[-1](x)
        return x

class MLPPredictor(nn.Module):
    def __init__(
        self,
        input_dim,
        n_layers,
        output_dim,
        layer_norm=False,
        dropout=0.0,
    ):
        super(MLPPredictor, self).__init__()
        # the network emits 2 * output_dim logits:
        #   first  output_dim -> mean
        #   second output_dim -> variance (raw logit, made positive via softplus below)
        self.mlp = MLP(
            input_dim,
            n_layers,
            output_dim * 2,
            dropout=dropout,
        )
        self.input_dim  = input_dim
        self.n_targets  = output_dim
        self.output_dim = output_dim*2
        self.dropout_rate = dropout
        self.softplus   = nn.Softplus()
        self.layer_norm = nn.LayerNorm(input_dim) if layer_norm else None
    
    def forward(self, x):
        if self.layer_norm:
            x = self.layer_norm(x)
        raw  = self.mlp(x)
        # mean: softplus keeps predictions non-negative (targets are >= 0).
        # variance: softplus gives a strictly positive, *unbounded* variance, i.e. a
        # proper Gaussian scale (unlike sigmoid, which caps it at 1).
        mean = self.softplus(raw[..., :self.n_targets])
        var  = self.softplus(raw[..., self.n_targets:])
        return torch.cat([mean, var], dim=-1)

class MLPEnsemble(nn.Module):
    def __init__(
        self,
        n_models,
        input_dim,
        n_layers,
        output_dim,
        layer_norm=False,
        dropout=0.0,
    ):
        super(MLPEnsemble, self).__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.models = nn.ModuleList([
            MLPPredictor(
                input_dim,
                n_layers,
                output_dim,
                layer_norm,
                dropout,
            )
            for _ in range(n_models)
        ])
        self.dropout_rate = dropout
    
    def forward(self, x):
        if self.training: # during training, each model gets its own forward pass (for independent gradients)
            means = []
            vars  = []
            for model in self.models:
                model.train()
                preds = model(x)
                means.append(preds[..., :self.output_dim])
                vars.append(preds[..., self.output_dim:])
            return means, vars

        else: # calculate aggregated predictions and uncertainties
            means = []
            vars  = []
            for model in self.models:
                model.eval()
                preds = model(x) # shape (batch_size, output_dim*2)
                mu    = preds[..., :self.output_dim]
                var   = preds[..., self.output_dim:]
                means.append(mu)
                vars.append(var)

            prediction    = torch.mean(torch.stack(means), dim=0)   # MC estimate of mean
            aleatoric_unc = torch.mean(torch.stack(vars), dim=0)    # MC estimate of variance
            epistemic_unc = torch.mean(torch.stack([mu**2 for mu in means]), dim=0) - prediction**2 # MC estimate of mu^2 - prediction

            return prediction, aleatoric_unc, epistemic_unc
    
if __name__ == "__main__":
    # example usage
    model = MLPEnsemble(n_models=5, input_dim=20, n_layers=3, output_dim=5)
    sample_input = torch.randn(4, 20)
    model.train()
    output = model(sample_input)
    print(f"Output shape: {len(output[0])} models. Mean shape: {output[0][0].shape}, Variance shape: {output[1][0].shape}")
    model.eval()
    output = model(sample_input)
    print(f"Output shape: {output[0].shape}, Aleatoric Uncertainty shape: {output[1].shape}, Epistemic Uncertainty shape: {output[2].shape}")
