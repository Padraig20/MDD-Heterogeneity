import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, n_layers, output_dim):
        super(MLP, self).__init__()
        layer_sizes = np.linspace(input_dim, output_dim, n_layers+2) # input, hidden..., output

        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes)-1):
            self.layers.append(nn.Linear(int(layer_sizes[i]), int(layer_sizes[i+1])))

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus() # ensure non-negative outputs
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.relu(x)
        x = self.softplus(self.layers[-1](x))
        return x

class MLPPredictor(nn.Module):
    def __init__(self, input_dim, n_layers, output_dim, layer_norm=False):
        super(MLPPredictor, self).__init__()
        # output mean and variance
        # first half of output_dim is mean, second half is variance
        self.mlp = MLP(input_dim, n_layers, output_dim*2)
        self.input_dim  = input_dim
        self.output_dim = output_dim*2
        self.layer_norm = nn.LayerNorm(input_dim) if layer_norm else None
    
    def forward(self, x):
        if self.layer_norm:
            x = self.layer_norm(x)
        return self.mlp(x)

class MLPEnsemble(nn.Module):
    def __init__(self, n_models, input_dim, n_layers, output_dim, layer_norm=False):
        super(MLPEnsemble, self).__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.models = nn.ModuleList([
            MLPPredictor(input_dim, n_layers, output_dim, layer_norm) for _ in range(n_models)
        ])
    
    def forward(self, x):
        if self.training: # during training, each model gets its own forward pass (for independent gradients)
            means = []
            vars  = []
            for model in self.models:
                model.train()
                means.append(model(x)[..., :self.output_dim])
                vars.append(model(x)[..., self.output_dim:]).clamp(min=1e-8) # avoid zero div
            return np.array(means), np.array(vars)

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