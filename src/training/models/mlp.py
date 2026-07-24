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
        # Keep dropout outside ``layers`` so old checkpoint parameter names
        # (``layers.N.weight``) remain unchanged.
        self.dropout = nn.Dropout(dropout)
        self.softplus = nn.Softplus() # ensure non-negative outputs
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.softplus(self.layers[-1](x))
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
        self.mlp = MLP(input_dim, n_layers, output_dim, dropout=dropout)
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout
        self.layer_norm = nn.LayerNorm(input_dim) if layer_norm else None
    
    def forward(self, x):
        if self.layer_norm:
            x = self.layer_norm(x)
        return self.mlp(x)
    
if __name__ == "__main__":
    # example usage
    model = MLPPredictor(input_dim=20, n_layers=3, output_dim=5)
    sample_input = torch.randn(4, 20)
    output = model(sample_input)
    print(f"Output shape: {output.shape}")
