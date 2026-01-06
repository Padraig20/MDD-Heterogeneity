import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1  = nn.Linear(input_dim, hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, hidden_dim)
        self.fc3  = nn.Linear(hidden_dim, hidden_dim)
        self.fc4  = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus() # ensure non-negative outputs
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.softplus(self.fc4(x))
        return x

class MLPUncertainty(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPUncertainty, self).__init__()
        self.mlp = MLP(input_dim, hidden_dim, output_dim)
        self.input_dim  = input_dim
        self.output_dim = output_dim
    
    def forward(self, x):
        return self.mlp(x)

class MLPPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPPredictor, self).__init__()
        self.mlp = MLP(input_dim, hidden_dim, output_dim)
        self.input_dim  = input_dim
        self.output_dim = output_dim
    
    def forward(self, x):
        return self.mlp(x)