import torch.nn as nn

class HousingNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),  # First layer: 13 → 64
            nn.ReLU(),
            nn.Linear(64, 32),         # Second layer: 64 → 32
            nn.ReLU(),
            nn.Linear(32, 1)           # Output layer: 32 → 1 (regression)
        )

    def forward(self, x):
        return self.net(x)
