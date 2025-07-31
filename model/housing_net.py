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

class HousingNet2(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.2):
        super().__init__()
        
        # Feature extraction layers with batch normalization
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # Residual connection path
        self.residual = nn.Linear(input_dim, 64)
        
        # Final prediction layers
        self.predictor = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),  # Less dropout in later layers
            
            nn.Linear(32, 16),
            nn.ReLU(),
            
            nn.Linear(16, 1)
        )
        
        # Initialize weights for better training
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization for better convergence"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Feature extraction
        features = self.feature_extractor(x)
        
        # Residual connection (skip connection from input to features)
        residual = self.residual(x)
        
        # Combine features with residual connection
        combined = features + residual
        
        # Final prediction
        return self.predictor(combined)