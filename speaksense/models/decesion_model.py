import torch
import torch.nn as nn

class DecisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Main decision network
        self.decision_network = nn.Sequential(
            nn.Linear(config.model.fusion.hidden_dim * 3, 512),
            nn.ReLU(),
            nn.Dropout(config.model.fusion.dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, fused_features):
        return self.decision_network(fused_features)