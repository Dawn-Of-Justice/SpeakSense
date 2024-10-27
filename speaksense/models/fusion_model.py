import torch
import torch.nn as nn

class MultimodalFusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.model.fusion.hidden_dim,
            num_heads=config.model.fusion.num_heads,
            dropout=config.model.fusion.dropout
        )
        
        # Fusion transformer
        self.fusion_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.model.fusion.hidden_dim,
                nhead=config.model.fusion.num_heads,
                dropout=config.model.fusion.dropout
            ),
            num_layers=config.model.fusion.num_layers
        )
        
        # Modality-specific confidence estimation
        self.confidence_nets = nn.ModuleDict({
            'audio': ConfidenceNet(config),
            'video': ConfidenceNet(config),
            'lip': ConfidenceNet(config)
        })
        
    def forward(self, features):
        # Get confidence scores
        confidences = {
            modality: self.confidence_nets[modality](features[f'{modality}_features'])
            for modality in ['audio', 'video', 'lip']
        }
        
        # Apply confidence-weighted fusion
        weighted_features = {
            modality: features[f'{modality}_features'] * confidences[modality]
            for modality in ['audio', 'video', 'lip']
        }
        
        # Cross-modal attention
        audio_attended = self.cross_attention(
            weighted_features['audio'],
            weighted_features['video'],
            weighted_features['video']
        )[0]
        
        # Concatenate features
        fused_features = torch.cat([
            audio_attended,
            weighted_features['video'],
            weighted_features['lip']
        ], dim=1)
        
        # Apply fusion transformer
        output = self.fusion_transformer(fused_features)
        
        return output, confidences

class ConfidenceNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.model.fusion.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)