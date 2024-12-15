import torch.nn as nn


class VideoCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(VideoCNN, self).__init__()
        
        # CNN layers for spatial features
        self.spatial_features = nn.Sequential(
            # Initial conv layer
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Conv block 1
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv block 2
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv block 3
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Temporal modeling with 3D convolution
        self.temporal_features = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(512),
            nn.ReLU(),
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, time, channels, height, width)
        batch_size, time_steps, c, h, w = x.size()
        
        # Combine batch and time dimensions for spatial CNN
        x = x.view(batch_size * time_steps, c, h, w)
        
        # Extract spatial features
        spatial_features = self.spatial_features(x)
        
        # Reshape for temporal processing
        _, c, h, w = spatial_features.size()
        spatial_features = spatial_features.view(batch_size, time_steps, c, h, w)
        spatial_features = spatial_features.permute(0, 2, 1, 3, 4)
        
        # Extract temporal features
        temporal_features = self.temporal_features(spatial_features)
        
        # Reshape for classification
        temporal_features = temporal_features.permute(0, 2, 1, 3, 4)
        temporal_features = temporal_features.contiguous()
        temporal_features = temporal_features.view(batch_size * time_steps, c, h, w)
        
        # Classification
        output = self.classifier(temporal_features)
        output = output.view(batch_size, time_steps, -1)
        
        return output