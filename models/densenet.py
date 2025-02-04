import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DenseNetBackbone(nn.Module):
    """
    DenseNet201 backbone. Produces a feature vector of size 'feature_dim'.
    """
    def __init__(self, pretrained=True, feature_dim=1024):
        super().__init__()
        densenet = models.densenet201(pretrained=pretrained)
        self.features = densenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # DenseNet201 final conv output typically 1920 channels
        self.fc = nn.Linear(1920, feature_dim)

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
