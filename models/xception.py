import torch
import torch.nn as nn
from timm import create_model

class XceptionBackbone(nn.Module):
    """
    Xception backbone from timm, final projection to 'feature_dim'.
    """
    def __init__(self, pretrained=True, feature_dim=1024, model_name="xception"):
        super().__init__()
        self.xception = create_model(model_name, pretrained=pretrained, num_classes=0)
        in_features = self.xception.num_features
        self.fc = nn.Linear(in_features, feature_dim)

    def forward(self, x):
        feats = self.xception(x)
        feats = self.fc(feats)
        return feats
