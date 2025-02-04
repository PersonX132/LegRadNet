import torch
import torch.nn as nn
from timm import create_model

class ViTBackbone(nn.Module):
    """
    Vision Transformer from timm, final projection to 'feature_dim'.
    """
    def __init__(self, pretrained=True, feature_dim=1024, model_name="vit_base_patch16_224"):
        super().__init__()
        self.vit = create_model(model_name, pretrained=pretrained, num_classes=0)
        in_features = self.vit.num_features
        self.fc = nn.Linear(in_features, feature_dim)

    def forward(self, x):
        feats = self.vit(x)
        feats = self.fc(feats)
        return feats
