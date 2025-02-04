import torch
import torch.nn as nn
from timm import create_model

class ConvNeXtBackbone(nn.Module):
    """
    ConvNeXt backbone from timm, final projection to 'feature_dim'.
    """
    def __init__(self, pretrained=True, feature_dim=1024, model_name="convnext_base"):
        super().__init__()
        self.convnext = create_model(model_name, pretrained=pretrained, num_classes=0)
        in_features = self.convnext.num_features
        self.fc = nn.Linear(in_features, feature_dim)

    def forward(self, x):
        feats = self.convnext(x)
        feats = self.fc(feats)
        return feats
