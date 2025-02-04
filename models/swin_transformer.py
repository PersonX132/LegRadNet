import torch
import torch.nn as nn
from timm import create_model

class SwinTransformerBackbone(nn.Module):
    """
    Swin Transformer using timm, then project to 'feature_dim'.
    """
    def __init__(self, pretrained=True, feature_dim=1024, model_name="swin_base_patch4_window7_224"):
        super().__init__()
        self.swin = create_model(model_name, pretrained=pretrained)
        if hasattr(self.swin, 'head'):
            in_features = self.swin.head.in_features
            self.swin.reset_classifier(0)
        else:
            raise ValueError("Swin model missing 'head' attribute.")
        self.fc = nn.Linear(in_features, feature_dim)

    def forward(self, x):
        feats = self.swin.forward_features(x)
        feats = self.fc(feats)
        return feats
