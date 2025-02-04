import torch
import torch.nn as nn

from models.densenet import DenseNetBackbone
from models.swin_transformer import SwinTransformerBackbone
from models.convnext import ConvNeXtBackbone
from models.vit import ViTBackbone
from models.xception import XceptionBackbone
from models.gating_network import GatingNetwork

class EnsembleModel(nn.Module):
    """
    Combines multiple backbones, fuses their features via gating,
    and produces a 5-class output by default.
    """
    def __init__(self,
                 backbones=["densenet","swin_transformer","convnext","vit","xception"],
                 gating_network=True,
                 gating_mode="multi_head_attention",
                 feature_dim=1024,
                 num_classes=5,
                 pretrained=True):
        super().__init__()
        self.backbones_list = backbones
        self.backbone_modules = nn.ModuleDict()

        # Initialize each backbone
        for b in backbones:
            if b == "densenet":
                self.backbone_modules[b] = DenseNetBackbone(pretrained=pretrained, feature_dim=feature_dim)
            elif b == "swin_transformer":
                self.backbone_modules[b] = SwinTransformerBackbone(pretrained=pretrained, feature_dim=feature_dim)
            elif b == "convnext":
                self.backbone_modules[b] = ConvNeXtBackbone(pretrained=pretrained, feature_dim=feature_dim)
            elif b == "vit":
                self.backbone_modules[b] = ViTBackbone(pretrained=pretrained, feature_dim=feature_dim)
            elif b == "xception":
                self.backbone_modules[b] = XceptionBackbone(pretrained=pretrained, feature_dim=feature_dim)
            else:
                raise ValueError(f"Unknown backbone: {b}")

        self.gating_enabled = gating_network
        self.num_backbones = len(backbones)
        self.feature_dim = feature_dim

        if gating_network:
            self.gate = GatingNetwork(
                input_dim=feature_dim,
                num_backbones=self.num_backbones,
                mode=gating_mode
            )
        else:
            self.fusion_fc = nn.Linear(feature_dim * self.num_backbones, feature_dim)

        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        # Extract features from each backbone
        features_list = []
        for b in self.backbones_list:
            feats = self.backbone_modules[b](x)
            features_list.append(feats)

        if self.gating_enabled:
            fused_features, gating_scores = self.gate(features_list)
        else:
            concat_feats = torch.cat(features_list, dim=1)
            fused_features = self.fusion_fc(concat_feats)
            gating_scores = None

        logits = self.classifier(fused_features)
        return logits, gating_scores
