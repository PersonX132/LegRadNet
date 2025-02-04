import torch
import torch.nn as nn
import torch.nn.functional as F

class GatingNetwork(nn.Module):
    """
    Fuses feature vectors from multiple backbones via either
    an MLP-based gating or multi-head attention. 
    Defaults set for multi-head attention usage in the config.
    """
    def __init__(self, input_dim=1024, num_backbones=5, mode="multi_head_attention"):
        super().__init__()
        self.mode = mode
        self.num_backbones = num_backbones
        self.input_dim = input_dim

        # Simple MLP gating approach
        self.mlp_fc1 = nn.Linear(input_dim * num_backbones, 256)
        self.mlp_fc2 = nn.Linear(256, num_backbones)

        # Multi-head attention approach
        self.query = nn.Parameter(torch.randn(1, 1, input_dim))
        self.key_fc = nn.Linear(input_dim, input_dim)
        self.value_fc = nn.Linear(input_dim, input_dim)
        self.num_heads = 4
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=self.num_heads, batch_first=True)

    def forward(self, features_list):
        """
        features_list = [f1, f2, ..., fN], each [B, input_dim].
        """
        if self.mode == "simple_mlp":
            concat_feats = torch.cat(features_list, dim=1)
            x = F.relu(self.mlp_fc1(concat_feats))
            gating_logits = self.mlp_fc2(x)
            gating_scores = torch.softmax(gating_logits, dim=1)
            fused = 0
            for i, feats in enumerate(features_list):
                w = gating_scores[:, i].unsqueeze(-1)
                fused = fused + feats * w
            return fused, gating_scores

        else:  # "multi_head_attention"
            # Stack features => [B, num_backbones, input_dim]
            feats = torch.stack(features_list, dim=1)
            bsz = feats.size(0)

            # Expand query => [B, 1, input_dim]
            query = self.query.expand(bsz, -1, -1)

            # Linear transformations
            keys = self.key_fc(feats)
            values = self.value_fc(feats)

            # attn(query, key, value) => out shape [B, 1, input_dim], weights shape [B, 1, num_backbones]
            out, attn_weights = self.attn(query, keys, values)
            out = out.squeeze(1)               # [B, input_dim]
            gating_scores = attn_weights.squeeze(1)  # [B, num_backbones]

            return out, gating_scores
