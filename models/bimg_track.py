import torch
import torch.nn as nn

from .patch_embed import PatchEmbed
from .bta_block import BTABlock
from .amgi import AMGI
from .prediction_head import PredictionHead


class BiMGTrack(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=3,
                 embed_dim=512, depth=12, num_heads=8):
        super().__init__()
        self.patch_embed_rgb = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.patch_embed_tir = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        self.bta_blocks = nn.ModuleList([
            BTABlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.amgis = nn.ModuleList([
            AMGI(embed_dim, num_heads) for _ in range(depth)
        ])
        self.head = PredictionHead(embed_dim)

    def forward(self, rgb_seq, tir_seq):
        # rgb_seq, tir_seq: (B, T, 3, H, W)
        B, T, C, H, W = rgb_seq.shape

        rgb_tokens, tir_tokens = [], []
        for t in range(T):
            rgb_tokens.append(self.patch_embed_rgb(rgb_seq[:, t]))
            tir_tokens.append(self.patch_embed_tir(tir_seq[:, t]))

        rgb = torch.stack(rgb_tokens, dim=1)   # (B, T, N, C)
        tir = torch.stack(tir_tokens, dim=1)

        for bta, amgi in zip(self.bta_blocks, self.amgis):
            rgb, tir, h_fused, a_R, a_T = bta(rgb, tir)
            fusion = amgi(rgb, tir)
            rgb = rgb + fusion
            tir = tir + fusion

        bbox, cls = self.head(rgb)
        return bbox, cls
