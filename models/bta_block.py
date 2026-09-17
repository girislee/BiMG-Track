# models/bta_block.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BPLSTM(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.forward_lstm = nn.LSTM(dim, dim, batch_first=True)
        self.backward_lstm = nn.LSTM(dim, dim, batch_first=True)
        self.query_proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, T, N, C)
        B, T, N, C = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()      # (B, N, T, C)
        x = x.view(B * N, T, C)

        # forward_scan
        h_f_seq, _ = self.forward_lstm(x)           # (B*N, T, C)

        # backward scan
        x_rev = torch.flip(x, dims=[1])
        h_b_seq_rev, _ = self.backward_lstm(x_rev)
        h_b_seq = torch.flip(h_b_seq_rev, dims=[1]) # (B*N, T, C)

        # attention 
        q = self.query_proj(x)                      # (B*N, T, C)
        d = C

        attn_f = torch.matmul(q, h_f_seq.transpose(1, 2)) / math.sqrt(d)
        attn_f = F.softmax(attn_f, dim=-1)
        h_f_attn = torch.matmul(attn_f, h_f_seq)

        attn_b = torch.matmul(q, h_b_seq.transpose(1, 2)) / math.sqrt(d)
        attn_b = F.softmax(attn_b, dim=-1)
        h_b_attn = torch.matmul(attn_b, h_b_seq)

        h_attn = h_f_attn + h_b_attn                # (B*N, T, C)
        h_attn = h_attn.view(B, N, T, C).permute(0, 2, 1, 3).contiguous()
        return h_attn                               # (B, T, N, C)


class CrossModalFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4 * dim, dim),
            nn.GELU(),
            nn.Linear(dim, 2)
        )

    def forward(self, h_rgb, h_tir):
        # h_rgb, h_tir: (B, T, N, C)
        C = h_rgb * h_tir
        z = torch.cat([h_rgb, h_tir, h_rgb - h_tir, C], dim=-1)  # (B,T,N,4C)
        logits = self.mlp(z)                                     # (B,T,N,2)
        weights = F.softmax(logits, dim=-1)

        a_R = weights[..., 0:1]
        a_T = weights[..., 1:2]
        h_fused = a_R * h_rgb + a_T * h_tir
        return h_fused, a_R, a_T

# BTA Block
class BTABlock(nn.Module):
    def __init__(self, dim=512, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.vit = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=0.0,
            activation='gelu',
            batch_first=True
        )
        # two bplstm branch
        self.bplstm_rgb = BPLSTM(dim, num_heads)
        self.bplstm_tir = BPLSTM(dim, num_heads)
        self.cross_fusion = CrossModalFusion(dim)

    def forward(self, rgb, tir):
        # rgb, tir: (B, T, N, C)
        B, T, N, C = rgb.shape

        # vit encoder
        rgb_flat = rgb.reshape(B * T, N, C)
        tir_flat = tir.reshape(B * T, N, C)
        rgb_flat = self.vit(rgb_flat)
        tir_flat = self.vit(tir_flat)
        rgb = rgb_flat.reshape(B, T, N, C)
        tir = tir_flat.reshape(B, T, N, C)

        # bplstm
        h_rgb = self.bplstm_rgb(rgb)
        h_tir = self.bplstm_tir(tir)

        # cross fusion
        h_fused, a_R, a_T = self.cross_fusion(h_rgb, h_tir)

        # residual output
        rgb_out = rgb + h_fused
        tir_out = tir + h_fused
        return rgb_out, tir_out, h_fused, a_R, a_T
