# models/amgi.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DSM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim)
        )
        self.tau = math.sqrt(dim)

    def forward(self, rgb, tir):
        # rgb, tir: (B*T, N, C)
        B, N, C = rgb.shape
        x = torch.cat([rgb, tir], dim=-1)          # (B, N, 2C)
        S = self.mlp(x)                            # (B, N, C)

        #  delta
        S_flat = S.reshape(B, -1)                  # (B, N*C)
        p = F.softmax(S_flat, dim=-1)
        H = -torch.sum(p * torch.log2(p + 1e-8), dim=-1)
        H_norm = H / math.log2(N * C)

        delta = torch.where(
            H_norm <= 0.3,
            torch.tensor(0.7, device=S.device),
            torch.where(
                H_norm >= 0.7,
                torch.tensor(0.3, device=S.device),
                1 - H_norm
            )
        ).view(B, 1, 1)

        # mask 
        mask_soft = torch.sigmoid(S / self.tau)
        mask_hard = (mask_soft > delta).float()
        mask = mask_hard - mask_soft.detach() + mask_soft

        sparse_rgb = mask * rgb
        sparse_tir = mask * tir
        return sparse_rgb, sparse_tir

class IPA(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, 1, 1))

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn + self.rel_pos_bias
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class CPA(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


# Adaptive Multi-Granularity Interactor (AMGI)
class AMGI(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dsm = DSM(dim)
        self.ipa = IPA(dim, num_heads)
        self.cpa = CPA(dim, num_heads)
        self.fusion = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.Sigmoid()
        )

    def forward(self, rgb, tir):
        # rgb, tir: (B, T, N, C)
        B, T, N, C = rgb.shape
        rgb_flat = rgb.reshape(B * T, N, C)
        tir_flat = tir.reshape(B * T, N, C)

        # dsm
        sparse_rgb, sparse_tir = self.dsm(rgb_flat, tir_flat)

        # dual attention
        local = self.ipa(sparse_rgb)      # RGB → IPA
        global_ = self.cpa(sparse_tir)    # TIR → CPA

        # fusion
        alpha = self.fusion(torch.cat([local, global_], dim=-1))
        fusion = alpha * local + (1 - alpha) * global_
        return fusion.reshape(B, T, N, C)
