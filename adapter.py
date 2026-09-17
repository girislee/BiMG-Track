import torch
from torch import nn
import timm
import math


'''
def forward_block(self, x):
    x = x + self.drop_path(self.attn(self.norm1(x))) + self.drop_path(self.adapter_attn(self.norm1(x))) * self.s
    x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adapter_mlp(self.norm2(x))) * self.s
    return x


def forward_block_attn(self, x):
    x = x + self.drop_path(self.attn(self.norm1(x))) + self.drop_path(self.adapter_attn(self.norm1(x))) * self.s
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x
'''


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


import torch
import torch.nn as nn


class EnhancedBiDirectAdapter(nn.Module):
    """增强版双向适配器，融合注意力机制"""

    def __init__(self, dim=8, xavier_init=False):
        super().__init__()
        # 降维投影
        self.adapter_down = nn.Linear(768, dim)
        # 注意力增强模块（轻量级设计）
        self.attention = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            TokenMixer(dim=dim),  # 使用前述的改进版TokenMixer
            nn.Linear(dim, dim)
        )
        # 升维恢复
        self.adapter_up = nn.Linear(dim, 768)
        # 残差缩放因子（可学习）
        self.alpha = nn.Parameter(torch.zeros(1))

        # 初始化策略
        self._init_weights(xavier_init)

    def _init_weights(self, xavier_init):
        # 注意力层合理初始化
        nn.init.xavier_uniform_(self.attention[0].weight)
        nn.init.xavier_uniform_(self.attention[3].weight)
        # 降维/升维层零初始化
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_up.weight)
        if xavier_init:
            nn.init.xavier_uniform_(self.adapter_down.weight)
            nn.init.xavier_uniform_(self.adapter_up.weight)

    def forward(self, x):
        # 原始特征保存
        identity = x

        # 降维处理
        x_down = self.adapter_down(x)  # [B, N, dim]

        # 注意力增强
        x_attn = self.attention(x_down)  # [B, N, dim]
        x_down = x_down + x_attn  # 残差连接

        # 升维恢复
        x_up = self.adapter_up(x_down)  # [B, N, 768]

        # 残差融合
        return identity + self.alpha * x_up


class TokenMixer(nn.Module):
    """轻量级Token混合器（兼容三维输入）"""

    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.mlp(self.norm(x)) + x


class Bi_direct_adapter(nn.Module):
    def __init__(self, dim=8, xavier_init=False):
        super().__init__()

        self.adapter_down = nn.Linear(768, dim)  
        self.adapter_up = nn.Linear(dim, 768)  
        self.adapter_mid = nn.Linear(dim, dim)

        #nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_mid.bias)
        nn.init.zeros_(self.adapter_mid.weight)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        #self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        B, N, C = x.shape
        x_down = self.adapter_down(x)   
        #x_down = self.act(x_down)
        x_down = self.adapter_mid(x_down)
        #x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  
        #print("return adap x", x_up.size())
        return x_up



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_



class Attention(nn.Module):
    """ Basic attention of IPSA and CPSA.

    Args:
        dim (int): Number of input channels.
        patch_size (tuple[int]): Patch size.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value.
        qk_scale (float | None, optional): Default qk scale is head_dim ** -0.5.
        attn_drop (float, optional): Dropout ratio of attention weight.
        proj_drop (float, optional): Dropout ratio of output.
        rpe (bool): Use relative position encoding or not.
    """

    def __init__(self, dim, patch_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., rpe=True):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size  # Ph, Pw
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.rpe = rpe

        if self.rpe:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * patch_size[0] - 1) * (2 * patch_size[1] - 1), num_heads))  # 2*Ph-1 * 2*Pw-1, nH

            # get pair-wise relative position index for each token inside one patch
            coords_h = torch.arange(self.patch_size[0])
            coords_w = torch.arange(self.patch_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Ph, Pw
            coords_flatten = torch.flatten(coords, 1)  # 2, Ph*Pw
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Ph*Pw, Ph*Pw
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Ph*Pw, Ph*Pw, 2
            relative_coords[:, :, 0] += self.patch_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.patch_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.patch_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Ph*Pw, Ph*Pw
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_patches*B, N, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.patch_size[0] * self.patch_size[1], self.patch_size[0] * self.patch_size[1], -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



import warnings
import torch
import torch.nn as nn
import numpy as np
import warnings


class DualAttention(nn.Module):
    """ 双路注意力机制 (Intra-Patch + Cross-Patch Attention)

    Args:
        dim (int): 输入特征维度
        patch_size (tuple): 基础分块尺寸 (ph, pw)
        num_heads (int): 注意力头数
        qkv_bias (bool): 是否在QKV线性层使用偏置
        qk_scale (float): 注意力缩放因子
        attn_drop (float): 注意力矩阵dropout率
        proj_drop (float): 输出投影层dropout率
        rpe (bool): 是否使用相对位置编码
    """

    def __init__(self, dim, patch_size, num_heads=8,
                 qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0., rpe=True):
        super().__init__()
        self.dim = dim
        self.patch_size = tuple(patch_size)
        self.num_heads = num_heads

        # 块内自注意力 (Intra-Patch Self-Attention)
        self.ipsa = Attention(
            dim=dim,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            rpe=rpe
        )

        # 跨块自注意力 (Cross-Patch Self-Attention)
        self.cpsa = Attention(
            dim=dim,
            patch_size=(1, 1),  # 单点注意力模式
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            rpe=False  # 跨块注意力不使用位置编码
        )

    def factorize_patches(self, num_patches):
        """ 智能分解分块数为行列数

        Args:
            num_patches (int): 总块数
        Returns:
            tuple: (Hp, Wp) 分块行列数
        """
        if num_patches <= 0:
            raise ValueError(f"无效分块数: {num_patches}，必须为正整数")

        # 从平方根开始向下搜索最大因数
        max_factor = int(np.sqrt(num_patches))
        for i in range(max_factor, 0, -1):
            if num_patches % i == 0:
                return i, num_patches // i
        return 1, num_patches  # 处理素数情况

    def forward(self, x, Hp=None, Wp=None):
        """ 前向传播 (含自动维度适配)

        Args:
            x: 输入特征张量 (B*num_patches, ph*pw, C)
            Hp (int, optional): 预设分块行数
            Wp (int, optional): 预设分块列数
        """
        B_, N, C = x.shape
        ph, pw = self.patch_size

        #######################################
        # 阶段1：分块参数校验与修正
        #######################################

        # 验证基础分块尺寸合法性
        if ph <= 0 or pw <= 0:
            raise ValueError(f"无效分块尺寸 ({ph}, {pw})，必须为正整数")

        # 计算实际分块数
        actual_num_patches = B_ // x.size(0)

        # 自动修正分块行列数
        if Hp is None or Wp is None:
            Hp, Wp = self.factorize_patches(actual_num_patches)
        else:
            # 强制类型转换（处理可能传入的Tensor类型）
            Hp, Wp = int(Hp), int(Wp)
            if Hp * Wp != actual_num_patches:
                warnings.warn(
                    f"预设分块参数 ({Hp}x{Wp}) 不匹配实际分块数 {actual_num_patches}，"
                    f"正在自动调整...",
                    UserWarning
                )
                Hp, Wp = self.factorize_patches(actual_num_patches)

        #######################################
        # 阶段2：块内注意力计算
        #######################################
        ipsa_out = self.ipsa(x)

        #######################################
        # 阶段3：跨块注意力计算
        #######################################

        # 维度重组 (B*num_patches, ph*pw, C) -> (B, num_patches, ph*pw, C)
        batch_size = B_ // (Hp * Wp)
        x_cpsa = x.view(batch_size, Hp * Wp, ph * pw, C)

        # 维度置换 (B, Hp*Wp, ph*pw, C) -> (B*ph*pw, Hp*Wp, C)
        x_cpsa = x_cpsa.permute(0, 2, 1, 3).contiguous()
        x_cpsa = x_cpsa.view(-1, Hp * Wp, C)

        # 跨块注意力
        cpsa_out = self.cpsa(x_cpsa)

        # 恢复原始维度 (B*ph*pw, Hp*Wp, C) -> (B*num_patches, ph*pw, C)
        cpsa_out = cpsa_out.view(batch_size, ph * pw, Hp * Wp, C)
        cpsa_out = cpsa_out.permute(0, 2, 1, 3).contiguous()
        cpsa_out = cpsa_out.view(-1, ph * pw, C)

        #######################################
        # 阶段4：特征融合与输出
        #######################################
        return ipsa_out + cpsa_out

    def flops(self, N):
        """ FLOPs计算 (示例) """
        flops = 0
        # 块内注意力FLOPs
        flops += self.ipsa.flops(N)
        # 跨块注意力FLOPs
        flops += self.cpsa.flops(N)
        return flops





class DynamicSparseMask(nn.Module):
    def __init__(self, dim, temperature=1.0, threshold=0.5):
        super().__init__()
        self.mask_generator = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1)
        )
        self.temperature = temperature
        self.threshold = threshold

        # 初始化参数
        nn.init.kaiming_normal_(self.mask_generator[0].weight)
        nn.init.zeros_(self.mask_generator[0].bias)
        nn.init.zeros_(self.mask_generator[2].weight)
        nn.init.zeros_(self.mask_generator[2].bias)

    def forward(self, x):
        """ 生成动态稀疏掩码
        Args:
            x: 输入特征 (B, N, C)
        Returns:
            masked_x: 掩码处理后的特征
            keep_ratio: 保留token的比例（用于监控）
        """
        # 生成重要性分数
        scores = self.mask_generator(x).squeeze(-1)  # (B, N)

        # Gumbel-Softmask实现可微分二值化
        mask = torch.sigmoid(scores / self.temperature)

        # 动态阈值调整
        sorted_scores, _ = torch.sort(scores, descending=True)
        adaptive_threshold = sorted_scores[:, int(scores.size(1) * self.threshold)]
        mask = (scores >= adaptive_threshold.unsqueeze(1)).float()

        # 应用掩码
        masked_x = x * mask.unsqueeze(-1)

        # 计算保留比例
        keep_ratio = mask.sum() / mask.numel()

        return masked_x, keep_ratio




class EnhancedDualAttentionAdapter(nn.Module):
    def __init__(self, dim=8, num_heads=2, patch_size=(2, 2), sparsity=0.3, attn_drop=0.1):
        super().__init__()
        self.adapter_down = nn.Linear(768, dim)
        self.adapter_up = nn.Linear(dim, 768)
        self.patch_size = patch_size  # 显式存储分块尺寸
        self.sparse_mask = DynamicSparseMask(dim=dim, threshold=sparsity)
        self.dual_attn = DualAttention(
            dim=dim,
            patch_size=patch_size,
            num_heads=num_heads,
            attn_drop=attn_drop
        )
        self._init_weights()

    def _init_weights(self):
        """ 初始化保持不变 """
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

    def safe_factorize(self, num_patches):
        """ 安全因数分解方法 """
        if num_patches <= 0:
            raise ValueError(f"无效分块数: {num_patches}，请检查输入特征长度和分块尺寸")

        # 寻找最接近平方根的因数对
        max_factor = int(np.sqrt(num_patches))
        for i in range(max_factor, 0, -1):
            if num_patches % i == 0:
                return i, num_patches // i
        return 1, num_patches  # 处理素数情况

    def forward(self, x):
        # 降维处理
        x_down = self.adapter_down(x)  # (B, N, dim)

        # 动态稀疏掩码
        x_sparse, keep_ratio = self.sparse_mask(x_down)

        # 分块处理
        B, N, C = x_sparse.shape
        ph, pw = self.patch_size
        ph_pw = ph * pw

        # 动态填充保证可分块
        if N % ph_pw != 0:
            pad = ph_pw - (N % ph_pw)
            x_sparse = F.pad(x_sparse, (0, 0, 0, pad, 0, 0))  # 右侧填充
            N = x_sparse.shape[1]

        # 计算分块参数
        num_patches = N // ph_pw
        Hp, Wp = self.safe_factorize(num_patches)

        # 维度校验
        try:
            assert Hp * Wp == num_patches, f"分块数校验失败 {Hp}x{Wp}≠{num_patches}"
            assert ph_pw == ph * pw, f"分块尺寸异常 {ph}x{pw}≠{ph * pw}"
        except AssertionError as e:
            raise RuntimeError(
                f"分块参数异常，请检查:\n"
                f"- 输入特征长度: {N} (填充后)\n"
                f"- 分块尺寸: {ph}x{pw}\n"
                f"- 分块数: {num_patches}\n"
                f"- 计算行列数: {Hp}x{Wp}"
            ) from e

        # 注意力计算
        x_attn = x_sparse.view(B * num_patches, ph_pw, C)
        attn_out = self.dual_attn(x_attn, Hp, Wp)

        # 恢复维度（自动去除填充）
        x_out = attn_out.view(B, N, C)
        x_out = x_down + x_out
        return self.adapter_up(x_out)  # 移除了keep_ratio





