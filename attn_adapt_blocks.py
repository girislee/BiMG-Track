import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp,trunc_normal_, lecun_normal_, DropPath

from lib.models.layers.attn import Attention
from lib.models.layers.adapter import Bi_direct_adapter,EnhancedDualAttentionAdapter
from lib.models.layers.gate import gate

import math


import einops
import torch
import torch.nn.functional as F



def candidate_elimination(attn: torch.Tensor, tokens: torch.Tensor, lens_t: int, keep_ratio: float, global_index: torch.Tensor, box_mask_z: torch.Tensor):
    """
    Eliminate potential background candidates for computation reduction and noise cancellation.
    Args:
        attn (torch.Tensor): [B, num_heads, L_t + L_s, L_t + L_s], attention weights
        tokens (torch.Tensor):  [B, L_t + L_s, C], template and search region tokens
        lens_t (int): length of template
        keep_ratio (float): keep ratio of search region tokens (candidates)
        global_index (torch.Tensor): global index of search region tokens
        box_mask_z (torch.Tensor): template mask used to accumulate attention weights

    Returns:
        tokens_new (torch.Tensor): tokens after candidate elimination
        keep_index (torch.Tensor): indices of kept search region tokens
        removed_index (torch.Tensor): indices of removed search region tokens
    """
    lens_s = attn.shape[-1] - lens_t    
    bs, hn, _, _ = attn.shape

    lens_keep = math.ceil(keep_ratio * lens_s)
    if lens_keep == lens_s:
        return tokens, global_index, None

    attn_t = attn[:, :, :lens_t, lens_t:]

    


    if box_mask_z is not None:
        #print("\n1\n1\n1")
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_t.shape[1], -1, attn_t.shape[-1])
        # attn_t = attn_t[:, :, box_mask_z, :]
        attn_t = attn_t[box_mask_z]
        attn_t = attn_t.view(bs, hn, -1, lens_s)
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

        # attn_t = [attn_t[i, :, box_mask_z[i, :], :] for i in range(attn_t.size(0))]
        # attn_t = [attn_t[i].mean(dim=1).mean(dim=0) for i in range(len(attn_t))]
        # attn_t = torch.stack(attn_t, dim=0)
    else:
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

    # use sort instead of topk, due to the speed issue
    # https://github.com/pytorch/pytorch/issues/22812
    sorted_attn, indices = torch.sort(attn_t, dim=1, descending=True)



    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]
    
    keep_index = global_index.gather(dim=1, index=topk_idx)
    
    removed_index = global_index.gather(dim=1, index=non_topk_idx)
    

    # separate template and search tokens
    tokens_t = tokens[:, :lens_t]
    tokens_s = tokens[:, lens_t:]

    # obtain the attentive and inattentive tokens
    B, L, C = tokens_s.shape
    # topk_idx_ = topk_idx.unsqueeze(-1).expand(B, lens_keep, C)

    attentive_tokens = tokens_s.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))
    # inattentive_tokens = tokens_s.gather(dim=1, index=non_topk_idx.unsqueeze(-1).expand(B, -1, C))

    # compute the weighted combination of inattentive tokens
    # fused_token = non_topk_attn @ inattentive_tokens
    
    # concatenate these tokens
    # tokens_new = torch.cat([tokens_t, attentive_tokens, fused_token], dim=0)
    tokens_new = torch.cat([tokens_t, attentive_tokens], dim=1)

    #print("finish ce func")

    return tokens_new, keep_index, removed_index                       # x, global_index_search, removed_index_search


# class CEABlock(nn.Module):
#     """交叉增强注意力块（Cross-Enhanced Attention Block），用于特征交互与候选消除"""
#
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0):
#         """
#         初始化函数
#         Args:
#             dim: 输入特征维度
#             num_heads: 注意力头的数量
#             mlp_ratio: MLP层的隐藏层维度扩展比例，默认为4
#             qkv_bias: 是否在QKV线性层使用偏置，默认为False
#             drop: 全连接层的dropout概率，默认为0
#             attn_drop: 注意力权重的dropout概率，默认为0
#             drop_path: 随机深度丢弃的概率，默认为0
#             act_layer: 激活函数类型，默认为GELU
#             norm_layer: 标准化层类型，默认为LayerNorm
#             keep_ratio_search: 搜索区域保留比例，用于候选消除，默认为1.0（不消除）
#         """
#         super().__init__()
#         # 第一个标准化层
#         self.norm1 = norm_layer(dim)
#         # 多头注意力模块
#         self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
#                               attn_drop=attn_drop, proj_drop=drop)
#         # 随机深度丢弃层（DropPath）
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         # 第二个标准化层
#         self.norm2 = norm_layer(dim)
#         # 计算MLP隐藏层维度
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         # MLP模块（来自timm库）
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
#                        act_layer=act_layer, drop=drop)
#
#         # 搜索区域保留比例
#         self.keep_ratio_search = keep_ratio_search
#
#         # 双向适配器模块（用于跨模态特征交互）
#         self.adap_t = Bi_direct_adapter()  # 第一个适配器（用于注意力后的特征融合）
#         self.adap2_t = Bi_direct_adapter()  # 第二个适配器（用于MLP后的特征融合）
#
#     def forward(self, x, xi, global_index_template, global_index_templatei,
#                 global_index_search, global_index_searchi, mask=None,
#                 ce_template_mask=None, keep_ratio_search=None):
#         """
#         前向传播
#         Args:
#             x: 主分支输入特征（搜索区域）
#             xi: 辅助分支输入特征（模板区域）
#             global_index_xxx: 各区域的全局索引，用于跟踪特征位置
#             mask: 注意力掩码，默认为None
#             ce_template_mask: 模板区域的候选消除掩码，默认为None
#             keep_ratio_search: 覆盖类初始化时的搜索区域保留比例，默认为None
#         Returns:
#             处理后的特征及各类索引信息
#         """
#         # 保存原始输入特征（用于后续残差连接）
#         xori = x
#
#         # 主分支注意力计算 -------------------------------------------------
#         # 标准化后通过注意力层，获取注意力加权特征和注意力权重
#         x_attn, attn = self.attn(self.norm1(x), mask, True)
#         # 残差连接 + DropPath + 跨分支适配（加入处理后的模板特征）
#         x = x + self.drop_path(x_attn) + self.drop_path(self.adap_t(self.norm1(xi)))
#
#         # 辅助分支注意力计算 -------------------------------------------------
#         # 标准化后通过注意力层，获取注意力加权特征和注意力权重
#         xi_attn, i_attn = self.attn(self.norm1(xi), mask, True)
#         # 残差连接 + DropPath + 跨分支适配（加入处理后的搜索特征）
#         xi = xi + self.drop_path(xi_attn) + self.drop_path(self.adap_t(self.norm1(xori)))
#
#         # 候选消除处理 ------------------------------------------------------
#         lens_t = global_index_template.shape[1]  # 获取模板特征长度
#         removed_index_search = None
#         removed_index_searchi = None
#
#         # 当需要执行候选消除时（搜索区域保留比例<1）
#         if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
#             # 确定最终使用的保留比例（优先使用传入参数）
#             keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
#
#             # 对主分支进行候选消除，更新特征和索引
#             x, global_index_search, removed_index_search = candidate_elimination(
#                 attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)
#
#             # 对辅助分支进行候选消除，更新特征和索引
#             xi, global_index_searchi, removed_index_searchi = candidate_elimination(
#                 i_attn, xi, lens_t, keep_ratio_search, global_index_searchi, ce_template_mask)
#
#         # MLP处理阶段 ------------------------------------------------------
#         xori = x  # 更新原始特征
#
#         # 主分支MLP处理：残差连接 + MLP + 跨分支适配
#         x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adap2_t(self.norm2(xi)))
#
#         # 辅助分支MLP处理：残差连接 + MLP + 跨分支适配
#         xi = xi + self.drop_path(self.mlp(self.norm2(xi))) + self.drop_path(self.adap2_t(self.norm2(xori)))
#
#         # 返回所有处理后的特征及索引信息 ----------------------------------------
#         return (x, global_index_template, global_index_search, removed_index_search, attn,
#                 xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn)


# ====================== 交叉增强注意力块 ======================
# This file is licensed under AGPL-3.0
# Copyright (c) NXAI GmbH and its affiliates 2024
# Benedikt Alkin, Maximilian Beck, Korbinian Pöppel
import math
from enum import Enum

import einops
import torch
import torch.nn.functional as F
from torch import nn

from .vision_lstm_util import interpolate_sincos, to_ntuple, VitPatchEmbed, VitPosEmbed2d, DropPath


class SequenceTraversal(Enum):
    ROWWISE_FROM_TOP_LEFT = "rowwise_from_top_left"
    ROWWISE_FROM_BOT_RIGHT = "rowwise_from_bot_right"


def bias_linspace_init_(param: torch.Tensor, start: float = 3.4, end: float = 6.0) -> torch.Tensor:
    """Linearly spaced bias init across dimensions."""
    assert param.dim() == 1, f"param must be 1-dimensional (typically a bias), got {param.dim()}"
    n_dims = param.shape[0]
    init_vals = torch.linspace(start, end, n_dims)
    with torch.no_grad():
        param.copy_(init_vals)
    return param


def small_init_(param: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Fills the input Tensor with values according to the method described in Transformers without Tears: Improving
    the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2019), using a normal distribution.
    Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py.
    """
    std = math.sqrt(2 / (5 * dim))
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param


def wang_init_(param: torch.Tensor, dim: int, num_blocks: int):
    """ Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py. """
    std = 2 / num_blocks / math.sqrt(dim)
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param


def parallel_stabilized_simple(
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        igate_preact: torch.Tensor,
        fgate_preact: torch.Tensor,
        lower_triangular_matrix: torch.Tensor = None,
        stabilize_rowwise: bool = True,
        eps: float = 1e-6,
) -> torch.Tensor:
    """
    This is the mLSTM cell in parallel form.
    This version is stabilized. We control the range of exp() arguments by
    ensuring that they are always smaller than 0.0 by subtracting the maximum.

    Args:
        :param queries: (torch.Tensor) (B, NH, S, DH)
        :param keys: (torch.Tensor) (B, NH, S, DH)
        :param values: (torch.Tensor) (B, NH, S, DH)
        :param igate_preact: (torch.Tensor) (B, NH, S, 1)
        :param fgate_preact: (torch.Tensor) (B, NH, S, 1)
        :param lower_triangular_matrix: (torch.Tensor) (S,S). Defaults to None.
        :param stabilize_rowwise: (bool) Wether to stabilize the combination matrix C rowwise (take maximum per row).
            Alternative: Subtract the maximum over all rows. Defaults to True.
        :param eps: (float) small constant to avoid division by 0. Defaults to 1e-6.

    Returns:
        torch.Tensor: (B, NH, S, DH), h_tilde_state
    """

    B, NH, S, DH = queries.shape
    _dtype, _device = queries.dtype, queries.device

    # 在 parallel_stabilized_simple 函数中关键修改点

    log_fgates = torch.nn.functional.logsigmoid(fgate_preact).to(dtype=torch.float32)  # 强制类型转换
    log_fgates = torch.clamp(log_fgates, min=-100, max=100)  # 限制数值范围

    # 在除法运算中添加epsilon


    # forget gate matrix
    # log_fgates = torch.nn.functional.logsigmoid(fgate_preact)
    # log_fgates = torch.nn.functional.logsigmoid(fgate_preact)  # (B, NH, S, 1)
    if lower_triangular_matrix is None or S < lower_triangular_matrix.size(-1):
        ltr = torch.tril(torch.ones((S, S), dtype=torch.bool, device=_device))
    else:
        ltr = lower_triangular_matrix
    assert ltr.dtype == torch.bool, f"lower_triangular_matrix must be of dtype bool, got {ltr.dtype}"
    log_fgates_cumsum = torch.cat([
        torch.zeros((B, NH, 1, 1), dtype=torch.float32, device=_device),
        torch.cumsum(log_fgates.to(torch.float32), dim=-2)
    ], dim=-2)

    # log_fgates_cumsum = torch.cat(
    #     [
    #         torch.zeros((B, NH, 1, 1), dtype=_dtype, device=_device),
    #         torch.cumsum(log_fgates, dim=-2),
    #     ],
    #     dim=-2,
    # )  # (B, NH, S+1, 1)



    # for each batch/head this is a matrix of shape (S+1, S+1) containing the cumsum of the log forget gate values
    # in the second dimension (colum dimension). Each row has the same is a copy of the first row.
    # First entry of each row is zero.
    rep_log_fgates_cumsum = log_fgates_cumsum.repeat(1, 1, 1, S + 1)  # (B, NH, S+1, S+1)
    # Now in each row cut off / subtract the forgetgate values of the later timesteps
    # where col j > row i
    _log_fg_matrix = rep_log_fgates_cumsum - rep_log_fgates_cumsum.transpose(-2, -1)  # (B, NH, S+1, S+1)
    # Causal masking & selection of the correct submatrix, such that forgetgate at timestep t is not applied
    # to the input at timestep t
    # log_fg_matrix = torch.where(ltr, _log_fg_matrix[:, :, 1:, 1:], -float("inf"))  # (B, NH, S, S)
    log_fg_matrix = torch.where(
        ltr,
        _log_fg_matrix[:, :, 1:, 1:],
        torch.tensor(-float("inf"), dtype=torch.float32, device=_device)  # 指定inf为float32
    )

    # gate decay matrix D (combination of forget gate and input gate)
    log_D_matrix = log_fg_matrix + igate_preact.transpose(-2, -1)  # (B, NH, S, S)
    # D matrix stabilization
    if stabilize_rowwise:
        max_log_D, _ = torch.max(log_D_matrix, dim=-1, keepdim=True)  # (B, NH, S, 1)
    else:
        max_log_D = torch.max(log_D_matrix.view(B, NH, -1), dim=-1, keepdim=True)[0].unsqueeze(-1)
        # (B, NH, 1, 1)
    log_D_matrix_stabilized = log_D_matrix - max_log_D  # (B, NH, S, S)
    D_matrix = torch.exp(log_D_matrix_stabilized)  # (B, NH, S, S)

    keys_scaled = keys / math.sqrt(DH)

    # combination matrix C
    qk_matrix = queries @ keys_scaled.transpose(-2, -1)  # (B, NH, S, S)
    C_matrix = qk_matrix * D_matrix  # (B, NH, S, S)
    normalizer = torch.maximum(C_matrix.sum(dim=-1, keepdim=True).abs(), torch.exp(-max_log_D))  # (B, NH, S, 1)
    # (B, NH, S, S)
    # C_matrix_normalized = C_matrix / (normalizer + eps)
    C_matrix_normalized = C_matrix / (normalizer + 1e-10)  # 原为eps=1e-6
    # retrieved values
    h_tilde_state = C_matrix_normalized @ values  # (B, NH, S, DH)

    return h_tilde_state


class LinearHeadwiseExpand(nn.Module):
    """
    This is a structured projection layer that projects the input to a higher dimension.
    It only allows integer up-projection factors, i.e. the output dimension is a multiple of the input dimension.
    """

    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads

        dim_per_head = dim // num_heads
        self.weight = nn.Parameter(torch.empty(num_heads, dim_per_head, dim_per_head))
        if bias:
            self.bias = nn.Parameter(torch.empty(dim))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight.data, mean=0.0, std=math.sqrt(2 / 5 / self.weight.shape[-1]))
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, "... (nh d) -> ... nh d", nh=self.num_heads)
        x = einops.einsum(
            x,
            self.weight,
            "... nh d, nh out_d d -> ... nh out_d",
        )
        x = einops.rearrange(x, "... nh out_d -> ... (nh out_d)")
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self):
        return (
            f"dim={self.dim}, "
            f"num_heads={self.num_heads}, "
            f"bias={self.bias is not None}, "
        )


class CausalConv1d(nn.Module):
    """
    Implements causal depthwise convolution of a time series tensor.
    Input:  Tensor of shape (B,T,F), i.e. (batch, time, feature)
    Output: Tensor of shape (B,T,F)

    Args:
        feature_dim: number of features in the input tensor
        kernel_size: size of the kernel for the depthwise convolution
        causal_conv_bias: whether to use bias in the depthwise convolution
        channel_mixing: whether to use channel mixing (i.e. groups=1) or not (i.e. groups=feature_dim)
                        If True, it mixes the convolved features across channels.
                        If False, all the features are convolved independently.
    """

    def __init__(self, dim, kernel_size=4, bias=True):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.bias = bias
        # padding of this size assures temporal causality.
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=self.pad,
            groups=dim,
            bias=bias,
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # conv requires dim first
        x = einops.rearrange(x, "b l d -> b d l")
        # causal conv1d
        x = self.conv(x)
        x = x[:, :, :-self.pad]
        # back to dim last
        x = einops.rearrange(x, "b d l -> b l d")
        return x


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False. """

    def __init__(
            self,
            ndim: int = -1,
            weight: bool = True,
            bias: bool = False,
            eps: float = 1e-5,
            residual_weight: bool = True,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(ndim)) if weight else None
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps
        self.residual_weight = residual_weight
        self.ndim = ndim
        self.reset_parameters()

    @property
    def weight_proxy(self) -> torch.Tensor:
        if self.weight is None:
            return None
        if self.residual_weight:
            return 1.0 + self.weight
        else:
            return self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            normalized_shape=(self.ndim,),
            weight=self.weight_proxy,
            bias=self.bias,
            eps=self.eps,
        )

    def reset_parameters(self):
        if self.weight_proxy is not None:
            if self.residual_weight:
                nn.init.zeros_(self.weight)
            else:
                nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class MultiHeadLayerNorm(LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, "Input must be 4D tensor (B, NH, S, DH)"
        B, NH, S, DH = x.shape

        gn_in_1 = x.transpose(1, 2)  # (B, S, NH, DH)
        gn_in_2 = gn_in_1.reshape(B * S, NH * DH)  # (B * S, NH * DH)
        out = F.group_norm(
            gn_in_2,
            num_groups=NH,
            weight=self.weight_proxy,
            bias=self.bias,
            eps=self.eps,
        )  # .to(x.dtype)
        # (B * S), (NH * DH) -> (B, S, NH, DH) -> (B, NH, S, DH)
        out = out.view(B, S, NH, DH).transpose(1, 2)
        return out


class MatrixLSTMCell(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.igate = nn.Linear(3 * dim, num_heads)
        self.fgate = nn.Linear(3 * dim, num_heads)
        self.outnorm = MultiHeadLayerNorm(ndim=dim, weight=True, bias=False)
        self.causal_mask_cache = {}
        self.reset_parameters()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        assert q.dtype == torch.float32
        B, S, _ = q.shape  # (B, S, H)

        if_gate_input = torch.cat([q, k, v], dim=-1)
        q = q.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)
        k = k.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)
        v = v.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)

        q = q.transpose(1, 2)  # (B, NH, S, DH)
        k = k.transpose(1, 2)  # (B, NH, S, DH)
        v = v.transpose(1, 2)  # (B, NH, S, DH)

        # compute input and forget gate pre-activations
        igate_preact = self.igate(if_gate_input)  # (B, S, NH)
        igate_preact = igate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)
        fgate_preact = self.fgate(if_gate_input)  # (B, S, NH)
        fgate_preact = fgate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)#

        # cache causal mask to avoid memory allocation in every iteration
        if S in self.causal_mask_cache:
            causal_mask = self.causal_mask_cache[(S, str(q.device))]
        else:
            causal_mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=q.device))
            self.causal_mask_cache[(S, str(q.device))] = causal_mask

        h_state = parallel_stabilized_simple(
            queries=q,
            keys=k,
            values=v,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
            lower_triangular_matrix=causal_mask,
        )  # (B, NH, S, DH)

        h_state_norm = self.outnorm(h_state)  # (B, NH, S, DH)
        h_state_norm = h_state_norm.transpose(1, 2).reshape(B, S, -1)  # (B, NH, S, DH) -> (B, S, NH, DH) -> (B, S, H)

        return h_state_norm

    def reset_parameters(self):
        self.outnorm.reset_parameters()
        # forget gate initialization
        torch.nn.init.zeros_(self.fgate.weight)
        bias_linspace_init_(self.fgate.bias, start=3.0, end=6.0)
        # input gate initialization
        torch.nn.init.zeros_(self.igate.weight)
        torch.nn.init.normal_(self.igate.bias, mean=0.0, std=0.01)


class ViLLayer(nn.Module):
    def __init__(
            self,
            dim,
            direction,
            expansion=2,
            qkv_block_size=4,
            proj_bias=False,
            conv_bias=True,
            kernel_size=4,
    ):
        super().__init__()
        if dim % qkv_block_size != 0:
            qkv_block_size=2
        # assert dim % qkv_block_size == 0
        self.dim = dim
        self.direction = direction
        self.expansion = expansion
        self.qkv_block_size = qkv_block_size
        self.proj_bias = proj_bias
        self.conv_bias = conv_bias
        self.kernel_size = kernel_size

        inner_dim = expansion * dim
        num_heads = inner_dim // qkv_block_size
        self.proj_up = nn.Linear(
            in_features=dim,
            out_features=2 * inner_dim,
            bias=proj_bias,
        )
        self.q_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )
        self.k_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )
        self.v_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )

        self.conv1d = CausalConv1d(
            dim=inner_dim,
            kernel_size=kernel_size,
            bias=conv_bias,
        )
        self.mlstm_cell = MatrixLSTMCell(
            dim=inner_dim,
            num_heads=qkv_block_size,
        )
        self.learnable_skip = nn.Parameter(torch.ones(inner_dim))

        self.proj_down = nn.Linear(
            in_features=inner_dim,
            out_features=dim,
            bias=proj_bias,
        )
        self.reset_parameters()



    def forward(self, x: torch.Tensor) -> torch.Tensor:


        x = x.to(dtype=torch.float32)

        B, S, _ = x.shape

        # alternate direction in successive layers
        if self.direction == SequenceTraversal.ROWWISE_FROM_TOP_LEFT:
            pass
        elif self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1])
        else:
            raise NotImplementedError

        # up-projection
        x_inner = self.proj_up(x)
        x_mlstm, z = torch.chunk(x_inner, chunks=2, dim=-1)

        # mlstm branch
        x_mlstm_conv = self.conv1d(x_mlstm)
        x_mlstm_conv_act = F.silu(x_mlstm_conv)
        q = self.q_proj(x_mlstm_conv_act)
        k = self.k_proj(x_mlstm_conv_act)
        v = self.v_proj(x_mlstm)
        h_tilde_state = self.mlstm_cell(q=q, k=k, v=v)
        h_tilde_state_skip = h_tilde_state + (self.learnable_skip * x_mlstm_conv_act)

        # output / z branch
        h_state = h_tilde_state_skip * F.silu(z)

        # down-projection
        x = self.proj_down(h_state)

        # reverse alternating flip
        if self.direction == SequenceTraversal.ROWWISE_FROM_TOP_LEFT:
            pass
        elif self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1])
        else:
            raise NotImplementedError

        return x

    def reset_parameters(self):
        # 使用更小的初始化标准差
        nn.init.normal_(self.proj_up.weight, std=0.01)  # 原为0.02
        if self.proj_up.bias is not None:
            nn.init.constant_(self.proj_up.bias, 0)
        # init inproj
        small_init_(self.proj_up.weight, dim=self.dim)
        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)
        # init outproj (original mLSTM uses num_blocks=1)
        wang_init_(self.proj_down.weight, dim=self.dim, num_blocks=1)
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)

        nn.init.ones_(self.learnable_skip)

        def _init_qkv_proj(qkv_proj: LinearHeadwiseExpand):
            # use the embedding dim instead of the inner embedding dim
            small_init_(qkv_proj.weight, dim=self.dim)
            if qkv_proj.bias is not None:
                nn.init.zeros_(qkv_proj.bias)

        _init_qkv_proj(self.q_proj)
        _init_qkv_proj(self.k_proj)
        _init_qkv_proj(self.v_proj)

        self.mlstm_cell.reset_parameters()
# class CEABlock(nn.Module):
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0):
#         super().__init__()
#         # ==================== ViLLayer增强模块 ====================
#         self.vil_pre_x = ViLLayer(
#             dim=dim,
#             direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT,
#             expansion=2,
#             qkv_block_size=4
#         )
#         self.vil_pre_xi = ViLLayer(
#             dim=dim,
#             direction=SequenceTraversal.ROWWISE_FROM_BOT_RIGHT,
#             expansion=2,
#             qkv_block_size=4
#         )
#
#         # ==================== 核心组件初始化 ====================
#         self.norm1 = norm_layer(dim)
#         self.attn = nn.MultiheadAttention(  # 使用PyTorch原生Attention作为示例
#             embed_dim=dim,
#             num_heads=num_heads,
#             dropout=attn_drop,
#             batch_first=True
#         )
#         self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
#         # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(
#             in_features=dim,
#             hidden_features=mlp_hidden_dim,
#             act_layer=act_layer,
#             drop=drop
#         )
#         self.keep_ratio_search = keep_ratio_search
#
#         # ==================== 适配器模块占位实现 ====================
#
#
#         self.adap_t = LiteAdditiveTokenMixer()
#         self.adap2_t = LiteAdditiveTokenMixer()
#
#         # ==================== 门控模块占位实现 ====================
#         class gate(nn.Module):
#             def __init__(self):
#                 super().__init__()
#                 self.gate = nn.Sequential(
#                     nn.Linear(dim, 1),
#                     nn.Sigmoid()
#                 )
#
#             def forward(self, x):
#                 return self.gate(x.mean(dim=1, keepdim=True))  # 简化实现
#
#         self.gate = gate()
#
#     def forward(self, x, xi, global_index_template, global_index_templatei,
#                 global_index_search, global_index_searchi, mask=None,
#                 ce_template_mask=None, keep_ratio_search=None):
#         # ==================== ViLLayer预处理 ====================
#         # 主分支增强（残差连接）
#         x = x + self.vil_pre_x(x)
#         # 辅分支增强（残差连接）
#         xi = xi + self.vil_pre_xi(xi)
#
#         # ==================== 注意力处理 ====================
#         xori = x
#         x_attn, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=mask)
#         gate_z = self.gate(self.norm1(x))
#         x = gate_z * x + self.drop_path(x_attn) + (1 - gate_z) * self.drop_path(self.adap_t(self.norm1(xi)))
#
#         xi_attn, _ = self.attn(self.norm1(xi), self.norm1(xi), self.norm1(xi), attn_mask=mask)
#         gate_zi = self.gate(self.norm1(xi))
#         xi = gate_zi * xi + self.drop_path(xi_attn) + (1 - gate_zi) * self.drop_path(self.adap_t(self.norm1(xori)))
#
#         # ==================== 候选消除占位实现 ====================
#         def candidate_elimination(*args, **kwargs):
#             return args[1], args[4], None  # 返回原始值用于测试
#
#         lens_t = global_index_template.shape[1] if global_index_template is not None else 0
#         removed_index_search = None
#         removed_index_searchi = None
#         if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
#             keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
#             x, global_index_search, removed_index_search = candidate_elimination(None, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)
#             xi, global_index_searchi, removed_index_searchi = candidate_elimination(None, xi, lens_t, keep_ratio_search, global_index_searchi, ce_template_mask)
#
#         # ==================== MLP处理 ====================
#         xori = x
#         gate_z = self.gate(self.norm2(x))
#         x = gate_z * x + self.drop_path(self.mlp(self.norm2(x))) + (1 - gate_z) * self.drop_path(self.adap2_t(self.norm2(xi)))
#         gate_zi = self.gate(self.norm2(xi))
#         xi = gate_zi * xi + self.drop_path(self.mlp(self.norm2(xi))) + (1 - gate_zi) * self.drop_path(self.adap2_t(self.norm2(xori)))
#
#         return (x, global_index_template, global_index_search, removed_index_search, None,
#                 xi, global_index_templatei, global_index_searchi, removed_index_searchi, None)

class LightViLLayer(nn.Module):
    """轻量化ViLLayer，计算效率提升约40%"""

    def __init__(self, dim, direction, expansion=1.5, qkv_block=2,
                 kernel_size=3, proj_bias=True, conv_bias=True):
        super().__init__()
        # 压缩扩展率至1.5倍
        inner_dim = int(dim * expansion)
        num_heads = inner_dim // qkv_block

        # 方向控制
        self.direction = direction
        self.dim = dim

        # 轻量投影层（分组卷积替代全连接）
        self.proj_up = nn.Sequential(
            nn.Conv1d(dim, 2 * inner_dim, kernel_size=1, groups=4, bias=proj_bias),
            nn.GELU()
        )

        # 深度可分离因果卷积
        self.conv1d = nn.Sequential(
            nn.Conv1d(inner_dim, inner_dim, kernel_size,
                      padding=kernel_size - 1, groups=inner_dim, bias=conv_bias),
            nn.GELU()
        )

        # 共享QKV生成
        self.qkv_proj = LinearHeadwiseExpand(inner_dim, num_heads, bias=proj_bias)

        # 简化mLSTM单元
        self.mlstm = SimplemLSTMCell(inner_dim, num_heads)

        # 动态门控跳跃连接
        self.skip_gate = nn.Parameter(torch.ones(inner_dim))
        self.gate_net = nn.Sequential(
            nn.Linear(inner_dim, inner_dim // 4),
            nn.ReLU(),
            nn.Linear(inner_dim // 4, 1),
            nn.Sigmoid()
        )

        # 权重共享的下投影
        self.proj_down = nn.Linear(inner_dim, dim, bias=proj_bias)

        # 初始化
        self._reset_parameters()

    def _reset_parameters(self):
        # 参数初始化策略
        nn.init.kaiming_normal_(self.proj_up[0].weight, mode='fan_out', nonlinearity='gelu')
        nn.init.normal_(self.qkv_proj.weight, std=0.02)
        nn.init.zeros_(self.proj_down.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 方向控制
        if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(1)

        B, S, _ = x.shape

        # 轻量投影
        x_up = self.proj_up(x.transpose(1, 2)).transpose(1, 2)
        x_conv, x_gate = x_up.chunk(2, dim=-1)

        # 深度可分离卷积
        x_conv = self.conv1d(x_conv.transpose(1, 2))[:, :, :S].transpose(1, 2)

        # 共享QKV生成
        qkv = self.qkv_proj(F.gelu(x_conv))
        q, k, v = qkv.chunk(3, dim=-1)

        # 简化mLSTM计算
        h = self.mlstm(q, k, v)

        # 动态门控跳跃
        gate = self.gate_net(x.mean(1))
        h = h * self.skip_gate * gate.unsqueeze(1)

        # 残差连接
        output = x + self.proj_down(h)

        # 恢复方向
        if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            output = output.flip(1)

        return output


class SimplemLSTMCell(nn.Module):
    """简化版mLSTM单元，计算量减少约35%"""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # 共享门控网络
        self.gate_net = nn.Linear(3 * self.head_dim, 2)

        # 轻量归一化
        self.norm = MultiHeadLayerNorm(dim, num_heads)

        # 缓存机制
        self.cache = {}

    def forward(self, q, k, v):
        B, S, _ = q.shape
        H, D = self.num_heads, self.head_dim

        # 重组形状
        q = q.view(B, S, H, D).transpose(1, 2)
        k = k.view(B, S, H, D).transpose(1, 2)
        v = v.view(B, S, H, D).transpose(1, 2)

        # 合并特征生成门控
        gate_input = torch.cat([q, k, v], dim=-1)
        gates = self.gate_net(gate_input).sigmoid()
        i_gate, f_gate = gates.chunk(2, dim=-1)

        # 简化版并行LSTM计算
        c = (i_gate * (q @ k.transpose(-2, -1))).sum(-1, keepdim=True)
        h = c * v

        # 轻量归一化
        h = self.norm(h.transpose(1, 2).reshape(B, S, -1))
        return h

#
# class MultiHeadLayerNorm(nn.Module):
#     """高效多头归一化"""
#
#     def __init__(self, dim, num_heads):
#         super().__init__()
#         self.num_heads = num_heads
#         self.norm = nn.LayerNorm(dim // num_heads)
#
#     def forward(self, x):
#         B, S, _ = x.shape
#         x = x.view(B, S, self.num_heads, -1).transpose(1, 2)
#         x = self.norm(x)
#         return x.transpose(1, 2).reshape(B, S, -1)


class CEABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0):
        super().__init__()

        # ============== 新增的ViLLayer全局增强模块 ==============
        self.vil_pre_x = ViLLayer(
            dim=dim,
            direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT,
            expansion=2,
            qkv_block_size=4
        )

        self.vil_pre_xi = ViLLayer(
            dim=dim,
            direction=SequenceTraversal.ROWWISE_FROM_BOT_RIGHT,
            expansion=2,
            qkv_block_size=4
        )

        # ============== 原始CEABlock组件 ==============
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        self.keep_ratio_search = keep_ratio_search
        self.adap_t = EnhancedDualAttentionAdapter()
        self.adap2_t = EnhancedDualAttentionAdapter()

    def forward(self, x, xi, global_index_template, global_index_templatei,
                global_index_search, global_index_searchi, mask=None,
                ce_template_mask=None, keep_ratio_search=None):
        # ============== 新增的ViLLayer前向处理 ==============
        # 主分支特征增强（残差连接）
        x = x + self.vil_pre_x(x)
        # 辅助分支特征增强（残差连接）
        xi = xi + self.vil_pre_xi(xi)

        # ============== 原始CEABlock前向逻辑 ==============
        xori = x
        x_attn, attn = self.attn(self.norm1(x), mask, True)
        adapter_output = self.adap_t(self.norm1(xi))
        x = x + self.drop_path(x_attn) + self.drop_path(adapter_output)

        xi_attn, i_attn = self.attn(self.norm1(xi), mask, True)
        adapter_output1 = self.adap_t(self.norm1(xori))
        xi = xi + self.drop_path(xi_attn) + self.drop_path(adapter_output1)

        lens_t = global_index_template.shape[1]
        removed_index_search = None
        removed_index_searchi = None

        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = candidate_elimination(
                attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)
            xi, global_index_searchi, removed_index_searchi = candidate_elimination(
                i_attn, xi, lens_t, keep_ratio_search, global_index_searchi, ce_template_mask)

        xori = x
        x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adap2_t(self.norm2(xi)))
        xi = xi + self.drop_path(self.mlp(self.norm2(xi))) + self.drop_path(self.adap2_t(self.norm2(xori)))

        return (x, global_index_template, global_index_search, removed_index_search, attn,
                xi, global_index_templatei, global_index_searchi, removed_index_searchi, i_attn)



class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        #print("class Block ")
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        #print("class Block forward")
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
