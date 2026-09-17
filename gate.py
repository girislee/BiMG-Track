import torch
from torch import nn
import timm
import math
from lib.models.bat.utils import token2feature


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


class gate(nn.Module):
    def __init__(self):
        super().__init__()

        self.gate_conv=nn.Linear(768,768)
        self.bn=nn.BatchNorm2d(num_features=768)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        # x=token2feature(x)
        x_conv=self.gate_conv(x)
        # x_bn=self.bn(x_conv)
        x_sigmoid=self.sigmoid(x_conv)
        return x_sigmoid


