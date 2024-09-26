import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DynaMixerOp(nn.Module):
    def __init__(self, dim, seq_len, num_head, reduced_dim=2):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.num_head = num_head
        self.reduced_dim = reduced_dim
        self.out = nn.Linear(dim, dim)
        self.compress = nn.Linear(dim, num_head * reduced_dim)
        self.generate = nn.Linear(seq_len * reduced_dim, seq_len * seq_len)
        self.activation = nn.Softmax(dim=-2)

    def forward(self, x):
        B, L, C = x.shape
        weights = self.compress(x).reshape(B, L, self.num_head, self.reduced_dim).permute(0, 2, 1, 3).reshape(B, self.num_head, -1)
        weights = self.generate(weights).reshape(B, self.num_head, L, L)
        weights = self.activation(weights)
        x = x.reshape(B, L, self.num_head, C//self.num_head).permute(0, 2, 3, 1)
        x = torch.matmul(x, weights)
        x = x.permute(0, 3, 1, 2).reshape(B, L, C)
        x = self.out(x)
        return x


class DynaMixerBlock(nn.Module):
    def __init__(self, dim, resolution=32, num_head=8, reduced_dim=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.resolution = resolution
        self.num_head = num_head
        self.mix_h = DynaMixerOp(dim, resolution, self.num_head, reduced_dim=reduced_dim)
        self.mix_w = DynaMixerOp(dim, resolution, self.num_head, reduced_dim=reduced_dim)
        self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)
        self.reweight = Mlp(dim, dim // 4, dim * 3)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        h = self.mix_h(x)
        w = self.mix_w(x)
        c = self.mlp_c(x)

        a = (h + w + c).mean(1)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2)

        x = h * a[0] + w * a[1] + c * a[2]

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class DynaMixerLayer(nn.Module):
    def __init__(self, num_token, token_dim, mlp_ratio):
        super().__init__()
        self.expanded_dim_t = int(num_token * mlp_ratio)
        self.mix_t = DynaMixerBlock(dim=token_dim, resolution=num_token, num_head=1, reduced_dim=2)

        self.expanded_dim_c = int(token_dim * mlp_ratio)
        self.mix_c = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, self.expanded_dim_c),
            nn.GELU(),
            nn.Linear(self.expanded_dim_c , token_dim),
        )
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x + self.mix_t(x)
        return x + self.mix_c(x)
