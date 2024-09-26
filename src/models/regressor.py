from functools import partial

# import flash_attn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from flash_attn import flash_attn_func
from torch.autograd import Variable

from models.dynamixer import DynaMixerLayer
from models.salad import SALAD
from utils.pointnet_util import PointNetSetAbstraction


def linear(in_dim, out_dim, bias=True):
    return nn.Sequential(
        nn.Linear(
            in_dim, out_dim, bias),
        nn.ReLU(inplace=True),
    )

class FeatureMixerLayer(nn.Module):
    def __init__(self, num_token, token_dim, mlp_ratio):
        super().__init__()
        self.expanded_dim_t = int(num_token * mlp_ratio)
        self.mix_t = nn.Sequential(
            nn.LayerNorm(token_dim),
            Rearrange('b c n -> b n c'),
            nn.Linear(num_token, self.expanded_dim_t),
            nn.GELU(),
            nn.Linear(self.expanded_dim_t, num_token),
            Rearrange('b n c -> b c n'),
        )

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


class Regressor(nn.Module):
    """
    Scene Region Classification.
    """
    def __init__(self, args, N=1024, C=32):

        super(Regressor, self).__init__()

        channels = 1024

        mixer_channels = args['regressor']['num_mixer_channels']
        prl = args['regressor']['pose_regressor_layers']
        gdim_scale = args['regressor']['global_desc_dim_scale']

        self.mix = nn.Sequential(
            *[
                FeatureMixerLayer(N, C, mixer_channels)
                for _ in range(mixer_channels)
            ],
            nn.LayerNorm(C),
            linear(C, channels)
        )

        self.decoder = MLPDecoder(channels, [channels] * prl)
        self.fct = nn.Linear(channels, 3)
        self.fcq = nn.Linear(channels, 3)

        self.emb_prejector = nn.Sequential(
            nn.Linear(channels, int(gdim_scale * channels)),
            nn.ReLU(),
            nn.Linear(int(gdim_scale * channels), int(gdim_scale * channels)),
        )

    def forward(self, local_features, xyz=None, return_emb=False):
        emb = self.mix(local_features)
        emb = emb.mean(1)

        projected_emb = self.emb_prejector(emb)

        y = self.decoder(emb)
        t = self.fct(y)
        q = self.fcq(y)

        if return_emb:
            return t, q, None, None, projected_emb
        return t, q, None, None

    def is_cuda(self):
        return next(self.parameters()).is_cuda


class MLPDecoder(nn.Module):
    def __init__(self, in_channel, mlp):
        super(MLPDecoder, self).__init__()
        self.mlp_fcs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.activataions = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_fcs.append(nn.Linear(last_channel, out_channel))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            self.dropouts.append(nn.Dropout(0.08))
            self.activataions.append(nn.ReLU(inplace=True))
            last_channel = out_channel

    def forward(self, x):
        for i, fc in enumerate(self.mlp_fcs):
            bn = self.mlp_bns[i]
            act = self.activataions[i]
            # x  = F.relu(bn(fc(x)))  # [B, D]
            x = act(bn(fc(x)))
            x = self.dropouts[i](x)

        return x