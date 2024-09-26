import os
import sys
from datetime import datetime
from time import time

import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.regressor import Regressor
from models.SphereFormer.model.unet_spherical_transformer import Semantic
from utils.misc_utils import read_yaml_config


class FeatureExtractor(nn.Module):
    def __init__(self,voxel_sz):
        super(FeatureExtractor, self).__init__()
        config = read_yaml_config(os.path.join(os.path.dirname(__file__),'../config/salsa_model.yaml'))
        self.feature_dim = config['feat_extractor']['feature_dim']
        patch_size = config['feat_extractor']['patch_size']
        voxel_size = [voxel_sz, voxel_sz, voxel_sz]
        patch_size = np.array([voxel_size[i] * patch_size for i in range(3)]).astype(np.float32)
        window_size = patch_size * 6
        self.feature_encoder = Semantic(input_c=config['feat_extractor']['input_c'],
            m=config['feat_extractor']['m'],
            classes=self.feature_dim,
            block_reps=config['feat_extractor']['block_reps'],
            block_residual=True,
            layers=config['feat_extractor']['layers'],
            window_size=window_size,
            window_size_sphere=np.array(config['feat_extractor']['window_size_sphere']),
            quant_size=window_size/24,
            quant_size_sphere= np.array(config['feat_extractor']['window_size_sphere'])/24,
            rel_query=True,
            rel_key=True,
            rel_value=True,
            drop_path_rate=config['feat_extractor']['drop_path_rate'],
            window_size_scale=config['feat_extractor']['window_size_scale'],
            grad_checkpoint_layers=[],
            sphere_layers=config['feat_extractor']['sphere_layers'],
            a=config['feat_extractor']['a'],
        )

    def forward(self, coord, xyz, feat, batch, save_attn_weights=False):
        batch_shape = batch[-1]+1
        coord = torch.cat([batch.unsqueeze(-1), coord], -1)
        spatial_shape = np.clip((coord.max(0)[0][1:] + 1).cpu().numpy(), 128, None)

        sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, batch_shape)

        local_features = self.feature_encoder(sinput, xyz, batch)
        return local_features

class SceneRegressor(nn.Module):

    def __init__(self, args, N=1024, C=32):
        super(SceneRegressor, self).__init__()
        self.regressor = Regressor(args, N, C)

    def forward(self, local_features, xyz, return_emb=False):
        res = self.regressor(local_features, xyz, return_emb)
        return res
