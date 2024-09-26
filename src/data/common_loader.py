import copy
import json
import os
import pickle
import sys
from itertools import repeat
from typing import List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.datasets.base_datasets import get_pointcloud_loader

from .augmentation import get_augmentations_from_list


def read_json_file(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

class CommonLoader(Dataset):
    def __init__(self, pickle_loc, dataset_name, base_ds, do_augment=False, train=True):
        with open(pickle_loc, 'rb') as file:
            self.data_dict = pickle.load(file)

        ### Selecting only the database for training ####
        if train:
            self.data_dict = self.data_dict[1]
        else:
            self.data_dict = self.data_dict[0]
        #################################################
        self.all_file_loc = []
        self.poses = []
        ############# For Southbay ##################################
        for i in range(len(self.data_dict)):
            self.all_file_loc.append(base_ds + self.data_dict[i][1])
            self.poses.append(self.data_dict[i][3])

        self.poses = np.array(self.poses)
        self.voxel_size = 0.5
        self.pc_loader = get_pointcloud_loader(dataset_name)
        # self.mean_lidar_center = self._compute_mean_lidar_center()
        if dataset_name == 'robotcar':
            self.mean_lidar_center = torch.Tensor([5735480.0338361, 620014.6556220, -110.6479860])
            self.std_lidar_center = torch.Tensor([391.3828435, 253.5759080, 2.6142053])
        elif dataset_name == 'vReLoc':
            self.mean_lidar_center = torch.Tensor([0, 0, 0])
            self.std_lidar_center = torch.Tensor([1, 1, 1])
        elif dataset_name == 'dcc':
            self.mean_lidar_center = torch.Tensor([3.55816349e+05, 4.02697857e+06, 1.92038750e+01])
            self.std_lidar_center = torch.Tensor([197.21883464, 146.17630603, 0.35582199])

        self.augmentations = get_augmentations_from_list(['Jitter', 'Scale', 'Shift'])
        self.do_augment = do_augment

    def _compute_mean_lidar_center(self):
        mean_lidar_center = torch.zeros((3,))

        for pose in self.poses:
            # Get the translation component.
            mean_lidar_center += pose[0:3, 3]

        # Avg.
        mean_lidar_center /= len(self.poses)
        return mean_lidar_center

    def data_prepare(self, xyzr, pose ,voxel_size=np.array([0.1, 0.1, 0.1])):

        range_val = np.linalg.norm(xyzr[:, :3], axis=1)
        range_filter = np.logical_and(range_val > 0.1, range_val < 80)
        xyzr = xyzr[range_filter]

        if self.do_augment:
            p = np.random.rand(1)
            if p>0.3:
                aug = np.random.randint(len(self.augmentations))
                xyzr = self.augmentations[aug].apply(xyzr)

        lidar_pc = copy.deepcopy(xyzr)

        coords = np.round(lidar_pc[:, :3] / voxel_size)
        coords_min = coords.min(0, keepdims=1)
        coords -= coords_min
        feats = lidar_pc

        hash_vals, _, uniq_idx = self.sparse_quantize(coords, return_index=True, return_hash=True)
        coord_voxel, feat = coords[uniq_idx], feats[uniq_idx]
        coord = copy.deepcopy(feat[:,:3])

        coord = torch.FloatTensor(coord)
        feat = torch.FloatTensor(feat)
        coord_voxel = torch.LongTensor(coord_voxel)

        ### Find pose of each point in world frame #####
        coord_homogeneous = torch.cat([coord, torch.ones(coord.shape[0], 1, device=coord.device)], dim=1)  # (N, 4)
        coord_homogeneous = coord_homogeneous.unsqueeze(-1)  # (N, 4, 1)
        pose_expanded = torch.Tensor(pose).unsqueeze(0).expand(coord.shape[0], -1, -1)  # (N, 4, 4)
        transformed_coordinates = torch.bmm(pose_expanded, coord_homogeneous)  # (N, 4, 1)
        point_poses = transformed_coordinates.squeeze(-1)[:, :3]  # (N, 3)
        # print(coord_voxel.shape)
        return coord_voxel, coord, feat, point_poses

    def sparse_quantize(self, coords,
                    voxel_size: Union[float, Tuple[float, ...]] = 1,
                    *,
                    return_index: bool = False,
                    return_inverse: bool = False,
                    return_hash: bool = False) -> List[np.ndarray]:
        if isinstance(voxel_size, (float, int)):
            voxel_size = tuple(repeat(voxel_size, 3))
        assert isinstance(voxel_size, tuple) and len(voxel_size) == 3

        voxel_size = np.array(voxel_size)
        coords = np.floor(coords / voxel_size).astype(np.int32)

        hash_vals, indices, inverse_indices = np.unique(self.ravel_hash(coords),
                                                return_index=True,
                                                return_inverse=True)
        coords = coords[indices]

        if return_hash: outputs = [hash_vals, coords]
        else: outputs = [coords]

        if return_index:
            outputs += [indices]
        if return_inverse:
            outputs += [inverse_indices]
        return outputs[0] if len(outputs) == 1 else outputs


    def ravel_hash(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 2, x.shape

        x -= np.min(x, axis=0)
        x = x.astype(np.uint64, copy=False)
        xmax = np.max(x, axis=0).astype(np.uint64) + 1

        h = np.zeros(x.shape[0], dtype=np.uint64)
        for k in range(x.shape[1] - 1):
            h += x[:, k]
            h *= xmax[k + 1]
        h += x[:, -1]
        return h

    def __len__(self):
        return len(self.all_file_loc)

    def read_pcd_file(self,filename):
        xyzr = self.pc_loader(filename)
        return xyzr

    def __getitem__(self, idx):
        filename = self.all_file_loc[idx]
        pose = self.poses[idx]
        xyzr = self.read_pcd_file(filename)
        coords, xyz, feats, point_poses = self.data_prepare(xyzr,pose,voxel_size = np.array([self.voxel_size,self.voxel_size,self.voxel_size]))
        scene_pose = pose
        return coords, xyz, feats, point_poses, scene_pose

