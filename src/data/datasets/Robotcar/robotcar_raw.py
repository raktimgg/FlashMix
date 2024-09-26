import os
from typing import List

import numpy as np
import torch
from datasets.mulran.utils import in_test_split, in_train_split, read_lidar_poses
from sklearn.neighbors import KDTree
from torch.utils.data import ConcatDataset, Dataset
import struct

from misc.point_clouds import PointCloudLoader
from .robotcar_sdk.python.velodyne import load_velodyne_binary
from .robotcar_sdk.python.interpolate_poses import interpolate_ins_poses, interpolate_vo_poses
from .robotcar_sdk.python.transform import build_se3_transform


class RobotcarPointCloudLoader(PointCloudLoader):
    def set_properties(self):
        # Set point cloud properties, such as ground_plane_level.
        self.ground_plane_level = -100  # ground plane not removed because points are already very sparse

    def read_pc(self, file_pathname: str) -> torch.Tensor:
        ptcld     = load_velodyne_binary(file_pathname)  # (4, N)
        pc      = ptcld[:3].transpose()
        # print(pc.shape)
        return pc