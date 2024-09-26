import os
import struct
from typing import List

import numpy as np
import torch
from datasets.mulran.utils import in_test_split, in_train_split, read_lidar_poses
from sklearn.neighbors import KDTree
from torch.utils.data import ConcatDataset, Dataset

from misc.point_clouds import PointCloudLoader


class CRRLGo2PointCloudLoader(PointCloudLoader):
    def set_properties(self):
        # Set point cloud properties, such as ground_plane_level.
        self.ground_plane_level = -100  # ground plane not removed because points are already very sparse

    def read_pc(self, file_pathname: str):
        xyz = np.load(file_pathname)
        pc = np.stack([xyz['x'], xyz['y'], xyz['z']]).transpose()
        # print(pc.shape)
        return pc
