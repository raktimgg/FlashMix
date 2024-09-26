import os
import struct
from typing import List

import numpy as np
import torch
from datasets.mulran.utils import in_test_split, in_train_split, read_lidar_poses
from sklearn.neighbors import KDTree
from torch.utils.data import ConcatDataset, Dataset

from misc.point_clouds import PointCloudLoader


def load_velodyne_binary(velodyne_bin_path):
    """Decode a binary Velodyne example (of the form '<timestamp>.bin')
    Args:
        example_path (AnyStr): Oxford Radar RobotCar Dataset binary Velodyne pointcloud example path
    Returns:
        ptcld (np.ndarray): XYZI pointcloud from the binary Velodyne data Nx4
    Notes:
        - The pre computed points are *NOT* motion compensated.
        - Converting a raw velodyne scan to pointcloud can be done using the
            `velodyne_ranges_intensities_angles_to_pointcloud` function.
    """
    ext = os.path.splitext(velodyne_bin_path)[1]
    if ext != ".bin":
        raise RuntimeError("Velodyne binary pointcloud file should have `.bin` extension but had: {}".format(ext))
    if not os.path.isfile(velodyne_bin_path):
        raise FileNotFoundError("Could not find velodyne bin example: {}".format(velodyne_bin_path))
    data = np.fromfile(velodyne_bin_path, dtype=np.float32)
    ptcld = data.reshape((4, -1))
    return ptcld


class vReLocPointCloudLoader(PointCloudLoader):
    def set_properties(self):
        # Set point cloud properties, such as ground_plane_level.
        self.ground_plane_level = -100  # ground plane not removed because points are already very sparse

    def read_pc(self, file_pathname: str) -> torch.Tensor:
        ptcld     = load_velodyne_binary(file_pathname)  # (4, N)
        pc      = ptcld[:3].transpose()
        # print(pc.shape)
        return pc
