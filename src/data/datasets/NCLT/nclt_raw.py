import os
import struct
from typing import List

import numpy as np
import torch
from datasets.mulran.utils import in_test_split, in_train_split, read_lidar_poses
from sklearn.neighbors import KDTree
from torch.utils.data import ConcatDataset, Dataset

from misc.point_clouds import PointCloudLoader


def convert_nclt(x_s, y_s, z_s): # 输入点云转换函数
    # 文档种提供的转换函数
    # 原文档返回为x, y, z，但在绘制可视化图时z取负，此处先取负
    scaling = 0.005 # 5 mm
    offset = -100.0

    x = x_s * scaling + offset
    y = y_s * scaling + offset
    z = z_s * scaling + offset

    return x, y, z

class NCLTPointCloudLoader(PointCloudLoader):
    def set_properties(self):
        # Set point cloud properties, such as ground_plane_level.
        self.ground_plane_level = 0.4 # no ground plane removal
        self.remove_ground_plane = False

    def read_pc(self, file_pathname: str) -> torch.Tensor:
        # binary = np.fromfile(file_pathname, dtype=np.int16)
        # x = np.ascontiguousarray(binary[::4])
        # y = np.ascontiguousarray(binary[1::4])
        # z = np.ascontiguousarray(binary[2::4])
        # x = x.astype(np.float32).reshape(-1, 1)
        # y = y.astype(np.float32).reshape(-1, 1)
        # z = z.astype(np.float32).reshape(-1, 1)
        # x, y, z = convert_nclt(x, y, z)
        # pc = np.concatenate([x, y, -z], axis=1)
        # print(pc.shape)
        # print(np.max(pc,axis=0))
        # print(np.min(pc,axis=0))

        f_bin = open(file_pathname, "rb")
        hits = []
        while True:
            x_str = f_bin.read(2)
            if x_str == b'': #eof
                break
            x = struct.unpack('<H', x_str)[0]
            y = struct.unpack('<H', f_bin.read(2))[0]
            z = struct.unpack('<H', f_bin.read(2))[0]
            i = struct.unpack('B', f_bin.read(1))[0]
            l = struct.unpack('B', f_bin.read(1))[0]

            x, y, z = convert_nclt(x, y, z)

            hits += [[x, y, z]]

        f_bin.close()

        pc = np.array(hits)


        return pc
