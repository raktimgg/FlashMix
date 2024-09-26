import argparse
import copy
import csv
import glob
import math
import os
import pickle
import shutil
import sys

import natsort
import numpy as np
import tqdm
import transforms3d
from scipy.spatial.transform import Rotation as R

FAULTY_POINTCLOUDS = [1566279795718079314, 1567496784952532897]

def find_nearest_ndx(ts, timestamps):
    ndx = np.searchsorted(timestamps, ts)
    if ndx == 0:
        return ndx
    elif ndx == len(timestamps):
        return ndx - 1
    else:
        assert timestamps[ndx-1] <= ts <= timestamps[ndx]
        if ts - timestamps[ndx-1] < timestamps[ndx] - ts:
            return ndx - 1
        else:
            return ndx

def read_lidar_poses(poses_filepath: str, lidar_filepath: str, pose_time_tolerance: float = 1.):
    # Read global poses from .csv file and link each lidar_scan with the nearest pose
    # threshold: threshold in seconds
    # Returns a dictionary with (4, 4) pose matrix indexed by a timestamp (as integer)

    with open(poses_filepath, "r") as h:
        txt_poses = h.readlines()

    n = len(txt_poses)
    system_timestamps = np.zeros((n,), dtype=np.int64)
    poses = np.zeros((n, 4, 4), dtype=np.float64)       # 4x4 pose matrix

    for ndx, pose in enumerate(txt_poses):
        # Split by comma and remove whitespaces
        temp = [e.strip() for e in pose.split(',')]
        assert len(temp) == 13, f'Invalid line in global poses file: {temp}'
        system_timestamps[ndx] = int(temp[0])
        poses[ndx] = np.array([[float(temp[1]), float(temp[2]), float(temp[3]), float(temp[4])],
                               [float(temp[5]), float(temp[6]), float(temp[7]), float(temp[8])],
                               [float(temp[9]), float(temp[10]), float(temp[11]), float(temp[12])],
                               [0., 0., 0., 1.]])

    # Ensure timestamps and poses are sorted in ascending order
    sorted_ndx = np.argsort(system_timestamps, axis=0)
    system_timestamps = system_timestamps[sorted_ndx]
    poses = poses[sorted_ndx]

    # List LiDAR scan timestamps
    all_lidar_timestamps = [int(os.path.splitext(f)[0]) for f in os.listdir(lidar_filepath) if
                            os.path.splitext(f)[1] == '.bin']
    all_lidar_timestamps.sort()

    lidar_timestamps = []
    lidar_poses = []
    count_rejected = 0

    for ndx, lidar_ts in enumerate(all_lidar_timestamps):
        # Skip faulty point clouds
        if lidar_ts in FAULTY_POINTCLOUDS:
            # print(lidar_ts)
            continue

        # Find index of the closest timestamp
        closest_ts_ndx = find_nearest_ndx(lidar_ts, system_timestamps)
        delta = abs(system_timestamps[closest_ts_ndx] - lidar_ts)
        # Timestamp is in nanoseconds = 1e-9 second
        if delta > pose_time_tolerance * 1000000000:
            # Reject point cloud without corresponding pose
            count_rejected += 1
            continue

        lidar_timestamps.append(lidar_ts)
        lidar_poses.append(poses[closest_ts_ndx])

    lidar_timestamps = np.array(lidar_timestamps, dtype=np.int64)
    lidar_poses = np.array(lidar_poses, dtype=np.float64)     # (northing, easting) position

    print(f'{len(lidar_timestamps)} scans with valid pose, {count_rejected} rejected due to unknown pose')
    return lidar_timestamps, lidar_poses


def find_euc_dist(T1,T2):
    x1,y1,z1 = T1[0], T1[1], T1[2]
    x2,y2,z2 = T2[0], T2[1], T2[2]
    dist = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
    # dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    return dist


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create RobotCar pickle file')
    parser.add_argument(
        "-b",
        "--data_dir",
        type=str,
        default='/data/raktim/Datasets/Mulran/DCC/',
        help="data directory",
    )
    parser.add_argument(
        "-s",
        "--save_dir",
        type=str,
        default='/data/raktim/Projects/APR/Absolute-Pose-Regression/src/utils/',
        help="pickle save directory",
    )
    parser.add_argument(
        "-t",
        "--d_thresh",
        type=int,
        default=0,
        help="distnace threshold",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    save_dir = args.save_dir
    d_thresh = args.d_thresh
    
    basedir = '/data/raktim/Datasets/Mulran/DCC/'

    train_sequences = ['DCC1', 'DCC2']
    # train_sequences = []

    test_sequences = ['DCC3']
    # test_sequences = []

    positions = []

    train_pckle_file = []
    for sequence in train_sequences:
        print("Sequence", sequence)
        sequence_path = basedir + '/' + sequence
        lidar_path = sequence_path + '/Ouster'
        relative_sequence = sequence + '/Ouster'
        pose_path = sequence_path + '/global_pose.csv'


        lidar_timestamps, lidar_poses = read_lidar_poses(pose_path, lidar_path)

        position_old = [0,0,0]
        for i in tqdm.tqdm(range(len(lidar_poses))):
            pose = lidar_poses[i]

            position = pose[:3,3]
            positions.append(position.tolist())

            if find_euc_dist(position, position_old)<d_thresh:
                continue
            position_old = copy.deepcopy(position)
            transformation_matrix = pose
            name = os.path.join(relative_sequence,str(lidar_timestamps[i])+'.bin')
            time = lidar_timestamps[i]
            position2d = np.array([transformation_matrix[0,3], transformation_matrix[1,3]])
            train_pckle_file.append([time,name,position2d,transformation_matrix])

    print(np.mean(positions,0))
    print(np.std(positions,0))


    test_pckle_file = []
    for sequence in test_sequences:
        print("Sequence", sequence)
        sequence_path = basedir + '/' + sequence
        lidar_path = sequence_path + '/Ouster'
        relative_sequence = sequence + '/Ouster'
        pose_path = sequence_path + '/global_pose.csv'


        lidar_timestamps, lidar_poses = read_lidar_poses(pose_path, lidar_path)

        position_old = [0,0,0]
        for i in tqdm.tqdm(range(len(lidar_poses))):
            pose = lidar_poses[i]

            position = pose[:3,3]

            if find_euc_dist(position, position_old)<d_thresh:
                continue
            position_old = copy.deepcopy(position)
            transformation_matrix = pose
            name = os.path.join(relative_sequence,str(lidar_timestamps[i])+'.bin')
            time = lidar_timestamps[i]
            position2d = np.array([transformation_matrix[0,3], transformation_matrix[1,3]])
            test_pckle_file.append([time,name,position2d,transformation_matrix])

    pckle_loc = save_dir+'dcc_og.pkl'

    print(len(train_pckle_file), len(test_pckle_file))
    pckle_file = [test_pckle_file, train_pckle_file]
    with open(pckle_loc, 'wb') as file:
        pickle.dump(pckle_file, file)
