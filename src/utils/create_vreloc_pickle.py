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
        default='/data/raktim/Datasets/vReLoc',
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
    
    basedir = data_dir

    bad_poses = [basedir+'/seq-12/frame-000264.pose.txt',
                basedir+'/seq-12/frame-000265.pose.txt',
                basedir+'/seq-12/frame-000267.pose.txt',
                basedir+'/seq-12/frame-000268.pose.txt',
                basedir+'/seq-12/frame-000283.pose.txt',
                basedir+'/seq-15/frame-000511.pose.txt',
                basedir+'/seq-15/frame-000282.pose.txt',
                basedir+'/seq-15/frame-000351.pose.txt',
                basedir+'/seq-15/frame-000354.pose.txt',
                basedir+'/seq-15/frame-000355.pose.txt',
                basedir+'/seq-15/frame-000359.pose.txt',
                basedir+'/seq-15/frame-000360.pose.txt',
                basedir+'/seq-15/frame-000363.pose.txt',
                basedir+'/seq-15/frame-000367.pose.txt'
                ]

    train_sequences = ['seq-03', 'seq-12', 'seq-15', 'seq-16']
    # train_sequences = ['seq-03']
    # train_sequences = ['seq-12']
    # train_sequences = ['seq-15']
    # train_sequences = ['seq-16']
    # train_sequences = []

    test_sequences = ['seq-05', 'seq-06', 'seq-07', 'seq-14']
    # test_sequences = ['seq-05']
    # test_sequences = ['seq-06']
    # test_sequences = ['seq-07']
    # test_sequences = ['seq-14']
    # test_sequences = []

    y_thresh = 0.0  # 0 degres

    train_pckle_file = []
    for sequence in train_sequences:
        print("Sequence", sequence)
        sequence_path = basedir + '/' + sequence
        relative_sequence = sequence + '/'
        p_filenames = natsort.natsorted([n for n in os.listdir(sequence_path) if n.find('pose') >= 0])
        position_old = [0,0,0]
        for i in tqdm.tqdm(range(len(p_filenames))):
            pose_filename = os.path.join(sequence_path, p_filenames[i])
            if pose_filename in bad_poses:
                print('Removing bad pose', pose_filename)

            time = pose_filename.split('/')[-1].split('.')[0]
            name = relative_sequence + str(time) + '.bin'
            pose = np.loadtxt(pose_filename, delimiter=',')
            position = pose[:3,3]
            yaw = np.arctan2(pose[1, 0], pose[0, 0])
            if i==0:
                yaw_old = yaw
                yaw_diff = 1000
            else:
                yaw_diff = np.abs(yaw-yaw_old)
                yaw_old = yaw

            if find_euc_dist(position, position_old)<d_thresh and yaw_diff<y_thresh:
                continue
            position_old = copy.deepcopy(position)
            transformation_matrix = pose
            position2d = np.array([transformation_matrix[0,3], transformation_matrix[1,3]])
            train_pckle_file.append([time,name,position2d,transformation_matrix])



    test_pckle_file = []
    for sequence in test_sequences:
        print("Sequence", sequence)
        sequence_path = basedir + '/'  + sequence
        relative_sequence = sequence + '/'
        p_filenames = natsort.natsorted([n for n in os.listdir(sequence_path) if n.find('pose') >= 0])
        position_old = [0,0,0]
        for i in tqdm.tqdm(range(len(p_filenames))):
            pose_filename = os.path.join(sequence_path, p_filenames[i])
            time = pose_filename.split('/')[-1].split('.')[0]
            name = relative_sequence + str(time) + '.bin'
            pose = np.loadtxt(pose_filename, delimiter=',')
            position = pose[:3,3]
            # if find_euc_dist(position, position_old)<d_thresh:
            #     continue
            position_old = copy.deepcopy(position)
            transformation_matrix = pose
            position2d = np.array([transformation_matrix[0,3], transformation_matrix[1,3]])
            test_pckle_file.append([time,name,position2d,transformation_matrix])

    pckle_loc = save_dir + '/vReLoc_'+str(d_thresh)+'.pkl'

    print(len(train_pckle_file), len(test_pckle_file))
    pckle_file = [test_pckle_file, train_pckle_file]
    with open(pckle_loc, 'wb') as file:
        pickle.dump(pckle_file, file)
