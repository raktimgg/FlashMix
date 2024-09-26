import argparse
import copy
import csv
import glob
import math
import os
import pickle
import shutil
import sys

import h5py
import numpy as np
import tqdm



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
        default='/data/raktim/Datasets/Oxford-Radar',
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
        default=1,
        help="distnace threshold",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    save_dir = args.save_dir
    d_thresh = args.d_thresh

    basedir = data_dir + '/2019-01-'
    end_seq = '-radar-oxford-10k'

    ## You can comment and uncomment the sequences to create the pickle file for the desired sequences
    # train_sequences = ['11-14-02-26', '14-12-05-52', '14-14-48-55', '18-15-20-12']
    # train_sequences = ['11-14-02-26']
    # train_sequences = ['14-12-05-52']
    # train_sequences = ['14-14-48-55']
    # train_sequences = ['18-15-20-12']
    train_sequences = []


    # test_sequences = ['15-13-06-37', '17-13-26-39', '17-14-03-00', '18-14-14-42']
    test_sequences = ['15-13-06-37']
    # test_sequences = ['17-13-26-39']
    # test_sequences = ['17-14-03-00']
    # test_sequences = ['18-14-14-42']
    # test_sequences = []


    def delete_file_if_exists(file_path):
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File '{file_path}' deleted successfully.")
        else:
            print(f"File '{file_path}' does not exist.")


    train_pckle_file = []
    for sequence in train_sequences:
        print("Sequence", sequence)
        folder_path = basedir + sequence + end_seq
        relative_sequence = '2019-01-' + sequence + end_seq + '/velodyne_left/'
        h5_file = h5py.File(folder_path + '/velodyne_left_False.h5', 'r')
        timestamps = h5_file['valid_timestamps'][...]
        poses = h5_file['poses'][...]
        scan_loc_old = [0,0,0]
        for i in tqdm.tqdm(range(timestamps.shape[0])):
            # if i%d_thresh!=0:
            #     continue
            time = timestamps[i]
            name = relative_sequence + str(time) + '.bin'
            pose = poses[i].reshape([3,4])
            transformation_matrix = np.vstack([pose, np.array([[0,0,0,1]])])
            scan_loc = [transformation_matrix[0,3], transformation_matrix[1,3], transformation_matrix[2,3]]
            dist = find_euc_dist(scan_loc, scan_loc_old)
            if dist<d_thresh:
                continue
            scan_loc_old = copy.deepcopy(scan_loc)
            position2d = np.array([transformation_matrix[0,3], transformation_matrix[1,3]])
            train_pckle_file.append([time,name,position2d,transformation_matrix])


    test_pckle_file = []
    for sequence in test_sequences:
        print("Sequence", sequence)
        folder_path = basedir + sequence + end_seq
        relative_sequence = '2019-01-' + sequence + end_seq + '/velodyne_left/'
        h5_file = h5py.File(folder_path + '/velodyne_left_False.h5', 'r')
        timestamps = h5_file['valid_timestamps'][...]
        poses = h5_file['poses'][...]
        scan_loc_old = [0,0,0]
        for i in tqdm.tqdm(range(timestamps.shape[0])):
            # if i%d_thresh!=0:
            #     continue
            time = timestamps[i]
            name = relative_sequence + str(time) + '.bin'
            pose = poses[i].reshape([3,4])
            transformation_matrix = np.vstack([pose, np.array([[0,0,0,1]])])
            scan_loc = [transformation_matrix[0,3], transformation_matrix[1,3], transformation_matrix[2,3]]
            dist = find_euc_dist(scan_loc, scan_loc_old)
            if dist<d_thresh:
                continue
            scan_loc_old = copy.deepcopy(scan_loc)
            position2d = np.array([transformation_matrix[0,3], transformation_matrix[1,3]])
            test_pckle_file.append([time,name,position2d,transformation_matrix])

    pckle_loc = save_dir + '/robotcar_'+str(d_thresh)+'.pkl'

    print(len(train_pckle_file), len(test_pckle_file))
    pckle_file = [test_pckle_file, train_pckle_file]
    with open(pckle_loc, 'wb') as file:
        pickle.dump(pckle_file, file)
