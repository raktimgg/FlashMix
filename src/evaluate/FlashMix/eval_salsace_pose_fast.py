import argparse
import gc
import os
import random
import statistics
import sys
import time

import numpy as np
import torch
import torch_cluster
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from data.common_loader import CommonLoader
from loss.loss import CriterionPose
from models.model_zoo import FeatureExtractor, SceneRegressor
from utils.misc_utils import (
    collate_fn,
    find_anchor_positive_negative_pairs_indices,
    read_yaml_config,
    qexp,
    qlog,
    val_translation,
    val_rotation,
    process_poses,
    print_nb_params,
    print_model_size
)

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"


def main(args, dataset_name):

    base_ds = args['base_ds']
    pickle_loc = args['val_pickle_loc']

    dset = CommonLoader(pickle_loc=pickle_loc, dataset_name=dataset_name, base_ds = base_ds, train=False)
    loader = torch.utils.data.DataLoader(dset, batch_size=200, collate_fn=collate_fn, shuffle=False, num_workers=16)

    device = 'cuda'

    mean_t = torch.Tensor(dset.mean_lidar_center).to(device)
    std_t = torch.Tensor(dset.std_lidar_center).to(device)

    print("Initializing models")

    #### Get model ready
    feature_model = FeatureExtractor(voxel_sz=0.5).to(device)
    feat_model_save_path = args['feat_extractor_model_path']
    checkpoint = torch.load(feat_model_save_path)  # ,map_location='cuda:0')
    feature_model.feature_encoder.load_state_dict(checkpoint)

    if dataset_name=='vReLoc':
        pose_regressor = SceneRegressor(args, N=512).to(device)
    else:
        pose_regressor = SceneRegressor(args).to(device)
    pose_regressor = torch.compile(pose_regressor)
    regressor_save_path = 'src/checkpoints/FlashMix/'+str(dataset_name)+'_regressor_'+str(args['exp_num'])+'.pth'
    checkpoint = torch.load(regressor_save_path)  # ,map_location='cuda:0')
    pose_regressor.load_state_dict(checkpoint)



    feature_model.eval()
    pose_regressor.eval()
    print_nb_params(pose_regressor)
    print_model_size(pose_regressor)


    if dataset_name=='vReLoc':
        mean = torch.Tensor(np.load('src/utils/vReLoc_feature_mean.npy')).to(device)
        std = torch.Tensor(np.load('src/utils/vReLoc_feature_std.npy')).to(device)
    if dataset_name=='robotcar':
        mean = torch.Tensor(np.load('src/utils/robotcar_feature_mean.npy')).to(device)
        std = torch.Tensor(np.load('src/utils/robotcar_feature_std.npy')).to(device)
    if dataset_name=='dcc':
        mean = torch.Tensor(np.load('src/utils/dcc_feature_mean.npy')).to(device)
        std = torch.Tensor(np.load('src/utils/dcc_feature_std.npy')).to(device)


    pose_loss = CriterionPose(args['loss']['pose_loss_rot_weight'])
    
    with torch.inference_mode():
        error_t_list = []
        error_r_list = []
        success = []

        count = 0
        if dataset_name=='vReLoc':
            N_max = 512
        else:
            N_max = 1024
        for i, batch_data in tqdm(enumerate(loader),total = len(loader)):
            count = count+batch_data[0].shape[0]
            coord, xyz, feat, batch_number, pose, scene_pose = batch_data
            coord, xyz, feat, batch_number, pose = coord.to(device), xyz.to(device), feat.to(device), batch_number.to(device), pose.to(device)
            scene_pose = scene_pose.to(device)
            local_features = feature_model(coord, xyz, feat, batch_number)
            _, counts = torch.unique(batch_number, return_counts=True)
            split_local_fetures = torch.split(local_features, list(counts))
            split_coords = torch.split(coord, list(counts))
            split_xyz = torch.split(xyz, list(counts))

            local_features_list = torch.zeros([len(split_local_fetures),N_max,32]).to(device)
            xyz_list = torch.zeros([len(split_local_fetures),N_max,3]).to(device)
            scene_pose_list = torch.zeros([len(split_local_fetures),6]).to(device)
            for idx in range(len(split_local_fetures)):
                local_features_sample = split_local_fetures[idx]
                coord_sample = split_coords[idx]
                scene_pose_sample = scene_pose[idx]
                xyz_sample = split_xyz[idx]
                if local_features_sample.shape[0] >= N_max:
                    selected_indices = torch_cluster.fps(xyz_sample,ratio=N_max/xyz_sample.shape[0])
                    local_features_sample = local_features_sample[selected_indices]
                    coord_sample = coord_sample[selected_indices]
                    xyz_sample = xyz_sample[selected_indices]
                else:
                    index = np.random.choice(local_features_sample.shape[0], N_max-local_features_sample.shape[0], replace=True)
                    local_features_sample = torch.cat([local_features_sample,local_features_sample[index]])
                    coord_sample = torch.cat([coord_sample,coord_sample[index]])
                    xyz_sample = torch.cat([xyz_sample,xyz_sample[index]])

                scene_pose_sample, _, _ = process_poses(scene_pose_sample[:3].view(-1,12).cpu().numpy(), mean_t.cpu().numpy(), std_t.cpu().numpy())
                local_features_list[idx] = local_features_sample
                xyz_list[idx] = xyz_sample
                scene_pose_list[idx] = torch.Tensor(scene_pose_sample).to(device)



            gt_t = torch.Tensor(scene_pose_list[:,:3]).to(device)
            gt_q = torch.Tensor(scene_pose_list[:,3:]).to(device)

            local_features_list = (local_features_list - mean) / (std+1e-6)
            with torch.cuda.amp.autocast():
                pred_t, pred_q, _, _ = pose_regressor(local_features_list,xyz_list)
                loss, loss_t, loss_q = pose_loss(pred_t, pred_q, gt_t*std_t, gt_q)

            if dataset_name=='vReLoc':
                pred_t_arr = (pred_t/5 + mean_t).cpu().numpy()
            else:
                pred_t_arr = (pred_t + mean_t).cpu().numpy()

           
            pred_q_arr = np.asarray([qexp(q) for q in pred_q.cpu().numpy()])


            gt_t_arr = (gt_t*std_t + mean_t).cpu().numpy()
            gt_q_arr = np.asarray([qexp(q) for q in gt_q.cpu().numpy()])


            rte = np.asarray([val_translation(p, q) for p, q in
                                                 zip(pred_t_arr,
                                                     gt_t_arr)])
            rre = np.asarray([val_rotation(p, q) for p, q in
                                                zip(pred_q_arr,
                                                    gt_q_arr)])

            error_t_list.extend(rte.tolist())
            error_r_list.extend(rre.tolist())
    ##############################################################################################################
    translation_error_mean = sum(error_t_list) / len(error_t_list)
    rotation_error_mean = sum(error_r_list) / len(error_r_list)
    translation_error_median = statistics.median(error_t_list)
    rotation_error_median = statistics.median(error_r_list)

    if dataset_name=='vReLoc':
        d_thres = 0.25
        y_thresh = 5
    else:
        d_thres = 5
        y_thresh = 5
    success = 0
    for i in range(len(error_r_list)):
        if error_t_list[i] < d_thres and error_r_list[i] < y_thresh:
            success+=1
    success_rate = success/len(error_r_list)

    print("Translation Error (Mean): {:.3f}, Rotation Error (Mean): {:.3f}, Translation Error (Median): {:.3f}, Rotation Error (Median): {:.3f}".format(
        translation_error_mean, rotation_error_mean, translation_error_median, rotation_error_median))
    print('Success rate', success_rate)