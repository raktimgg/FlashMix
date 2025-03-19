import argparse
import gc
import os
import sys
import time
from functools import partial

import numpy as np
import torch
import torch_cluster
from pytorch_metric_learning import losses, miners
from sklearn.cluster import KMeans
from torch.amp import autocast
from tqdm import tqdm

import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from data.common_loader import CommonLoader

from loss.loss import (
    CriterionPose,
    barlow_twins_loss,
)
from models.model_zoo import FeatureExtractor, SceneRegressor
from utils.misc_utils import (
    collate_fn,
    find_anchor_positive_negative_pairs_indices,
    read_yaml_config,
    qexp,
    qlog,
    val_translation,
    val_rotation,
    process_poses

)

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

def get_loss_function(loss_name):
    # Define your loss functions
    loss_functions = {
        "triplet_loss": losses.TripletMarginLoss(),
        "barlow_loss": partial(barlow_twins_loss, lambda_param=0.005),
    }

    # Select the appropriate loss function based on the provided loss_name
    if loss_name in loss_functions:
        return loss_functions[loss_name]
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


def main(args):
    exp_num = args['exp_num']
    print('Experiment instance ', exp_num)
    do_wandb = args['do_wandb']
    if do_wandb:
        wandb.login()
        run = wandb.init(
            project="FlashMix",
            name="Trial",
        )
    # Get data loader
    dataset_name = 'dcc'
    base_ds = args['base_ds']
    pickle_loc = args['pickle_loc']
    do_validation = args['do_validation']

    batch_size=200
    dset = CommonLoader(pickle_loc=pickle_loc, dataset_name=dataset_name, base_ds=base_ds, train=True)
    loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=16)

    if do_validation:
        val_pickle_loc = args['val_pickle_loc']
        val_dset = CommonLoader(pickle_loc=val_pickle_loc, dataset_name=dataset_name, base_ds=base_ds, train=False)
        val_loader = torch.utils.data.DataLoader(val_dset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=16)

    device = 'cuda'

    mean_t = torch.Tensor(dset.mean_lidar_center).to(device)
    std_t = torch.Tensor(dset.std_lidar_center).to(device)

    print("Initializing models")

    #### Get model ready
    feature_model = FeatureExtractor(voxel_sz=0.5).to(device)
    feat_model_save_path = args['feat_extractor_model_path']
    checkpoint = torch.load(feat_model_save_path)  # ,map_location='cuda:0')
    feature_model.feature_encoder.load_state_dict(checkpoint)

    pose_regressor = SceneRegressor(args).to(device)
    pose_regressor = torch.compile(pose_regressor)

    feature_model.eval()

    ################################################
    ###### Create feature buffer for training ######
    ################################################
    print("Creating Training Feature Buffer")
    local_feature_path = 'src/utils/local_features_dcc_0.npy'
    gt_pose_path = 'src/utils/gt_pose_dcc_0.npy'
    xyz_path = 'src/utils/xyz_dcc_0.npy'

    if os.path.exists(local_feature_path):
        local_features = np.load(local_feature_path)
        gt_pose = np.load(gt_pose_path)
        xyz = np.load(xyz_path)
    else:
        with torch.inference_mode():
            local_feature_list = []
            xyz_list = []
            gt_pose = []
            k = 0
            N_max = 1024
            for i in range(1):
                for i, batch_data in tqdm(enumerate(loader),total = len(loader)):
                    coord, xyz, feat, batch_number, pose, scene_pose = batch_data
                    coord, xyz, feat, batch_number, pose = coord.to(device), xyz.to(device), feat.to(device), batch_number.to(device), pose.to(device)
                    scene_pose = scene_pose.to(device)
                    local_features = feature_model(coord, xyz, feat, batch_number)

                    # local_features = xyz
                    _, counts = torch.unique(batch_number, return_counts=True)
                    split_local_fetures = torch.split(local_features, list(counts))
                    split_coord = torch.split(coord, list(counts))
                    split_xyz = torch.split(xyz, list(counts))
                    # split_scene_pose = torch.split(scene_pose, list(counts))

                    split_scene_pose = scene_pose
                    local_features_temp = []
                    xyz_temp = []
                    pose_list_temp = []

                    for local_features, coord, xyz, scene_pose in zip(split_local_fetures,split_coord,split_xyz, split_scene_pose):
                        k+=1
                        if local_features.shape[0] >= N_max:
                            selected_indices = torch_cluster.fps(xyz,ratio=N_max/xyz.shape[0])
                            if selected_indices.shape[0]!=N_max:
                                print(selected_indices.shape)
                            local_features = local_features[selected_indices]
                            xyz = xyz[selected_indices]
                            coord = coord[selected_indices]
                        else:
                            num_to_pad = N_max - local_features.shape[0]
                            index = np.random.choice(local_features.shape[0], size=num_to_pad, replace=True)
                            local_features = torch.cat([local_features,local_features[index]])
                            xyz = torch.cat([xyz,xyz[index]])
                            coord = torch.cat([coord,coord[index]])

                        scene_pose, _, _ = process_poses(scene_pose[:3].view(-1,12).cpu().numpy(), mean_t.cpu().numpy(), std_t.cpu().numpy())

                        pose_list_temp.append(scene_pose[0,:].astype(np.float32))
                        local_features_temp.append(local_features.cpu().numpy())
                        xyz_temp.append(xyz.cpu().numpy())

                    local_feature_list.extend(local_features_temp)
                    xyz_list.extend(xyz_temp)
                    gt_pose.extend(pose_list_temp)
            local_features = np.array(local_feature_list)
            xyz = np.array(xyz_list)
            gt_pose = np.array(gt_pose)

        np.save(local_feature_path, local_features)
        np.save(gt_pose_path, gt_pose)
        np.save(xyz_path, xyz)
    print("Training Feature Buffer Created")



    if do_validation:
        ################################################
        ##### Create feature buffer for validation #####
        ################################################
        print("Creating Validation Feature Buffer")
        local_feature_val_path = 'src/utils/local_features_dcc_val.npy'
        gt_pose_val_path = 'src/utils/gt_pose_dcc_val.npy'
        xyz_val_path = 'src/utils/xyz_dcc_val.npy'

        if os.path.exists(local_feature_val_path):
            local_features_val = np.load(local_feature_val_path)
            gt_pose_val = np.load(gt_pose_val_path)
            xyz_val = np.load(xyz_val_path)
        else:
            with torch.inference_mode():
                local_feature_val_list = []
                xyz_val_list = []
                gt_pose_val = []
                k = 0
                N_max = 1024
                for i, batch_data in tqdm(enumerate(val_loader),total = len(val_loader)):
                    coord, xyz_val, feat, batch_number, pose, scene_pose = batch_data
                    coord, xyz_val, feat, batch_number, pose = coord.to(device), xyz_val.to(device), feat.to(device), batch_number.to(device), pose.to(device)
                    scene_pose = scene_pose.to(device)
                    local_features_val = feature_model(coord, xyz_val, feat, batch_number)

                    _, counts = torch.unique(batch_number, return_counts=True)
                    split_local_fetures = torch.split(local_features_val, list(counts))
                    split_coord = torch.split(coord, list(counts))
                    split_xyz = torch.split(xyz_val, list(counts))

                    split_scene_pose = scene_pose
                    local_features_temp = []
                    xyz_temp = []
                    pose_list_temp = []

                    for local_features_val, coord, xyz_val, scene_pose in zip(split_local_fetures,split_coord,split_xyz, split_scene_pose):
                        k+=1
                        if local_features_val.shape[0] >= N_max:
                            selected_indices = torch_cluster.fps(xyz_val,ratio=N_max/xyz_val.shape[0])
                            local_features_val = local_features_val[selected_indices]
                            xyz_val = xyz_val[selected_indices]
                            coord = coord[selected_indices]
                        else:
                            num_to_pad = N_max - local_features_val.shape[0]
                            index = np.random.choice(local_features_val.shape[0], size=num_to_pad, replace=True)
                            local_features_val = torch.cat([local_features_val,local_features_val[index]])
                            xyz_val = torch.cat([xyz_val,xyz_val[index]])
                            coord = torch.cat([coord,coord[index]])

                        scene_pose, _, _ = process_poses(scene_pose[:3].view(-1,12).cpu().numpy(), mean_t.cpu().numpy(), std_t.cpu().numpy())

                        pose_list_temp.append(scene_pose[0,:].astype(np.float32))
                        local_features_temp.append(local_features_val.cpu().numpy())
                        xyz_temp.append(xyz_val.cpu().numpy())

                    local_feature_val_list.extend(local_features_temp)
                    xyz_val_list.extend(xyz_temp)
                    gt_pose_val.extend(pose_list_temp)
                local_features_val = np.array(local_feature_val_list)
                xyz_val = np.array(xyz_val_list)
                gt_pose_val = np.array(gt_pose_val)

            np.save(local_feature_val_path, local_features_val)
            np.save(gt_pose_val_path, gt_pose_val)
            np.save(xyz_val_path, xyz_val)

        print("Validation Feature Buffer Created")

    print('Clustering Poses for Contrastive/Metric Loss')
    xy_coords = gt_pose[:, :2]  # Shape (N, 2)
    K = args['num_clusters']
    kmeans = KMeans(n_clusters=K, random_state=0, n_init='auto').fit(xy_coords)
    labels = kmeans.labels_
    print('Clustering succesfull!!')

    print('Starting Training')
    batch_size = args['training']['batch_size']

    # local features are normalized and mean and std are saved for 
    # use in evaluation
    mean = np.mean(local_features, axis=(0,1), keepdims=True)
    std = np.std(local_features, axis=(0,1), keepdims=True)
    np.save('src/utils/dcc_feature_mean.npy', mean)
    np.save('src/utils/dcc_feature_std.npy', std)

    local_features = (local_features - mean) / (std+1e-6)
    
    # entire training data is directly loaded to the GPU for faster training
    local_features_all = torch.Tensor(local_features).to(device)
    labels_all = torch.Tensor(labels).to(device)
    gt_pose_all = torch.Tensor(gt_pose).to(device)

    if do_validation:
        local_features_val = (local_features_val - mean) / (std+1e-6)
        local_features_val_all = torch.Tensor(local_features_val).to(device)
        gt_pose_val_all = torch.Tensor(gt_pose_val).to(device)

    MAX_EPOCH = args['training']['epochs']

    if do_wandb:
        print('Logging gradients')
        wandb.watch(pose_regressor, log='all', log_freq=5)

    pose_regressor.train()
    scaler = torch.cuda.amp.GradScaler()
    pose_loss = CriterionPose(args['loss']['pose_loss_rot_weight'])
    contrastive_criterion = get_loss_function(args['loss']['contrastive_loss_name'])

    optimizer1 = torch.optim.Adam(list(pose_regressor.parameters()), lr=0.001)
    scheduler1 = torch.optim.lr_scheduler.OneCycleLR(optimizer1, max_lr=args['training']['max_lr'], 
                                                     pct_start=args['training']['pct_start'], 
                                                     final_div_factor=args['training']['final_div_factor'],
                                                     total_steps=int((int(len(gt_pose_all)/batch_size))*MAX_EPOCH), 
                                                     anneal_strategy='cos', cycle_momentum=False)

    EPOCH_LOSS = []
    time1 = time.time()
    for e in range(MAX_EPOCH):
        batch_indices = np.random.permutation(np.arange(0,len(gt_pose_all)))
        BATCH_LOSS = []
        BATCH_TRIPLET_LOSS = []
        pose_regressor.train()
        print('Training...')
        for i in range(int(len(batch_indices)/batch_size)):
            pose_regressor.zero_grad()
            optimizer1.zero_grad()

            start_ind = i*batch_size
            end_ind = (i+1)*batch_size
            batch_ind = batch_indices[start_ind:end_ind]

            local_features = local_features_all[batch_ind].float().contiguous()
            labels = labels_all[batch_ind].float().contiguous()
            gt_pose = gt_pose_all[batch_ind].float().contiguous()

            gt_t = gt_pose[:,:3]
            gt_q = gt_pose[:,3:]
            gt_t = gt_t*std_t
            with autocast('cuda'):
                pred_t1, pred_q1, _, _, emb = pose_regressor(local_features,None,return_emb=True)

                loss1, loss_t, loss_q = pose_loss(pred_t1, pred_q1, gt_t, gt_q)
                pairs = find_anchor_positive_negative_pairs_indices(emb, labels)
                if args['loss']['contrastive_loss_name'] == "triplet_loss":
                    loss_contrastive = contrastive_criterion(emb, labels)
                elif args['loss']['contrastive_loss_name'] == "barlow_loss":
                    loss_contrastive = contrastive_criterion(
                        emb[pairs[0]], emb[pairs[1]]
                    )

                loss = loss1 + args['loss']['contrastive_loss_scale'] * loss_contrastive

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(pose_regressor.parameters(), 1.0)
            scaler.step(optimizer1)
            scaler.update()
            scheduler1.step()

            BATCH_LOSS.append(loss1.item())
            BATCH_TRIPLET_LOSS.append(loss_contrastive.item())
            sys.stdout.write('\r' + 'Epoch ' + str(e + 1) + ' / ' + str(MAX_EPOCH) + ' Progress ' + str(i+1) + ' / ' + str(int(len(gt_pose_all)/batch_size))+ ' Loss1 ' + str(format(loss1.item(),'.3f'))+ ' Loss_t ' + str(format(loss_t.item(),'.3f')) + ' Loss_q ' + str(format(loss_q.item(),'.3f')) + ' Loss_triplet ' + str(format(loss_contrastive.item(),'.3f')) + ' time '+ str(format(time.time()-time1,'.2f'))+' seconds.')

            del local_features, gt_pose

            gc.collect()
            torch.cuda.empty_cache()

        current_lr = optimizer1.param_groups[0]['lr']

        epoch_loss_avg = sum(BATCH_LOSS)/len(BATCH_LOSS)
        epoch_loss_batch_avg = sum(BATCH_TRIPLET_LOSS)/len(BATCH_TRIPLET_LOSS)
        EPOCH_LOSS.append(epoch_loss_avg)

        print(' ')
        print(f'Avg. Training Loss {epoch_loss_avg:.2f} Current LR {current_lr:.6f} Time {time.time()-time1:.2f}')
        print(f'Avg. Triplet Loss {epoch_loss_batch_avg:.5f}')

        save_loc = args['training']['save_folder'] + '/dcc_regressor_' + str(exp_num) + '.pth'
        print('Saving model at ', save_loc)
        torch.save(pose_regressor.state_dict(), save_loc)


        if do_validation:
            print('Validating...')
            BATCH_LOSS = []
            pose_regressor.eval()

            with torch.inference_mode():
                error_t_list = []
                error_r_list = []

                local_features = local_features_val_all.float().contiguous()
                gt_pose = gt_pose_val_all.float().contiguous()

                gt_t = gt_pose[:,:3]
                gt_q = gt_pose[:,3:]
                with autocast('cuda'):
                    pred_t, pred_q, _, _, emb = pose_regressor(local_features,None,return_emb=True)

                    loss1, loss_t, loss_q = pose_loss(pred_t, pred_q, gt_t*std_t, gt_q)
 
                    loss = loss1

                BATCH_LOSS.append(loss1.item())

                del local_features, gt_pose

                gc.collect()
                torch.cuda.empty_cache()
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

            current_lr = optimizer1.param_groups[0]['lr']

            epoch_val_loss_avg = sum(BATCH_LOSS)/len(BATCH_LOSS)

            success = 0
            for i in range(len(error_r_list)):
                if error_t_list[i] < 5 and error_r_list[i] < 5:
                    success+=1
            success_rate = success/len(error_r_list)*100

            print(' ')

            print(f'Avg. Validation Loss {epoch_val_loss_avg:.2f}, Translation Error {np.mean(error_t_list):.2f}, Rotation Error {np.mean(error_r_list):.2f}, Success Rate {success_rate:.2f}')

        print(' ')
        if do_wandb:
            if do_validation:
                wandb.log({"Current LR":current_lr, "Training loss": epoch_loss_avg,"Validation loss": epoch_val_loss_avg, "Translation Error":np.mean(error_t_list), "Rotation Error":np.mean(error_r_list)})
            else:
                wandb.log({"Current LR":current_lr, "Training loss": epoch_loss_avg})
