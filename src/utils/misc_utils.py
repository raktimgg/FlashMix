import csv
import random

import numpy as np
import torch
import yaml
import transforms3d.quaternions as txq

def find_anchor_positive_negative_pairs_indices(embeddings, labels):
    # Group indices by label
    label_to_indices = {}
    for i, label in enumerate(labels):
        if label.item() not in label_to_indices:
            label_to_indices[label.item()] = []
        label_to_indices[label.item()].append(i)

    # Create anchor-positive pairs and find negatives
    anchor_indices = []
    positive_indices = []
    negative_indices = []

    all_labels = torch.unique(labels)
    used_labels = set()  # Track used labels

    for label, indices in label_to_indices.items():
        if label in used_labels:
            continue  # Skip if label is already used

        if len(indices) > 1:  # Ensure there are at least two samples to form a pair
            used_labels.add(label)
            anchor_indices.append(indices[0])
            positive_indices.append(indices[1])

            # Find a single negative sample
            while True:
                neg_label = all_labels[torch.randint(len(all_labels), (1,))].item()
                if neg_label != label and neg_label in label_to_indices:
                    neg_samples = label_to_indices[neg_label]
                    neg_idx = torch.randint(len(neg_samples), (1,)).item()
                    negative_indices.append(neg_samples[neg_idx])
                    break

    return torch.tensor(anchor_indices), torch.tensor(positive_indices), torch.tensor(negative_indices)


def collate_fn(batch):
    coord, xyz, feat, poses, scene_pose = list(zip(*batch))
    offset, count = [], 0

    new_coord, new_xyz, new_feat, new_poses, new_scene_poses = [], [], [], [], []
    k = 0
    for i, item in enumerate(xyz):

        count += item.shape[0]
        k += 1
        offset.append(count)
        new_coord.append(coord[i])
        new_xyz.append(xyz[i])
        new_feat.append(feat[i])
        new_poses.append(poses[i])
        new_scene_poses.append(scene_pose[i])

    offset_ = torch.IntTensor(offset[:k]).clone()
    offset_[1:] = offset_[1:] - offset_[:-1]
    batch_number = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()
    coords,xyz,feat,poses = torch.cat(new_coord[:k]), torch.cat(new_xyz[:k]), torch.cat(new_feat[:k]), torch.cat(new_poses[:k])
    return coords,xyz,feat,batch_number,poses, torch.Tensor(np.array(new_scene_poses))


def read_yaml_config(filename):
    with open(filename, 'r') as stream:
        try:
            # Load the YAML file
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)
            return None

def print_nb_params(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Trainable parameters: {params/1e6:.3}M')
    del model_parameters, params

def print_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

def qexp(q):
    """
    Applies the exponential map to q
    :param q: (3,)
    :return: (4,)
    """
    n = np.linalg.norm(q)
    q = np.hstack((np.cos(n), np.sinc(n/np.pi)*q))

    return q

def qlog(q):
    """
    Applies logarithm map to q
    :param q: (4,)
    :return: (3,)
    """
    if all(q[1:] == 0):
        q = np.zeros(3)
    else:
        q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])

    return q


def val_translation(pred_p, gt_p):
    """
    test model, compute error (numpy)
    input:
        pred_p: [3,]
        gt_p: [3,]
    returns:
        translation error (m):
    """
    if isinstance(pred_p, np.ndarray):
        predicted   = pred_p
        groundtruth = gt_p
    else:
        predicted   = pred_p.cpu().numpy()
        groundtruth = gt_p.cpu().numpy()
    error = np.linalg.norm(groundtruth - predicted)

    return error

def val_rotation(pred_q, gt_q):
    """
    test model, compute error (numpy)
    input:
        pred_q: [4,]
        gt_q: [4,]
    returns:
        rotation error (degrees):
    """
    if isinstance(pred_q, np.ndarray):
        predicted   = pred_q
        groundtruth = gt_q
    else:
        predicted   = pred_q.cpu().numpy()
        groundtruth = gt_q.cpu().numpy()

    d = abs(np.dot(groundtruth, predicted))
    d = min(1.0, max(-1.0, d))
    error = 2 * np.arccos(d) * 180 / np.pi

    return error


def process_poses(poses_in, mean_t, std_t, align_R=np.eye(3), align_t=np.zeros(3), align_s=1):
    poses_out = np.zeros((len(poses_in), 6))
    poses_out[:, 0:3] = poses_in[:, [3, 7, 11]]

    # align
    for i in range(len(poses_out)):
        R = poses_in[i].reshape((3, 4))[:3, :3]
        q = txq.mat2quat(np.dot(align_R, R))
        q *= np.sign(q[0])  # constrain to hemisphere
        q = qlog(q)
        poses_out[i, 3:] = q
        t = poses_out[i, :3] - align_t
        poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()

    # normalize translation
    poses_out[:, :3] -= mean_t
    poses_out[:, :3] /= std_t

    # max and min
    pose_max = np.max(poses_out[:, :2], axis=0)  # (2,)
    pose_min = np.min(poses_out[:, :2], axis=0)  # (2,)

    return poses_out, pose_max, pose_min