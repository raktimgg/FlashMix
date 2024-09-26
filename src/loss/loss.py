import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from loss.global_loss import triplet_margin_loss


def qlog(q):
    if torch.all(q[1:] == 0):
        q = torch.zeros(3).to(q.device)
    else:
        q = torch.acos(q[0]) * q[1:] / torch.norm(q[1:])
    return q

def qlog_batched(q):
    """
    Compute the logarithm of a batch of quaternions.

    Parameters:
    q (torch.Tensor): A batch of quaternions of shape (N, 4).

    Returns:
    torch.Tensor: The logarithm of the input quaternions of shape (N, 3).
    """
    if q.shape[1] != 4:
        raise ValueError("Input quaternions must have a shape of (N, 4).")

    # Extract the vector part (last three components) of each quaternion
    q_vec = q[:, 1:]

    # Calculate the norms of the vector parts
    norms = torch.norm(q_vec, dim=1, keepdim=True)

    # Initialize a tensor for the result
    result = torch.zeros_like(q_vec[:,:3])

    # Find quaternions with non-zero vector part
    nonzero_indices = torch.where(norms.squeeze() != 0)[0]

    # Compute the logarithm for quaternions with non-zero vector part
    result[nonzero_indices] = torch.acos(q[nonzero_indices, 0].unsqueeze(1)) * q_vec[nonzero_indices] / norms[nonzero_indices]

    return result

class CriterionPose(nn.Module):
    def __init__(self, rot_weight=6):
        super(CriterionPose, self).__init__()
        self.t_loss_fn = nn.L1Loss()
        self.q_loss_fn = nn.L1Loss()
        # self.t_loss_fn = nn.HuberLoss()
        # self.q_loss_fn = nn.HuberLoss()
        self.rot_weight = rot_weight
        # self.t_loss_fn = nn.MSELoss()
        # self.q_loss_fn = nn.MSELoss()

    def forward(self, pred_t, pred_q, gt_t, gt_q):
        loss_t = self.t_loss_fn(pred_t, gt_t)
        loss_q = self.q_loss_fn(pred_q, gt_q)
        # loss_t = torch.mean(0.4*loss_t[:,0] + 0.4*loss_t[:,1] + 0.2*loss_t[:,2])
        loss = 1 * loss_t + self.rot_weight * loss_q

        return loss, loss_t, loss_q


def barlow_twins_loss(embeddings_A, embeddings_B, lambda_param=0.0051):
    batch_size = embeddings_A.size(0)

    embeddings_A = (embeddings_A - embeddings_A.mean(0)) / (
        embeddings_A.std(0) + 1e-7
    )
    embeddings_B = (embeddings_B - embeddings_B.mean(0)) / (
        embeddings_B.std(0) + 1e-7
    )

    cross_correlation = torch.mm(embeddings_A.T, embeddings_B) / batch_size

    on_diag = torch.diagonal(cross_correlation).add_(-1).pow(2).sum()
    off_diag = (
        cross_correlation.pow(2).sum()
        - torch.diagonal(cross_correlation).pow(2).sum()
    )

    loss = on_diag + lambda_param * off_diag
    return loss
