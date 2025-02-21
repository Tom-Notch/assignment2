#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from pytorch3d.loss import mesh_laplacian_smoothing


# define losses
def voxel_loss(voxel_src: torch.Tensor, voxel_tgt: torch.Tensor):
    # voxel_src: b x h x w x d
    # voxel_tgt: b x h x w x d
    # implement some loss for binary voxel grids
    loss = F.binary_cross_entropy(voxel_src, voxel_tgt)
    return loss


def chamfer_loss(point_cloud_src, point_cloud_tgt):
    # point_cloud_src, point_cloud_src: b x n_points x 3
    # implement chamfer loss from scratch

    # Compute pairwise squared Euclidean distances.
    # This expands point_cloud_src to (B, N, 1, 3) and point_cloud_tgt to (B, 1, M, 3),
    # then subtracts and squares the differences.
    diff = point_cloud_src.unsqueeze(2) - point_cloud_tgt.unsqueeze(1)  # (B, N, M, 3)
    dists = torch.sum(diff**2, dim=-1)  # (B, N, M)

    # For each point in the source, find the minimum squared distance to the target points.
    min_dists_src, _ = torch.min(dists, dim=2)  # (B, N)

    # For each point in the target, find the minimum squared distance to the source points.
    min_dists_tgt, _ = torch.min(dists, dim=1)  # (B, M)

    # The Chamfer loss is the sum of the average minimum distances in both directions.
    loss_chamfer = torch.mean(min_dists_src) + torch.mean(min_dists_tgt)
    return loss_chamfer


def smoothness_loss(mesh_src):
    # implement laplacian smoothening loss
    loss_laplacian = mesh_laplacian_smoothing(mesh_src)
    return loss_laplacian
