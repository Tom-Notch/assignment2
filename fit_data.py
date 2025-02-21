#!/usr/bin/env python3
import argparse
import os
import time

import torch
from matplotlib import pyplot as plt
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere

import dataset_location
import losses
from r2n2_custom import R2N2
from render import render_mesh
from render import render_pointcloud_raw
from render import render_voxel_raw


def get_args_parser():
    parser = argparse.ArgumentParser("Model Fit", add_help=False)
    parser.add_argument("--lr", default=4e-4, type=float)
    parser.add_argument("--max_iter", default=100000, type=int)
    parser.add_argument(
        "--type", default="vox", choices=["vox", "point", "mesh"], type=str
    )
    parser.add_argument("--n_points", default=5000, type=int)
    parser.add_argument("--w_chamfer", default=1.0, type=float)
    parser.add_argument("--w_smooth", default=0.1, type=float)
    parser.add_argument("--device", default="cuda", type=str)
    return parser


def fit_mesh(mesh_src, mesh_tgt, args):
    start_iter = 0
    start_time = time.time()

    deform_vertices_src = torch.zeros(
        mesh_src.verts_packed().shape, requires_grad=True, device="cuda"
    )
    optimizer = torch.optim.Adam([deform_vertices_src], lr=args.lr)
    print("Starting training !")
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        new_mesh_src = mesh_src.offset_verts(deform_vertices_src)

        sample_trg = sample_points_from_meshes(mesh_tgt, args.n_points)
        sample_src = sample_points_from_meshes(new_mesh_src, args.n_points)

        loss_reg = losses.chamfer_loss(sample_src, sample_trg)
        loss_smooth = losses.smoothness_loss(new_mesh_src)

        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print(
            "[%4d/%4d]; time: %.0f (%.2f); loss: %.3f"
            % (step, args.max_iter, total_time, iter_time, loss_vis)
        )

    mesh_src.offset_verts_(deform_vertices_src)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(render_mesh(mesh_tgt))
    ax[0].axis("off")
    ax[0].text(
        0.5,
        -0.08,
        "Ground Truth Mesh",
        transform=ax[0].transAxes,
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=14,
    )

    ax[1].imshow(render_mesh(mesh_src))
    ax[1].axis("off")
    ax[1].text(
        0.5,
        -0.08,
        "Fit Mesh",
        transform=ax[1].transAxes,
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=14,
    )

    plt.savefig("vis/fit_mesh.png")
    plt.close(fig)

    print("Done!")


def fit_pointcloud(pointclouds_src, pointclouds_tgt, args):
    start_iter = 0
    start_time = time.time()
    optimizer = torch.optim.Adam([pointclouds_src], lr=args.lr)
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        loss = losses.chamfer_loss(pointclouds_src, pointclouds_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print(
            "[%4d/%4d]; time: %.0f (%.2f); loss: %.3f"
            % (step, args.max_iter, total_time, iter_time, loss_vis)
        )

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(render_pointcloud_raw(pointclouds_tgt))
    ax[0].axis("off")
    ax[0].text(
        0.5,
        -0.08,
        "Ground Truth Point Cloud",
        transform=ax[0].transAxes,
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=14,
    )

    ax[1].imshow(render_pointcloud_raw(pointclouds_src))
    ax[1].axis("off")
    ax[1].text(
        0.5,
        -0.08,
        "Fit Point Cloud",
        transform=ax[1].transAxes,
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=14,
    )

    plt.savefig("vis/fit_pointcloud.png")
    plt.close(fig)

    print("Done!")


def fit_voxel(voxels_src, voxels_tgt, args):
    start_iter = 0
    start_time = time.time()
    optimizer = torch.optim.Adam([voxels_src], lr=args.lr)
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        # Clamp voxels_src for the loss computation without modifying the original tensor
        voxels_src_clamped = voxels_src.clamp(0, 1)
        loss = losses.voxel_loss(voxels_src_clamped, voxels_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print(
            "[%4d/%4d]; time: %.0f (%.2f); loss: %.3f"
            % (step, args.max_iter, total_time, iter_time, loss_vis)
        )

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(render_voxel_raw(voxels_tgt))
    ax[0].axis("off")
    ax[0].text(
        0.5,
        -0.08,
        "Ground Truth Voxel",
        transform=ax[0].transAxes,
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=14,
    )

    ax[1].imshow(render_voxel_raw(voxels_src))
    ax[1].axis("off")
    ax[1].text(
        0.5,
        -0.08,
        "Fit Voxel",
        transform=ax[1].transAxes,
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=14,
    )

    plt.savefig("vis/fit_voxel.png")
    plt.close(fig)

    print("Done!")


def train_model(args):
    r2n2_dataset = R2N2(
        "train",
        dataset_location.SHAPENET_PATH,
        dataset_location.R2N2_PATH,
        dataset_location.SPLITS_PATH,
        return_voxels=True,
    )

    feed = r2n2_dataset[0]

    feed_cuda = {}
    for k in feed:
        if torch.is_tensor(feed[k]):
            feed_cuda[k] = feed[k].to(args.device).float()

    if args.type == "vox":
        # initialization
        voxels_src = torch.rand(
            feed_cuda["voxels"].shape, requires_grad=True, device=args.device
        )
        voxel_coords = feed_cuda["voxel_coords"].unsqueeze(0)
        voxels_tgt = feed_cuda["voxels"]

        # fitting
        fit_voxel(voxels_src, voxels_tgt, args)

    elif args.type == "point":
        # initialization
        pointclouds_src = torch.randn(
            [1, args.n_points, 3], requires_grad=True, device=args.device
        )
        mesh_tgt = Meshes(verts=[feed_cuda["verts"]], faces=[feed_cuda["faces"]])
        pointclouds_tgt = sample_points_from_meshes(mesh_tgt, args.n_points)

        # fitting
        fit_pointcloud(pointclouds_src, pointclouds_tgt, args)

    elif args.type == "mesh":
        # initialization
        # try different ways of initializing the source mesh
        mesh_src = ico_sphere(4, args.device)
        mesh_tgt = Meshes(verts=[feed_cuda["verts"]], faces=[feed_cuda["faces"]])

        # fitting
        fit_mesh(mesh_src, mesh_tgt, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Model Fit", parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)
