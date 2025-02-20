#!/usr/bin/env python3
import argparse
import math
import time

import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import torch
from pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
from pytorch3d.ops import knn_points
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.renderer import AlphaCompositor
from pytorch3d.renderer import FoVPerspectiveCameras
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.renderer import MeshRasterizer
from pytorch3d.renderer import MeshRenderer
from pytorch3d.renderer import PointLights
from pytorch3d.renderer import PointsRasterizationSettings
from pytorch3d.renderer import PointsRasterizer
from pytorch3d.renderer import PointsRenderer
from pytorch3d.renderer import RasterizationSettings
from pytorch3d.renderer import SoftPhongShader
from pytorch3d.renderer import TexturesVertex
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.transforms import Rotate

import dataset_location
import utils_vox
from model import SingleViewto3D
from r2n2_custom import R2N2


def get_args_parser():
    parser = argparse.ArgumentParser("Singleto3D", add_help=False)
    parser.add_argument("--arch", default="resnet18", type=str)
    parser.add_argument("--vis_freq", default=1000, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument(
        "--type", default="vox", choices=["vox", "point", "mesh"], type=str
    )
    parser.add_argument("--n_points", default=1000, type=int)
    parser.add_argument("--w_chamfer", default=1.0, type=float)
    parser.add_argument("--w_smooth", default=0.1, type=float)
    parser.add_argument("--load_checkpoint", action="store_true")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--load_feat", action="store_true")
    return parser


def preprocess(feed_dict, args):
    for k in ["images"]:
        feed_dict[k] = feed_dict[k].to(args.device)

    images = feed_dict["images"].squeeze(1)
    mesh = feed_dict["mesh"]
    if args.load_feat:
        images = torch.stack(feed_dict["feats"]).to(args.device)

    return images, mesh


def save_plot(thresholds, avg_f1_score, args):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(thresholds, avg_f1_score, marker="o")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1-score")
    ax.set_title(f"Evaluation {args.type}")
    plt.savefig(f"eval_{args.type}", bbox_inches="tight")


def compute_sampling_metrics(pred_points, gt_points, thresholds, eps=1e-8):
    metrics = {}
    lengths_pred = torch.full(
        (pred_points.shape[0],),
        pred_points.shape[1],
        dtype=torch.int64,
        device=pred_points.device,
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],),
        gt_points.shape[1],
        dtype=torch.int64,
        device=gt_points.device,
    )

    # For each predicted point, find its nearest-neighbor GT point
    knn_pred = knn_points(
        pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1
    )
    # Compute L1 and L2 distances between each pred point and its nearest GT
    pred_to_gt_dists2 = knn_pred.dists[..., 0]  # (N, S)
    pred_to_gt_dists = pred_to_gt_dists2.sqrt()  # (N, S)

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = knn_points(
        gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1
    )
    # Compute L1 and L2 dists between each GT point and its nearest pred point
    gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)
    gt_to_pred_dists = gt_to_pred_dists2.sqrt()  # (N, S)

    # Compute precision, recall, and F1 based on L2 distances
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        metrics["Precision@%f" % t] = precision
        metrics["Recall@%f" % t] = recall
        metrics["F1@%f" % t] = f1

    # Move all metrics to CPU
    metrics = {k: v.cpu() for k, v in metrics.items()}
    return metrics


def evaluate(predictions, mesh_gt, thresholds, args):
    if args.type == "vox":
        voxels_src = predictions
        H, W, D = voxels_src.shape[2:]
        vertices_src, faces_src = mcubes.marching_cubes(
            voxels_src.detach().cpu().squeeze().numpy(), isovalue=0.1
        )
        vertices_src = torch.tensor(vertices_src).float()
        faces_src = torch.tensor(faces_src.astype(int))
        mesh_src = pytorch3d.structures.Meshes([vertices_src], [faces_src])
        pred_points = sample_points_from_meshes(mesh_src, args.n_points)
        pred_points = utils_vox.Mem2Ref(pred_points, H, W, D)
        # Apply a rotation transform to align predicted voxels to gt mesh
        angle = -math.pi
        axis_angle = torch.as_tensor(np.array([[0.0, angle, 0.0]]))
        Rot = axis_angle_to_matrix(axis_angle)
        T_transform = Rotate(Rot)
        pred_points = T_transform.transform_points(pred_points)
        # re-center the predicted points
        pred_points = pred_points - pred_points.mean(1, keepdim=True)
    elif args.type == "point":
        pred_points = predictions.cpu()
    elif args.type == "mesh":
        pred_points = sample_points_from_meshes(predictions, args.n_points).cpu()

    gt_points = sample_points_from_meshes(mesh_gt, args.n_points)
    if args.type == "vox":
        gt_points = gt_points - gt_points.mean(1, keepdim=True)
    metrics = compute_sampling_metrics(pred_points, gt_points, thresholds)
    return metrics


def get_points_renderer(
    image_size=512, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.

    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=radius,
    )
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer


def evaluate_model(args):
    r2n2_dataset = R2N2(
        "test",
        dataset_location.SHAPENET_PATH,
        dataset_location.R2N2_PATH,
        dataset_location.SPLITS_PATH,
        return_voxels=True,
        return_feats=args.load_feat,
    )

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True,
    )
    eval_loader = iter(loader)

    model = SingleViewto3D(args)
    model.to(args.device)
    model.eval()

    start_iter = 0
    start_time = time.time()

    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]

    avg_f1_score_05 = []
    avg_f1_score = []
    avg_p_score = []
    avg_r_score = []

    if args.load_checkpoint:
        checkpoint = torch.load(f"checkpoints/{args.type}.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Successfully loaded iter {start_iter}")

    print("Starting evaluating !")
    max_iter = len(eval_loader)
    for step in range(start_iter, max_iter):
        iter_start_time = time.time()

        read_start_time = time.time()

        feed_dict = next(eval_loader)

        images_gt, mesh_gt = preprocess(feed_dict, args)

        read_time = time.time() - read_start_time

        predictions = model(images_gt, args)

        metrics = evaluate(predictions, mesh_gt, thresholds, args)

        cameras = FoVPerspectiveCameras(device=args.device)
        lights = PointLights(device=args.device, location=[[0.0, 0.0, -3.0]])
        raster_settings = RasterizationSettings(
            image_size=256,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        mesh_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(device=args.device, cameras=cameras, lights=lights),
        )

        if (step % args.vis_freq) == 0:
            # --- Common Rendering ---
            device = args.device

            fig = plt.figure(figsize=(18, 6))

            # Subplot 1: Input RGB
            ax1 = fig.add_subplot(1, 3, 1)
            input_rgb = images_gt[0].detach().cpu().numpy()
            ax1.imshow(input_rgb)
            ax1.axis("off")
            # Place caption below: y=-0.08 positions the text below the axes.
            ax1.text(
                0.5,
                -0.08,
                "Input RGB",
                transform=ax1.transAxes,
                horizontalalignment="center",
                verticalalignment="bottom",
                fontsize=14,
            )

            # Subplot 2: Ground Truth Mesh rendering
            # Compute camera parameters (common to both mesh and predict rendering)
            R, T = look_at_view_transform(dist=1, elev=10, azim=0, device=device)
            cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
            # Mesh renderer for ground truth:
            lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
            raster_settings_mesh = RasterizationSettings(
                image_size=256, blur_radius=0.0, faces_per_pixel=1
            )
            mesh_renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras, raster_settings=raster_settings_mesh
                ),
                shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
            )
            # Prepare ground truth mesh (colorize if needed)
            mesh_gt0 = mesh_gt[0].to(device)
            if mesh_gt0.textures is None:
                verts = mesh_gt0.verts_packed()
                colors = torch.full(
                    (1, verts.shape[0], 3), 0.7, device=device
                )  # light gray
                mesh_gt0 = pytorch3d.structures.Meshes(
                    verts=mesh_gt0.verts_list(),
                    faces=mesh_gt0.faces_list(),
                    textures=TexturesVertex(verts_features=colors),
                )
            rend_gt = mesh_renderer(mesh_gt0)[0, ..., :3].detach().cpu().numpy()
            ax2 = fig.add_subplot(1, 3, 3)
            ax2.imshow(rend_gt)
            ax2.axis("off")
            ax2.text(
                0.5,
                -0.08,
                "Ground Truth Mesh",
                transform=ax2.transAxes,
                horizontalalignment="center",
                verticalalignment="bottom",
                fontsize=14,
            )

            # Subplot 3: Predicted reconstruction
            ax3 = fig.add_subplot(1, 3, 2)

            # --- Modality-Specific Rendering ---
            if args.type == "point":
                points_renderer = get_points_renderer(
                    image_size=256, background_color=(1, 1, 1), device=device
                )
                # predictions is assumed to be (B, n_points, 3); get first sample.
                pred_points = predictions[0].detach().to(device)
                # Create a uniform blue color for the predicted point cloud.
                pred_colors = torch.zeros_like(pred_points)
                pred_colors[:, 2] = 1.0
                from pytorch3d.structures import Pointclouds

                pred_pc = Pointclouds(points=[pred_points], features=[pred_colors])
                rend_points = (
                    points_renderer(pred_pc, cameras=cameras)[0, ..., :3]
                    .detach()
                    .cpu()
                    .numpy()
                )

                ax3.imshow(rend_points)
                ax3.axis("off")
                ax3.text(
                    0.5,
                    -0.08,
                    "Predicted Point",
                    transform=ax3.transAxes,
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    fontsize=14,
                )
            elif args.type == "mesh":
                # predictions is assumed to be a Meshes object.
                pred_mesh = predictions.to(device)
                verts = pred_mesh.verts_packed()
                colors = torch.full(
                    (1, verts.shape[0], 3), 0.5, device=device
                )  # uniform mid-gray
                pred_mesh = pytorch3d.structures.Meshes(
                    verts=pred_mesh.verts_list(),
                    faces=pred_mesh.faces_list(),
                    textures=TexturesVertex(verts_features=colors),
                )
                rend_pred = mesh_renderer(pred_mesh)[0, ..., :3].detach().cpu().numpy()
                ax3.imshow(rend_pred)
                ax3.axis("off")
                ax3.text(
                    0.5,
                    -0.08,
                    "Predicted Mesh",
                    transform=ax3.transAxes,
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    fontsize=14,
                )

            plt.savefig(f"vis/{step}_{args.type}_renderer.png")
            plt.close(fig)

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        f1_05 = metrics["F1@0.050000"]
        avg_f1_score_05.append(f1_05)
        avg_p_score.append(
            torch.tensor([metrics["Precision@%f" % t] for t in thresholds])
        )
        avg_r_score.append(torch.tensor([metrics["Recall@%f" % t] for t in thresholds]))
        avg_f1_score.append(torch.tensor([metrics["F1@%f" % t] for t in thresholds]))

        print(
            "[%4d/%4d]; time: %.0f (%.2f, %.2f); F1@0.05: %.3f; Avg F1@0.05: %.3f"
            % (
                step,
                max_iter,
                total_time,
                read_time,
                iter_time,
                f1_05,
                torch.tensor(avg_f1_score_05).mean(),
            )
        )

    avg_f1_score = torch.stack(avg_f1_score).mean(0)

    save_plot(thresholds, avg_f1_score, args)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Singleto3D", parents=[get_args_parser()])
    args = parser.parse_args()
    evaluate_model(args)
