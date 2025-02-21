#!/usr/bin/env python3
#
# Created on Thu Feb 20 2025 19:50:44
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
from typing import Dict
from typing import Tuple

import matplotlib.cm as cm
import numpy as np
import torch
from pytorch3d.renderer import AlphaCompositor
from pytorch3d.renderer import EmissionAbsorptionRaymarcher
from pytorch3d.renderer import FoVPerspectiveCameras
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.renderer import MeshRasterizer
from pytorch3d.renderer import MeshRenderer
from pytorch3d.renderer import NDCGridRaysampler
from pytorch3d.renderer import PointLights
from pytorch3d.renderer import PointsRasterizationSettings
from pytorch3d.renderer import PointsRasterizer
from pytorch3d.renderer import PointsRenderer
from pytorch3d.renderer import RasterizationSettings
from pytorch3d.renderer import SoftPhongShader
from pytorch3d.renderer import TexturesVertex
from pytorch3d.renderer import VolumeRenderer
from pytorch3d.structures import Meshes
from pytorch3d.structures import Volumes
from pytorch3d.structures.pointclouds import Pointclouds
from torch import Tensor


def render_pointcloud_raw(
    pointcloud: Tensor,
    *args,
    **kwargs,
) -> np.ndarray:
    """
    Render a pointcloud provided as a tensor of shape (N, 3) or (N, 6) using PyTorch3D.
    If the tensor is of shape (N,3), no per-point color is assumed.
    If the tensor is of shape (N,6), the last three channels are used as RGB colors.
    This function converts the tensor to a Pointclouds object and then calls the pointcloud version.

    Args:
        pointcloud (Tensor): A tensor of shape (N, 3) or (N, 6) representing 3D points and optionally color.
        image_size (int): The size (width and height in pixels) for rendering.
        elev (float): Elevation angle (degrees) for the camera.
        azim (float): Azimuth angle (degrees) for the camera.
        dist (float): Distance from the camera to the object.
        background_color (tuple): Background color for the renderer.

    Returns:
        np.ndarray: the rendered pointcloud image.
    """
    pointcloud = pointcloud.clone().detach().cpu()
    # If there are extra channels (e.g. (N,6)), assume the first three are the coordinates.
    if pointcloud.shape[-1] > 3:
        pointcloud = pointcloud[:, :3]
    pointcloud = pointcloud.reshape(-1, 3)

    # Extract the height (z-coordinate) and normalize to [0, 1]
    y = pointcloud[:, 1]
    y_min, y_max = y.min(), y.max()
    if y_max - y_min > 0:
        y_norm = (y - y_min) / (y_max - y_min)
    else:
        # If all z values are the same, assign a constant value.
        y_norm = torch.zeros_like(y)

    colormap = cm.get_cmap("viridis")
    colors_np = colormap(y_norm.cpu().numpy())[:, :3]  # shape (N, 3)

    # Convert colors back to a torch tensor and move to the appropriate device.
    colors = torch.from_numpy(colors_np).to(pointcloud.device).type_as(pointcloud)

    # Create a Pointclouds object with the computed colors.
    pc = Pointclouds(points=[pointcloud], features=[colors])

    return render_pointcloud(pc, *args, **kwargs)


def render_pointcloud(
    pointcloud: Pointclouds,
    image_size: int = 256,
    elev: float = 30.0,
    azim: float = 45.0,
    dist: float = 1.0,
    background_color: Tuple[float, float, float] = (1, 1, 1),
) -> np.ndarray:
    """
    Render a PyTorch3D Pointclouds object using PyTorch3D's points renderer,
    and display the rendered image on the given matplotlib axis.

    Args:
        pointcloud (Pointclouds): A PyTorch3D Pointclouds object.
        image_size (int): The size (in pixels) of the output image.
        elev (float): Elevation angle (degrees) for the camera.
        azim (float): Azimuth angle (degrees) for the camera.
        dist (float): Distance from the camera to the object.
        background_color (tuple): Background color for the renderer.

    Returns:
        np.ndarray: the rendered pointcloud image.
    """
    # Get device from the pointcloud.
    device = pointcloud.points_packed().device

    # Set up camera.
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

    # Set up points rasterizer and renderer.
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=0.01,
    )
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )

    # Render the pointcloud.
    return renderer(pointcloud, cameras=cameras)[0, ..., :3].detach().cpu().numpy()


def render_mesh_raw(mesh: Dict[str, Tensor], *args, **kwargs) -> np.ndarray:
    """
    Render a mesh (given as a dictionary with keys "vertices" and "faces")
    using PyTorch3D and plot the rendered image on the provided matplotlib axis.

    Args:
        mesh (Dict[str, Tensor]): A dictionary with keys:
            - "vertices": Tensor of shape (N, 3) for vertex coordinates.
            - "faces": Tensor of shape (M, 3) for face indices.
        image_size (int): The size (width and height in pixels) of the rendered image.
        elev (float): Elevation angle in degrees for the camera.
        azim (float): Azimuth angle in degrees for the camera.
        dist (float): Distance from the camera to the object.

    Returns:
        np.ndarray: the rendered mesh image.
    """
    # Obtain the device from the mesh vertices.
    device = mesh["vertices"].device

    # Ensure vertices and faces are on the correct device.
    vertices = mesh["vertices"].to(device)
    faces = mesh["faces"].to(device)

    # Create a Meshes object with a uniform mid-gray texture.
    colors = torch.full((1, vertices.shape[0], 3), 0.7, device=device)  # mid-gray
    mesh_p3d = Meshes(
        verts=[vertices], faces=[faces], textures=TexturesVertex(verts_features=colors)
    )

    return render_mesh(mesh_p3d, *args, **kwargs)


def render_mesh(
    mesh: Meshes,
    image_size: int = 256,
    elev: float = 30.0,
    azim: float = 45.0,
    dist: float = 1.0,
) -> np.ndarray:
    """
    Render a PyTorch3D Meshes object using PyTorch3D and plot the rendered image
    on the provided matplotlib axis. If the mesh has no texture, assign a uniform grey texture.

    Args:
        mesh (Meshes): A PyTorch3D Meshes object.
        image_size (int): The width and height (in pixels) of the rendered image.
        elev (float): The elevation angle in degrees for the camera.
        azim (float): The azimuth angle in degrees for the camera.
        dist (float): The distance from the camera to the object.

    Returns:
        np.ndarray: the rendered mesh image.
    """
    # Obtain device from the mesh (default to CPU if not available).
    device = mesh.device if hasattr(mesh, "device") else torch.device("cpu")

    # If mesh has no textures, assign a uniform grey texture.
    if mesh.textures is None:
        verts = mesh.verts_packed()  # shape: (N, 3)
        colors = torch.full((1, verts.shape[0], 3), 0.7, device=device)  # mid-grey
        mesh = Meshes(
            verts=mesh.verts_list(),
            faces=mesh.faces_list(),
            textures=TexturesVertex(verts_features=colors),
        )

    # Set up camera parameters.
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

    # Set up lights and rasterization settings.
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1
    )

    # Create a mesh renderer.
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
    )

    # Render the mesh.
    return renderer(mesh)[0, ..., :3].detach().cpu().numpy()


def render_voxel_raw(
    voxels: Tensor,
    image_size: int = 256,
    elev: float = 30.0,
    azim: float = 45.0,
    dist: float = 1.0,
    extent: float = 1.0,
) -> np.ndarray:
    """
    Render a voxel grid using PyTorch3D's VolumeRenderer and plot the rendered image
    on the given matplotlib axis.

    The voxel grid is assumed to be of shape (B, 1, H, W, D) (with H == W == D).
    The parameter `extent` specifies the world-space size (in each dimension) that the voxel grid spans.
    For example, if extent=2.0 and H=32, then the voxel grid spans [0, 2] in each dimension,
    and each voxel has a size of (2.0/32).

    The camera is adjusted so that it looks at the center of the volume.

    Args:
        voxels (Tensor): A tensor of shape (B, 1, H, W, D) representing the voxel grid.
        image_size (int): The size (width and height) of the rendered image.
        elev (float): Elevation angle in degrees for the camera.
        azim (float): Azimuth angle in degrees for the camera.
        dist (float): Distance from the camera to the object.
        extent (float): The world-space size that the voxel grid spans (assumes cubic grid).
        background_color (tuple): Background color for the renderer.

    Returns:
        np.ndarray: the rendered voxel image.
    """
    # Get device from voxels tensor.
    device = voxels.device
    H, W, D = voxels.shape[-3:]
    voxels = voxels.reshape(-1, 1, H, W, D)
    # Compute voxel size so that the grid spans [0, extent] in each dimension.
    voxel_size = extent / H

    # Create the Volumes object.
    volume = Volumes(densities=voxels, voxel_size=voxel_size)

    # Set up camera parameters.
    # The Volumes object spans [0, extent]^3, so its center is at (extent/2, extent/2, extent/2).
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

    # Set up the volume renderer.
    raysampler = NDCGridRaysampler(
        image_width=image_size,
        image_height=image_size,
        n_pts_per_ray=50,
        min_depth=0.1,
        max_depth=4.0,
    )
    raymarcher = EmissionAbsorptionRaymarcher()
    vol_renderer = VolumeRenderer(raysampler=raysampler, raymarcher=raymarcher)

    # Render the volume.
    return (
        vol_renderer(volumes=volume, cameras=cameras)[0][0, ..., :3]
        .detach()
        .cpu()
        .numpy()
    )
