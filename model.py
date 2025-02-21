#!/usr/bin/env python3
import time
from collections import OrderedDict

import pytorch3d
import torch
import torch.nn as nn
from pytorch3d.utils import ico_sphere
from torchvision import models as torchvision_models
from torchvision import transforms


class VoxelBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.ConvTranspose3d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            # bias=False,
                        ),
                    ),
                    # ("batchnorm", nn.BatchNorm3d(out_channels)),
                    ("activation", nn.ReLU(inplace=True)),
                ]
            )
        )


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2, mode="trilinear", align_corners=False
        )
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


class VoxelDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Project latent vector to a small 3D volume
        self.fc = nn.Linear(512, 256 * 2 * 2 * 2)
        # Upsample with 3D transposed convolutions
        self.deconv = nn.Sequential(
            OrderedDict(
                [
                    ("vox1", VoxelBlock(256, 128)),
                    ("vox2", VoxelBlock(128, 64)),
                    ("vox3", UpsampleBlock(64, 32)),
                    ("vox4", VoxelBlock(32, 1)),
                    (
                        "binarize",
                        nn.Sigmoid(),
                    ),
                ]
            )
        )

    def forward(self, x):
        # x: (b, 512)
        x = self.fc(x)  # -> (b, 256*2*2*2)
        x = x.view(-1, 256, 2, 2, 2)  # Reshape to 3D volume
        x = self.deconv(x)  # -> (b, 1, 32, 32, 32)
        return x


class PointCloudDecoder(nn.Module):
    def __init__(self, n_points, latent_dim=512):
        """
        Args:
            n_points (int): Number of points in the predicted point cloud.
            latent_dim (int): Dimensionality of the input latent vector.
        """
        super(PointCloudDecoder, self).__init__()
        self.n_points = n_points

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, n_points * 3),
        )

    def forward(self, x):
        # x: (B, latent_dim)
        x = self.fc(x)  # (B, n_points * 3)
        x = x.view(-1, self.n_points, 3)  # reshape to (B, n_points, 3)
        return x


class MeshDecoder(nn.Module):
    def __init__(self, n_verts, latent_dim=512):
        """
        Args:
            n_verts (int): Number of vertices in the template mesh.
            latent_dim (int): Dimensionality of the input latent vector.
        """
        super(MeshDecoder, self).__init__()
        self.n_verts = n_verts
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, n_verts * 3),
        )

    def forward(self, x):
        # x: (B, latent_dim)
        batch_size = x.shape[0]
        x = self.fc(x)  # (B, n_verts * 3)
        x = x.view(batch_size, self.n_verts, 3)  # reshape to (B, n_verts, 3)
        return x


class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

        # define decoder
        if args.type == "vox":
            # Input: b x 512
            # Output: b x 32 x 32 x 32
            self.decoder = VoxelDecoder()
        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3
            self.n_point = args.n_points
            self.decoder = PointCloudDecoder(n_points=args.n_points, latent_dim=512)
        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3
            # try different mesh initializations
            template_mesh = ico_sphere(4, self.device)
            self.n_verts = template_mesh.verts_packed().shape[0]
            self.mesh_pred = pytorch3d.structures.Meshes(
                template_mesh.verts_list() * args.batch_size,
                template_mesh.faces_list() * args.batch_size,
            )
            self.decoder = MeshDecoder(n_verts=self.n_verts, latent_dim=512)

    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0, 3, 1, 2))
            encoded_feat = (
                self.encoder(images_normalize).squeeze(-1).squeeze(-1)
            )  # b x 512
        else:
            encoded_feat = images  # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        if args.type == "vox":
            voxels_pred = self.decoder(encoded_feat)
            return voxels_pred

        elif args.type == "point":
            pointclouds_pred = self.decoder(encoded_feat)
            return pointclouds_pred

        elif args.type == "mesh":
            deform_vertices_pred = self.decoder(encoded_feat)
            mesh_pred = self.mesh_pred.offset_verts(
                deform_vertices_pred.reshape([-1, 3])
            )
            return mesh_pred
