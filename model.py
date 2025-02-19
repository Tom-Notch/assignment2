#!/usr/bin/env python3
import time

import pytorch3d
import torch
import torch.nn as nn
from pytorch3d.utils import ico_sphere
from torchvision import models as torchvision_models
from torchvision import transforms


class VoxelDecoder(nn.Module):
    def __init__(self):
        super(VoxelDecoder, self).__init__()
        # Project latent vector to a small 3D volume
        self.fc = nn.Linear(512, 256 * 4 * 4 * 4)
        # Upsample with 3D transposed convolutions:
        #   4 -> 8 -> 16 -> 32 (in each spatial dimension)
        self.deconv = nn.Sequential(
            # Input: (b, 256, 4, 4, 4)
            nn.ConvTranspose3d(
                256, 128, kernel_size=4, stride=2, padding=1
            ),  # -> (b, 128, 8, 8, 8)
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),  # -> (b, 64, 16, 16, 16)
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(
                64, 1, kernel_size=4, stride=2, padding=1
            ),  # -> (b, 1, 32, 32, 32)
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (b, 512)
        x = self.fc(x)  # -> (b, 256*4*4*4)
        x = x.view(-1, 256, 4, 4, 4)  # Reshape to 3D volume
        x = self.deconv(x)  # -> (b, 1, 32, 32, 32)
        x = x.squeeze(1)  # Remove the channel dimension: (b, 32, 32, 32)
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
            # TODO:
            # self.decoder =
        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3
            # try different mesh initializations
            mesh_pred = ico_sphere(4, self.device)
            self.mesh_pred = pytorch3d.structures.Meshes(
                mesh_pred.verts_list() * args.batch_size,
                mesh_pred.faces_list() * args.batch_size,
            )
            # TODO:
            # self.decoder =

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
            # TODO:
            # pointclouds_pred =
            return pointclouds_pred

        elif args.type == "mesh":
            # TODO:
            # deform_vertices_pred =
            mesh_pred = self.mesh_pred.offset_verts(
                deform_vertices_pred.reshape([-1, 3])
            )
            return mesh_pred
