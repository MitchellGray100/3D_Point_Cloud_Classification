# /point_pillar/pillar_model.py
# Author: Yonghao Li (Paul)
# The pipeline that put the components together

# Reference:
# PointPillars: Fast Encoders for Object Detection from Point Clouds
# https://arxiv.org/pdf/1812.05784
# Section 2: PointPillars Network

# point_pillar/model.py
import torch
import torch.nn as nn

from .pillar_voxelizer import PillarVoxelizer
from .pillar_feature_net import PillarFeatureNet
from .pillar_scatter import PillarScatter
from .pillar_simple_backbone import SimplePillarBackbone

class PointPillarsClassifier(nn.Module):
    def __init__(self, num_classes, device, 
                 pfn_out_dim=64, backbone_base_channels=32):
        super().__init__()
        self.device = device

        self.voxelizer = PillarVoxelizer(
            x_range=(-1.0, 1.0),
            y_range=(-1.0, 1.0),
            z_range=(-1.0, 1.0),
            pillar_size=(0.1, 0.1),
            max_pillars=1024,
            max_points_per_pillar=32,
            device=device,
        )

        self.pfn = PillarFeatureNet(in_dim=8, out_dim=pfn_out_dim)
        self.scatter = PillarScatter(
            nx=self.voxelizer.nx,
            ny=self.voxelizer.ny,
        )
        self.backbone = SimplePillarBackbone(
            in_channels=pfn_out_dim,
            num_classes=num_classes,
            base_channels=backbone_base_channels,
        )

        self.to(device)

    def forward(self, points):
        """
        points: (B, N, 3) on self.device
        returns: logits: (B, num_classes)
        """
        # voxelize
        pillar_features, pillar_coords, pillar_mask = \
            self.voxelizer.voxelize_batch(points)

        # to device
        pillar_features = pillar_features.to(self.device)
        pillar_coords   = pillar_coords.to(self.device)
        pillar_mask     = pillar_mask.to(self.device)

        # PFN
        pillar_embeddings = self.pfn(pillar_features, pillar_mask)

        # scatter
        bev = self.scatter(pillar_embeddings, pillar_coords, pillar_mask)

        # 2D backbone -> logits
        logits = self.backbone(bev)
        return logits