"""
CNN Branch for Local Feature Extraction from Brain MRI Scans.
Supports ResNet18 and ResNet50 with transfer learning.
"""

import logging
from typing import Tuple, Optional
import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger("brain_tumor_project.models.cnn")


class CNNBranch(nn.Module):
    """
    Convolutional Neural Network branch that extracts local spatial MRI features.
    Retains access to the final convolutional feature maps for Grad-CAM explainability.
    """
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_early_layers: bool = False
    ):
        super().__init__()
        self.backbone_name = backbone.lower()
        self.pretrained = pretrained

        if self.backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = models.resnet18(weights=weights)
            self.feature_dim = 512
        elif self.backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported CNN backbone: {backbone}. Choose 'resnet18' or 'resnet50'.")

        # Extract stages:
        # Initial stem
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # Residual stages
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Adaptive global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if freeze_early_layers:
            self._freeze_layers([self.conv1, self.bn1, self.layer1, self.layer2])
            logger.info("Froze stem and initial residual blocks (conv1..layer2) for transfer learning.")

        logger.info(f"Initialized CNNBranch ({self.backbone_name}, pretrained={pretrained}, feature_dim={self.feature_dim})")

    def _freeze_layers(self, modules):
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def get_last_conv_layer(self) -> nn.Module:
        """Returns the final convolutional block (layer4) for Grad-CAM hook registration."""
        return self.layer4

    def forward(self, x: torch.Tensor, return_spatial_maps: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        Args:
            x: Input tensor [Batch, 3, 224, 224]
            return_spatial_maps: If True, returns spatial feature maps from layer4.
        Returns:
            pooled_features: [Batch, feature_dim]
            spatial_maps: [Batch, feature_dim, 7, 7] (if return_spatial_maps is True, else None)
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        spatial_maps = self.layer4(x)  # [B, feature_dim, H', W'] (e.g. 7x7 for 224x224 input)

        # Global average pool
        pooled = self.avgpool(spatial_maps)
        pooled_features = torch.flatten(pooled, 1)  # [B, feature_dim]

        if return_spatial_maps:
            return pooled_features, spatial_maps
        return pooled_features
