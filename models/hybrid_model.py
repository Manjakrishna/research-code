"""
Explainable Hybrid CNN–Vision Transformer Model for Brain Tumor Classification.
Integrates local CNN representations with global ViT self-attention representations
through attention-based feature fusion.
"""

import logging
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn

from .cnn_branch import CNNBranch
from .vit_branch import ViTBranch
from .fusion import CrossAttentionFusion, GatedFusion, ConcatenationFusion, ClassificationHead

logger = logging.getLogger("brain_tumor_project.models.hybrid")


class HybridBrainTumorModel(nn.Module):
    """
    Dual-branch Hybrid Architecture:
      - Branch 1: CNN (ResNet18 / ResNet50) -> Local inductive bias, lesion edge & texture features.
      - Branch 2: ViT (ViT-B/16) -> Global self-attention, long-range anatomical context.
      - Fusion: Cross-Attention / Gated / Concatenation -> Classification Head (4 Classes).
    """
    def __init__(
        self,
        cnn_backbone: str = "resnet18",
        cnn_pretrained: bool = True,
        vit_backbone: str = "vit_b_16",
        vit_pretrained: bool = True,
        fusion_type: str = "cross_attention",
        projection_dim: int = 512,
        num_heads: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        num_classes: int = 4,
        freeze_early_layers: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.fusion_type = fusion_type.lower()

        # 1. CNN Branch
        self.cnn_branch = CNNBranch(
            backbone=cnn_backbone,
            pretrained=cnn_pretrained,
            freeze_early_layers=freeze_early_layers
        )

        # 2. ViT Branch
        self.vit_branch = ViTBranch(
            backbone=vit_backbone,
            pretrained=vit_pretrained,
            freeze_early_layers=freeze_early_layers
        )

        # 3. Fusion Module
        if self.fusion_type == "cross_attention":
            self.fusion = CrossAttentionFusion(
                cnn_dim=self.cnn_branch.feature_dim,
                vit_dim=self.vit_branch.embedding_dim,
                projection_dim=projection_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        elif self.fusion_type == "gated":
            self.fusion = GatedFusion(
                cnn_dim=self.cnn_branch.feature_dim,
                vit_dim=self.vit_branch.embedding_dim,
                projection_dim=projection_dim,
                dropout=dropout
            )
        elif self.fusion_type == "concatenation":
            self.fusion = ConcatenationFusion(
                cnn_dim=self.cnn_branch.feature_dim,
                vit_dim=self.vit_branch.embedding_dim,
                projection_dim=projection_dim,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}. Choose 'cross_attention', 'gated', or 'concatenation'.")

        # 4. Classification Head
        self.classifier = ClassificationHead(
            input_dim=self.fusion.output_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout
        )

        logger.info(
            f"Initialized HybridBrainTumorModel: "
            f"CNN={cnn_backbone}, ViT={vit_backbone}, Fusion={fusion_type}, "
            f"FusedDim={self.fusion.output_dim}, Classes={num_classes}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass for training and inference.
        Args:
            x: Input batch of MRI images [Batch, 3, 224, 224]
        Returns:
            logits: Unnormalized class logits [Batch, 4]
        """
        cnn_feat = self.cnn_branch(x)         # [B, cnn_dim]
        vit_feat = self.vit_branch(x)         # [B, vit_dim]
        fused = self.fusion(cnn_feat, vit_feat)  # [B, fused_dim]
        logits = self.classifier(fused)       # [B, num_classes]
        return logits

    def forward_with_features(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning intermediate representations for analysis and explainability.
        Returns:
            logits: [B, 4]
            cnn_feat: [B, cnn_dim]
            vit_feat: [B, vit_dim]
            fused: [B, fused_dim]
        """
        cnn_feat = self.cnn_branch(x)
        vit_feat = self.vit_branch(x)
        fused = self.fusion(cnn_feat, vit_feat)
        logits = self.classifier(fused)
        return logits, cnn_feat, vit_feat, fused

    def get_gradcam_target_layer(self) -> nn.Module:
        """Returns the final convolutional block of the CNN branch for Grad-CAM."""
        return self.cnn_branch.get_last_conv_layer()

    def get_vit_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts self-attention weights from the ViT branch for Explainable AI."""
        return self.vit_branch.get_last_layer_attention(x)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HybridBrainTumorModel":
        """Factory method to instantiate model directly from a parsed config dictionary."""
        model_cfg = config.get("model", {})
        fusion_cfg = model_cfg.get("fusion", {})
        return cls(
            cnn_backbone=model_cfg.get("cnn_backbone", "resnet18"),
            cnn_pretrained=model_cfg.get("cnn_pretrained", True),
            vit_backbone=model_cfg.get("vit_backbone", "vit_b_16"),
            vit_pretrained=model_cfg.get("vit_pretrained", True),
            fusion_type=fusion_cfg.get("type", "cross_attention"),
            projection_dim=fusion_cfg.get("projection_dim", 512),
            num_heads=fusion_cfg.get("num_heads", 4),
            hidden_dim=fusion_cfg.get("hidden_dim", 256),
            dropout=fusion_cfg.get("dropout", 0.3),
            num_classes=fusion_cfg.get("num_classes", 4)
        )
