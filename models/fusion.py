"""
Feature Fusion Modules for Hybrid CNN-Vision Transformer Architecture.
Supports Cross-Attention Fusion, Gated Fusion, and Concatenation with MLP Head.
"""

import logging
from typing import Tuple
import torch
import torch.nn as nn

logger = logging.getLogger("brain_tumor_project.models.fusion")


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion Module.
    Projects CNN and ViT feature vectors into a shared latent space and
    fuses them using Multi-Head Bidirectional Self/Cross-Attention, LayerNorm,
    and a Feedforward Network (FFN) with residual connections.
    """
    def __init__(
        self,
        cnn_dim: int = 512,
        vit_dim: int = 768,
        projection_dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        self.projection_dim = projection_dim

        # Projections to shared latent space
        self.proj_cnn = nn.Sequential(
            nn.Linear(cnn_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.proj_vit = nn.Sequential(
            nn.Linear(vit_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Multi-Head Attention layer treating [CNN, ViT] as a sequence of length 2
        self.mha = nn.MultiheadAttention(
            embed_dim=projection_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(projection_dim)

        # Feedforward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(projection_dim, projection_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim * 2, projection_dim)
        )
        self.norm2 = nn.LayerNorm(projection_dim)

        # Fused output dimension (2 tokens * projection_dim)
        self.output_dim = projection_dim * 2

    def forward(self, cnn_feat: torch.Tensor, vit_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cnn_feat: [B, cnn_dim]
            vit_feat: [B, vit_dim]
        Returns:
            fused_representation: [B, output_dim]
        """
        # Project: [B, D_proj]
        z_cnn = self.proj_cnn(cnn_feat).unsqueeze(1)  # [B, 1, D_proj]
        z_vit = self.proj_vit(vit_feat).unsqueeze(1)  # [B, 1, D_proj]

        # Stack into sequence: [B, 2, D_proj]
        tokens = torch.cat([z_cnn, z_vit], dim=1)

        # Attention + Residual
        attn_out, _ = self.mha(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)

        # FFN + Residual
        ffn_out = self.ffn(tokens)
        tokens = self.norm2(tokens + ffn_out)

        # Flatten sequence into single fused representation: [B, 2 * D_proj]
        fused = torch.flatten(tokens, start_dim=1)
        return fused


class GatedFusion(nn.Module):
    """
    Gated Fusion Module.
    Dynamically balances CNN local spatial representations and ViT global representations
    via a learned element-wise gating mechanism.
    """
    def __init__(
        self,
        cnn_dim: int = 512,
        vit_dim: int = 768,
        projection_dim: int = 512,
        dropout: float = 0.3
    ):
        super().__init__()
        self.projection_dim = projection_dim
        self.proj_cnn = nn.Sequential(
            nn.Linear(cnn_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.proj_vit = nn.Sequential(
            nn.Linear(vit_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Learned gate vector
        self.gate_layer = nn.Sequential(
            nn.Linear(projection_dim * 2, projection_dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(projection_dim)
        self.output_dim = projection_dim

    def forward(self, cnn_feat: torch.Tensor, vit_feat: torch.Tensor) -> torch.Tensor:
        p_cnn = self.proj_cnn(cnn_feat)
        p_vit = self.proj_vit(vit_feat)

        gate = self.gate_layer(torch.cat([p_cnn, p_vit], dim=1))
        fused = gate * p_cnn + (1.0 - gate) * p_vit
        return self.norm(fused)


class ConcatenationFusion(nn.Module):
    """
    Simple projection and concatenation baseline fusion module.
    """
    def __init__(
        self,
        cnn_dim: int = 512,
        vit_dim: int = 768,
        projection_dim: int = 512,
        dropout: float = 0.3
    ):
        super().__init__()
        self.proj_cnn = nn.Sequential(
            nn.Linear(cnn_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU()
        )
        self.proj_vit = nn.Sequential(
            nn.Linear(vit_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU()
        )
        self.output_dim = projection_dim * 2
        self.dropout = nn.Dropout(dropout)

    def forward(self, cnn_feat: torch.Tensor, vit_feat: torch.Tensor) -> torch.Tensor:
        p_cnn = self.proj_cnn(cnn_feat)
        p_vit = self.proj_vit(vit_feat)
        fused = torch.cat([p_cnn, p_vit], dim=1)
        return self.dropout(fused)


class ClassificationHead(nn.Module):
    """
    Multi-layer Perceptron (MLP) classification head for 4 brain tumor classes:
    [Glioma, Meningioma, Pituitary, No Tumor].
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
