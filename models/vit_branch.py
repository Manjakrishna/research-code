"""
Vision Transformer (ViT) Branch for Global Context Extraction.
Extracts global attention-based features and attention weights from brain MRI scans.
"""

import logging
from typing import Tuple, Optional
import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger("brain_tumor_project.models.vit")


class ViTBranch(nn.Module):
    """
    Vision Transformer branch using pretrained ViT-B/16 (or ViT-B/32).
    Extracts global image features from the [CLS] token and captures
    multi-head self-attention maps for explainable AI.
    """
    def __init__(
        self,
        backbone: str = "vit_b_16",
        pretrained: bool = True,
        freeze_early_layers: bool = False
    ):
        super().__init__()
        self.backbone_name = backbone.lower()
        self.pretrained = pretrained

        if self.backbone_name == "vit_b_16":
            weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
            vit = models.vit_b_16(weights=weights)
            self.embedding_dim = 768
            self.patch_size = 16
        elif self.backbone_name == "vit_b_32":
            weights = models.ViT_B_32_Weights.DEFAULT if pretrained else None
            vit = models.vit_b_32(weights=weights)
            self.embedding_dim = 768
            self.patch_size = 32
        else:
            raise ValueError(f"Unsupported ViT backbone: {backbone}. Choose 'vit_b_16' or 'vit_b_32'.")

        self.conv_proj = vit.conv_proj
        self.class_token = vit.class_token
        self.encoder = vit.encoder
        # Exclude default 1000-class ImageNet head
        self.heads = nn.Identity()

        # Cache for attention weights from the last encoder layer
        self._attention_weights: Optional[torch.Tensor] = None

        if freeze_early_layers:
            # Freeze patch projection and first half of transformer layers (0..5)
            for param in self.conv_proj.parameters():
                param.requires_grad = False
            for layer in self.encoder.layers[:6]:
                for param in layer.parameters():
                    param.requires_grad = False
            logger.info("Froze patch projection and first 6 ViT encoder layers for transfer learning.")

        logger.info(f"Initialized ViTBranch ({self.backbone_name}, pretrained={pretrained}, embedding_dim={self.embedding_dim})")

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        """Projects image patches and prepends the [CLS] token."""
        n, c, h, w = x.shape
        p = self.patch_size
        assert h % p == 0 and w % p == 0, f"Input image dimensions ({h}, {w}) must be divisible by patch size {p}"

        # (n, hidden_dim, h/p, w/p) -> (n, hidden_dim, (h/p)*(w/p)) -> (n, (h/p)*(w/p), hidden_dim)
        x = self.conv_proj(x)
        x = x.reshape(n, self.embedding_dim, -1)
        x = x.permute(0, 2, 1)

        # Prepend class token
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        return x

    def forward(self, x: torch.Tensor, return_all_tokens: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        Args:
            x: Input tensor [Batch, 3, 224, 224]
            return_all_tokens: If True, returns all sequence tokens [Batch, 197, 768].
        Returns:
            cls_token: [Batch, 768] (global representation)
            all_tokens: [Batch, NumPatches+1, 768] if return_all_tokens is True else None
        """
        x = self._process_input(x)
        x = self.encoder(x)  # Passes through positional embeddings and encoder layers
        cls_token = x[:, 0]  # [CLS] token at index 0

        if return_all_tokens:
            return cls_token, x
        return cls_token

    def get_last_layer_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes attention matrix from the final transformer encoder block.
        Returns:
            attention_matrix: [Batch, num_heads, num_tokens, num_tokens]
        """
        with torch.no_grad():
            x = self._process_input(x)
            # Pass through positional embedding and dropout in encoder
            if hasattr(self.encoder, "pos_embedding"):
                x = x + self.encoder.pos_embedding
            x = self.encoder.dropout(x)

            # Pass through all encoder layers except the last
            num_layers = len(self.encoder.layers)
            for i in range(num_layers - 1):
                x = self.encoder.layers[i](x)

            # Last layer
            last_layer = self.encoder.layers[-1]
            ln_1 = last_layer.ln_1(x)
            mha = last_layer.self_attention

            # Compute Q, K, V from in_proj_weight / in_proj_bias
            embed_dim = self.embedding_dim
            num_heads = mha.num_heads
            head_dim = embed_dim // num_heads

            qkv = nn.functional.linear(ln_1, mha.in_proj_weight, mha.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)

            # Reshape for multi-head attention: [B, num_heads, seq_len, head_dim]
            b, seq_len, _ = q.shape
            q = q.view(b, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(b, seq_len, num_heads, head_dim).transpose(1, 2)

            # Attention scores: Q @ K^T / sqrt(head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
            attn_weights = nn.functional.softmax(scores, dim=-1)

            return attn_weights
