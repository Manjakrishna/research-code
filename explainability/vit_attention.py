"""
Vision Transformer (ViT) Self-Attention Visualizer.
Computes and visualizes spatial attention maps from the multi-head self-attention
mechanisms of the Vision Transformer branch.
"""

import logging
from typing import Optional, Union, Tuple
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F

logger = logging.getLogger("brain_tumor_project.explainability.vit_attention")


class ViTAttentionVisualizer:
    """
    Extracts and maps multi-head self-attention weights from the ViT branch
    back to 2D spatial image space for brain MRI interpretability.
    """
    def __init__(self, model: torch.nn.Module, image_size: int = 224, patch_size: int = 16):
        self.model = model
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size  # e.g., 224 // 16 = 14

    def generate(
        self,
        input_tensor: torch.Tensor,
        head_reduction: str = "mean"
    ) -> np.ndarray:
        """
        Extracts the [CLS] token attention map over spatial patches.
        Args:
            input_tensor: [1, 3, H, W] PyTorch tensor.
            head_reduction: How to aggregate multiple attention heads: 'mean' or 'max'.
        Returns:
            attention_heatmap: 2D float32 array [H, W] normalized to [0, 1].
        """
        self.model.eval()
        with torch.no_grad():
            # Get attention weights: [B, num_heads, num_tokens, num_tokens]
            attn = self.model.get_vit_attention(input_tensor)

            # Reduce heads
            if head_reduction == "max":
                attn_heads, _ = torch.max(attn, dim=1)  # [B, N, N]
            else:
                attn_heads = torch.mean(attn, dim=1)   # [B, N, N]

            # Extract [CLS] token attention to image patch tokens (excluding token 0 which is CLS itself)
            # shape: [num_patches] -> [196]
            cls_attention = attn_heads[0, 0, 1:]

            # Reshape into 2D patch grid: [14, 14]
            grid = cls_attention.reshape(self.grid_size, self.grid_size)

            # Upsample to full image resolution (H, W) using bicubic interpolation
            grid_tensor = grid.unsqueeze(0).unsqueeze(0)  # [1, 1, 14, 14]
            upsampled = F.interpolate(
                grid_tensor,
                size=(self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False
            )
            heatmap = upsampled.squeeze().cpu().numpy()

            # Min-Max normalize
            h_min, h_max = np.min(heatmap), np.max(heatmap)
            if h_max - h_min > 1e-8:
                heatmap = (heatmap - h_min) / (h_max - h_min)
            else:
                heatmap = np.zeros_like(heatmap)

            return heatmap.astype(np.float32)

    def overlay(
        self,
        image: Union[np.ndarray, Image.Image],
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_VIRIDIS
    ) -> np.ndarray:
        """
        Overlays normalized attention heatmap on an RGB image.
        """
        if isinstance(image, Image.Image):
            image_np = np.array(image.convert("RGB"))
        else:
            image_np = image.copy()
            if image_np.dtype != np.uint8:
                image_np = (image_np * 255).astype(np.uint8)

        h, w, _ = image_np.shape
        if (heatmap.shape[0], heatmap.shape[1]) != (h, w):
            heatmap = cv2.resize(heatmap, (w, h))

        heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        blended = cv2.addWeighted(heatmap_rgb, alpha, image_np, 1.0 - alpha, 0)
        return blended
