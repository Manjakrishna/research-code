"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for the CNN branch.
Generates interpretable visual heatmaps highlighting salient anatomical regions
that influenced the brain tumor classification decision.
"""

import logging
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("brain_tumor_project.explainability.gradcam")


class GradCAM:
    """
    Grad-CAM implementation for convolutional neural network architectures.
    Computes class activation maps using gradients flowing into the final conv layer.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            # output may be a tensor or tuple
            if isinstance(output, tuple):
                self.activations = output[0].detach()
            else:
                self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0] is gradient w.r.t layer output
            self.gradients = grad_output[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self._hooks.extend([h1, h2])

    def remove_hooks(self):
        """Remove registered hooks to prevent memory leaks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generates 2D Grad-CAM heatmap normalized to [0, 1].
        Args:
            input_tensor: [1, 3, H, W] normalized PyTorch tensor on same device as model.
            target_class: Target class index. If None, uses top predicted class.
        Returns:
            heatmap: [H, W] float32 numpy array with values in [0.0, 1.0]
            predicted_class: Integer predicted class index
            confidence: Float probability of predicted class
        """
        self.model.eval()
        input_tensor.requires_grad_(True)

        # Forward pass
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(torch.argmax(probs, dim=1).item())

        confidence = float(probs[0, target_class].item())

        # Zero gradients and backpropagate target score
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to capture gradients or activations.")

        # Channel-wise global average pooling of gradients: [1, C, 1, 1]
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3], keepdim=True)

        # Weight activations by pooled gradients: [C, H_act, W_act]
        weighted_activations = self.activations[0] * pooled_gradients[0]
        cam = torch.sum(weighted_activations, dim=0)

        # Apply ReLU to retain only positive evidence towards the class
        cam = F.relu(cam)

        # Interpolate to match input spatial dimensions (H, W)
        _, _, h, w = input_tensor.shape
        cam = cam.unsqueeze(0).unsqueeze(0)  # [1, 1, H_act, W_act]
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 1e-8:
            heatmap = (cam - cam_min) / (cam_max - cam_min)
        else:
            heatmap = np.zeros_like(cam)

        return heatmap.astype(np.float32), target_class, confidence


def overlay_heatmap_on_image(
    image: Union[np.ndarray, Image.Image],
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlays a [0, 1] normalized heatmap onto an RGB image.
    Args:
        image: Original image as PIL Image or uint8 RGB numpy array.
        heatmap: 2D float32 array in [0, 1].
        alpha: Blending weight for heatmap (1-alpha for original image).
        colormap: OpenCV colormap constant.
    Returns:
        blended_rgb: uint8 RGB numpy array.
    """
    if isinstance(image, Image.Image):
        image_np = np.array(image.convert("RGB"))
    else:
        image_np = image.copy()
        if image_np.dtype != np.uint8:
            image_np = (image_np * 255).astype(np.uint8)

    h, w, _ = image_np.shape

    # Ensure heatmap matches image spatial size
    if (heatmap.shape[0], heatmap.shape[1]) != (h, w):
        heatmap = cv2.resize(heatmap, (w, h))

    # Scale to 0-255 uint8 and apply colormap
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend
    blended = cv2.addWeighted(heatmap_rgb, alpha, image_np, 1.0 - alpha, 0)
    return blended
