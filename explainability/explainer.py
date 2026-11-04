"""
Unified Explainable AI (XAI) Suite for Brain Tumor Diagnosis.
Produces multi-modal visualizations combining CNN Grad-CAM and ViT Attention maps
alongside clinical interpretability reports and medical disclaimers.
"""

import logging
from typing import Dict, Any, Union, Optional
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from .gradcam import GradCAM, overlay_heatmap_on_image
from .vit_attention import ViTAttentionVisualizer
from src.dataset import ID_TO_CLASS, get_transforms

logger = logging.getLogger("brain_tumor_project.explainability.explainer")

MEDICAL_DISCLAIMER = (
    "RESEARCH PROTOTYPE DISCLAIMER: This explainability framework and deep learning model "
    "are developed strictly for academic, educational, and computational research purposes. "
    "The generated heatmaps, saliency overlays, and class predictions reflect mathematical feature "
    "attributions of artificial neural networks and DO NOT constitute clinical diagnoses, "
    "radiological evaluations, or medical advice. A licensed radiologist or healthcare provider "
    "must interpret real-world diagnostic neuroimaging."
)


class BrainTumorExplainer:
    """
    High-level orchestrator for multimodal model interpretability.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        image_size: int = 224
    ):
        self.model = model.to(device)
        self.device = device
        self.image_size = image_size
        self.transform = get_transforms(image_size=image_size, is_training=False)

        # Grad-CAM for CNN branch
        target_layer = self.model.get_gradcam_target_layer()
        self.gradcam = GradCAM(model=self.model, target_layer=target_layer)

        # ViT Attention Visualizer
        self.vit_vis = ViTAttentionVisualizer(model=self.model, image_size=image_size)

    def explain(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        target_class: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Runs comprehensive explainability analysis on an MRI scan.
        Args:
            image_input: File path, PIL Image, or RGB numpy array.
            target_class: Optional target class index. If None, top predicted class is used.
        Returns:
            Dictionary containing prediction, probabilities, heatmaps, and overlays.
        """
        # Load and convert image to RGB PIL Image
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert("RGB")
        else:
            pil_img = image_input.convert("RGB")

        # Resize original image to match model input dimensions
        pil_resized = pil_img.resize((self.image_size, self.image_size))
        rgb_original = np.array(pil_resized)

        # Preprocess tensor for model: [1, 3, 224, 224]
        tensor = self.transform(pil_resized).unsqueeze(0).to(self.device)

        # 1. Prediction Pass
        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        top_pred_id = int(np.argmax(probs))
        class_to_explain = top_pred_id if target_class is None else target_class

        # 2. CNN Grad-CAM
        # Need gradients enabled for Grad-CAM
        tensor_cam = tensor.clone().detach()
        cnn_heatmap, _, _ = self.gradcam.generate(tensor_cam, target_class=class_to_explain)
        cnn_overlay = overlay_heatmap_on_image(rgb_original, cnn_heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET)

        # 3. ViT Attention Map
        vit_heatmap = self.vit_vis.generate(tensor)
        vit_overlay = self.vit_vis.overlay(rgb_original, vit_heatmap, alpha=0.5, colormap=cv2.COLORMAP_VIRIDIS)

        # 4. Hybrid Fused Saliency (Combined local + global)
        hybrid_heatmap = 0.5 * cnn_heatmap + 0.5 * vit_heatmap
        hybrid_heatmap = (hybrid_heatmap - hybrid_heatmap.min()) / (hybrid_heatmap.max() - hybrid_heatmap.min() + 1e-8)
        hybrid_overlay = overlay_heatmap_on_image(rgb_original, hybrid_heatmap, alpha=0.5, colormap=cv2.COLORMAP_TURBO if hasattr(cv2, 'COLORMAP_TURBO') else cv2.COLORMAP_JET)

        # Build class probabilities dictionary
        class_probabilities = {ID_TO_CLASS[i]: float(probs[i]) for i in range(len(probs))}

        return {
            "predicted_class_id": top_pred_id,
            "predicted_class_name": ID_TO_CLASS[top_pred_id],
            "confidence": float(probs[top_pred_id]),
            "class_probabilities": class_probabilities,
            "original_image": rgb_original,
            "cnn_gradcam_heatmap": cnn_heatmap,
            "cnn_gradcam_overlay": cnn_overlay,
            "vit_attention_heatmap": vit_heatmap,
            "vit_attention_overlay": vit_overlay,
            "hybrid_fused_heatmap": hybrid_heatmap,
            "hybrid_fused_overlay": hybrid_overlay,
            "medical_disclaimer": MEDICAL_DISCLAIMER
        }

    def save_explanation_figure(
        self,
        explanation_result: Dict[str, Any],
        output_path: str = "results/figures/explanation_quad.png"
    ) -> str:
        """
        Creates and saves a publication-quality 4-panel comparison figure.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        pred_name = explanation_result["predicted_class_name"]
        conf = explanation_result["confidence"] * 100.0

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f"Hybrid Explainability Analysis - Prediction: {pred_name} ({conf:.1f}% Confidence)", fontsize=16, fontweight="bold")

        axes[0].imshow(explanation_result["original_image"])
        axes[0].set_title("Original MRI Scan", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(explanation_result["cnn_gradcam_overlay"])
        axes[1].set_title("CNN Grad-CAM (Local Features)", fontsize=12)
        axes[1].axis("off")

        axes[2].imshow(explanation_result["vit_attention_overlay"])
        axes[2].set_title("ViT Attention (Global Context)", fontsize=12)
        axes[2].axis("off")

        axes[3].imshow(explanation_result["hybrid_fused_overlay"])
        axes[3].set_title("Hybrid Saliency Fusion", fontsize=12)
        axes[3].axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved explanation comparison figure to {output_path}")
        return output_path
