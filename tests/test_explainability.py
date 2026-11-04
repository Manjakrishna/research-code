"""
Unit tests for Explainable AI (Grad-CAM, ViT Attention, and Unified Explainer).
"""

import pytest
import numpy as np
from PIL import Image
import torch

from models.hybrid_model import HybridBrainTumorModel
from explainability.gradcam import GradCAM, overlay_heatmap_on_image
from explainability.vit_attention import ViTAttentionVisualizer
from explainability.explainer import BrainTumorExplainer


@pytest.fixture
def mock_hybrid_model():
    model = HybridBrainTumorModel(
        cnn_backbone="resnet18",
        cnn_pretrained=False,
        vit_backbone="vit_b_16",
        vit_pretrained=False,
        projection_dim=128,
        num_classes=4
    )
    model.eval()
    return model


def test_gradcam(mock_hybrid_model):
    """Verify Grad-CAM produces 224x224 normalized heatmap and overlays."""
    target_layer = mock_hybrid_model.get_gradcam_target_layer()
    gradcam = GradCAM(mock_hybrid_model, target_layer)

    x = torch.randn(1, 3, 224, 224)
    heatmap, pred_class, conf = gradcam.generate(x)

    assert heatmap.shape == (224, 224)
    assert isinstance(heatmap, np.ndarray)
    assert 0 <= pred_class < 4
    assert 0.0 <= conf <= 1.0
    assert 0.0 <= heatmap.min()
    assert heatmap.max() <= 1.0 + 1e-6

    # Test overlay
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    blended = overlay_heatmap_on_image(dummy_img, heatmap)
    assert blended.shape == (224, 224, 3)
    assert blended.dtype == np.uint8

    gradcam.remove_hooks()


def test_vit_attention(mock_hybrid_model):
    """Verify ViT attention visualizer generates 224x224 heatmap from self-attention."""
    vis = ViTAttentionVisualizer(mock_hybrid_model, image_size=224, patch_size=16)

    x = torch.randn(1, 3, 224, 224)
    heatmap = vis.generate(x)

    assert heatmap.shape == (224, 224)
    assert 0.0 <= heatmap.min()
    assert heatmap.max() <= 1.0 + 1e-6

    dummy_img = Image.new("RGB", (224, 224), color=(50, 50, 50))
    overlay = vis.overlay(dummy_img, heatmap)
    assert overlay.shape == (224, 224, 3)
    assert overlay.dtype == np.uint8


def test_brain_tumor_explainer(mock_hybrid_model):
    """Verify the unified explainer produces complete diagnostic results and medical disclaimer."""
    device = torch.device("cpu")
    explainer = BrainTumorExplainer(mock_hybrid_model, device=device, image_size=224)

    test_img = Image.new("RGB", (224, 224), color=(70, 70, 70))
    result = explainer.explain(test_img)

    required_keys = [
        "predicted_class_id",
        "predicted_class_name",
        "confidence",
        "class_probabilities",
        "original_image",
        "cnn_gradcam_heatmap",
        "cnn_gradcam_overlay",
        "vit_attention_heatmap",
        "vit_attention_overlay",
        "hybrid_fused_heatmap",
        "hybrid_fused_overlay",
        "medical_disclaimer"
    ]

    for key in required_keys:
        assert key in result, f"Missing key in explanation result: {key}"

    assert len(result["class_probabilities"]) == 4
    assert result["predicted_class_name"] in ["Glioma", "Meningioma", "Pituitary Tumor", "No Tumor"]
    assert "DISCLAIMER" in result["medical_disclaimer"].upper()
