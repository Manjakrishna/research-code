"""
Unit tests for CNN Branch, ViT Branch, Fusion Modules, and Hybrid Model forward passes.
"""

import pytest
import torch
from models.cnn_branch import CNNBranch
from models.vit_branch import ViTBranch
from models.fusion import CrossAttentionFusion, GatedFusion, ConcatenationFusion, ClassificationHead
from models.hybrid_model import HybridBrainTumorModel


def test_cnn_branch():
    """Verify ResNet CNN branch forward pass and spatial feature maps."""
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    cnn = CNNBranch(backbone="resnet18", pretrained=False)

    pooled = cnn(x)
    assert pooled.shape == (batch_size, 512)

    pooled2, spatial = cnn(x, return_spatial_maps=True)
    assert pooled2.shape == (batch_size, 512)
    assert spatial.shape == (batch_size, 512, 7, 7)

    # Check target layer for Grad-CAM exists
    target_layer = cnn.get_last_conv_layer()
    assert target_layer is not None


def test_vit_branch():
    """Verify ViT branch forward pass and attention weight extraction."""
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    vit = ViTBranch(backbone="vit_b_16", pretrained=False)

    cls_token = vit(x)
    assert cls_token.shape == (batch_size, 768)

    # Test attention matrix extraction
    attn_matrix = vit.get_last_layer_attention(x)
    # 197 tokens (1 CLS + 14x14 patches), 12 heads
    assert attn_matrix.shape == (batch_size, 12, 197, 197)


def test_fusion_modules():
    """Verify Cross-Attention, Gated, and Concatenation fusion blocks."""
    batch_size = 2
    cnn_feat = torch.randn(batch_size, 512)
    vit_feat = torch.randn(batch_size, 768)

    # 1. Cross-Attention Fusion
    cross_fuse = CrossAttentionFusion(cnn_dim=512, vit_dim=768, projection_dim=256, num_heads=4)
    out_cross = cross_fuse(cnn_feat, vit_feat)
    assert out_cross.shape == (batch_size, 512)  # 2 tokens * 256

    # 2. Gated Fusion
    gated_fuse = GatedFusion(cnn_dim=512, vit_dim=768, projection_dim=256)
    out_gated = gated_fuse(cnn_feat, vit_feat)
    assert out_gated.shape == (batch_size, 256)

    # 3. Concatenation Fusion
    concat_fuse = ConcatenationFusion(cnn_dim=512, vit_dim=768, projection_dim=256)
    out_concat = concat_fuse(cnn_feat, vit_feat)
    assert out_concat.shape == (batch_size, 512)

    # 4. Classification Head
    head = ClassificationHead(input_dim=512, hidden_dim=128, num_classes=4)
    logits = head(out_cross)
    assert logits.shape == (batch_size, 4)


def test_hybrid_brain_tumor_model():
    """Verify complete end-to-end Hybrid model output dimensions."""
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)

    model = HybridBrainTumorModel(
        cnn_backbone="resnet18",
        cnn_pretrained=False,
        vit_backbone="vit_b_16",
        vit_pretrained=False,
        fusion_type="cross_attention",
        projection_dim=256,
        num_classes=4
    )

    logits = model(x)
    assert logits.shape == (batch_size, 4)

    # Test forward with intermediate features
    logits2, cnn_f, vit_f, fused_f = model.forward_with_features(x)
    assert logits2.shape == (batch_size, 4)
    assert cnn_f.shape == (batch_size, 512)
    assert vit_f.shape == (batch_size, 768)
    assert fused_f.shape == (batch_size, 512)
