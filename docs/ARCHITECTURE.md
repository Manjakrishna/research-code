# Architectural Specification: Hybrid CNN–ViT Network

## 1. Network Component Specifications

### 1.1 Local Convolutional Branch (ResNet)
- **Base Architecture:** ResNet18 (configurable to ResNet50 in `configs/config.yaml`)
- **Input Tensor:** $X_{\text{input}} \in \mathbb{R}^{B \times 3 \times 224 \times 224}$
- **Stem:**
  - Conv2d: $7 \times 7$ kernel, stride 2, padding 3, 64 channels $\rightarrow [B, 64, 112, 112]$
  - BatchNorm2d + ReLU
  - MaxPool2d: $3 \times 3$ kernel, stride 2, padding 1 $\rightarrow [B, 64, 56, 56]$
- **Residual Stages:**
  - `layer1`: 2 Residual Blocks $\rightarrow [B, 64, 56, 56]$
  - `layer2`: 2 Residual Blocks $\rightarrow [B, 128, 28, 28]$
  - `layer3`: 2 Residual Blocks $\rightarrow [B, 256, 14, 14]$
  - `layer4`: 2 Residual Blocks $\rightarrow [B, 512, 7, 7]$ *(Target layer for Grad-CAM)*
- **Pooling:**
  - AdaptiveAvgPool2d((1, 1)) $\rightarrow [B, 512, 1, 1]$
  - Flatten $\rightarrow [B, 512]$

### 1.2 Global Vision Transformer Branch (ViT-B/16)
- **Base Architecture:** Vision Transformer Base 16 (`torchvision.models.vit_b_16`)
- **Input Tensor:** $X_{\text{input}} \in \mathbb{R}^{B \times 3 \times 224 \times 224}$
- **Patch Projection:**
  - Conv2d: $16 \times 16$ kernel, stride 16, 768 channels $\rightarrow [B, 768, 14, 14]$
  - Reshaped to token sequence: $[B, 196, 768]$
- **Token Embedding:**
  - Learnable `[CLS]` token concatenated at position 0 $\rightarrow [B, 197, 768]$
  - 1D Learnable Positional Embeddings added: $[B, 197, 768]$
- **Transformer Encoder Stack:**
  - 12 sequential Transformer Encoder Blocks.
  - Each block comprises:
    - LayerNorm(768)
    - Multi-Head Self-Attention (12 heads, 64-dim head dimension)
    - Residual Connection
    - LayerNorm(768)
    - MLP (Linear 768 $\rightarrow$ 3072 $\rightarrow$ GELU $\rightarrow$ Linear 768)
    - Residual Connection
- **Output:**
  - Representation at index 0 (`[CLS]` token): $[B, 768]$
  - Final attention weights matrix from 12th layer: $[B, 12, 197, 197]$

### 1.3 Feature Fusion Modules

#### Cross-Attention Fusion (Default)
1. **Projection Layers:**
   - CNN Projection: `Linear(512, 512)` $\rightarrow$ `LayerNorm(512)` $\rightarrow$ `GELU()` $\rightarrow$ `Dropout(0.3)` $\rightarrow [B, 1, 512]$
   - ViT Projection: `Linear(768, 512)` $\rightarrow$ `LayerNorm(512)` $\rightarrow$ `GELU()` $\rightarrow$ `Dropout(0.3)` $\rightarrow [B, 1, 512]$
2. **Token Concatenation:**
   - Stack along sequence dimension $\rightarrow [B, 2, 512]$
3. **Multi-Head Cross-Attention:**
   - `MultiheadAttention(embed_dim=512, num_heads=4, batch_first=True)`
   - LayerNorm + Residual connection $\rightarrow [B, 2, 512]$
4. **Feedforward Network (FFN):**
   - `Linear(512, 1024)` $\rightarrow$ `GELU()` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Linear(1024, 512)`
   - LayerNorm + Residual connection $\rightarrow [B, 2, 512]$
5. **Flattening:**
   - Flatten sequence $\rightarrow [B, 1024]$

### 1.4 Classification Head
- `Linear(1024, 256)`
- `LayerNorm(256)` + `GELU()`
- `Dropout(0.3)`
- `Linear(256, 4)`
- Softmax output across: `[Glioma, Meningioma, Pituitary, No Tumor]`
