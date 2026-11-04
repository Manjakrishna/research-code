# An Explainable Hybrid CNN–Vision Transformer Framework for Multi-Class Brain Tumor Classification

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.15](https://img.shields.io/badge/PyTorch-2.15-EE4C2C.svg)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/Hardware-Apple%20MPS%20%7C%20CUDA%20%7C%20CPU-success.svg)](https://developer.apple.com/metal/pytorch/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An advanced deep learning and computer vision framework for automated, multi-class brain tumor classification from cranial Magnetic Resonance Imaging (MRI) scans, equipped with dual-scale Explainable AI (XAI) feature attribution.

---

## 📌 Executive Summary & Motivation

Brain tumors represent among the most lethal and diagnostically challenging human neoplasms. Accurate classification into specific histological categories (**Glioma**, **Meningioma**, and **Pituitary Tumor**) versus **Healthy Parenchyma (No Tumor)** from neuroimaging is critical for surgical planning, radiotherapy delineation, and prognosis.

Standard Deep Learning approaches present a fundamental trade-off:
1. **Convolutional Neural Networks (CNNs)** (e.g., ResNet) possess strong local inductive biases, effectively capturing sharp lesion boundaries and high-frequency textural variations, but struggle with global anatomical context due to localized receptive fields.
2. **Vision Transformers (ViTs)** utilize multi-head self-attention to capture long-range structural dependencies and bilateral brain symmetry, but often miss localized micro-architectural cues.
3. **The Black-Box Dilemma:** Traditional deep models lack interpretability, severely hindering clinical acceptance.

### Proposed Solution
This framework integrates:
- **Local Spatial Branch:** ResNet backbone extracting fine-grained convolutional features ($D=512$).
- **Global Contextual Branch:** Pretrained Vision Transformer (ViT-B/16) modeling long-range dependencies ($D=768$).
- **Cross-Attention Fusion Module:** Shared latent projection ($D=512$) with bidirectional Multi-Head Cross-Attention.
- **Multimodal Explainable AI:** Dual saliency mapping combining **CNN Grad-CAM** (local activation) and **ViT Self-Attention Rollout** (global attention).

---

## 🗂️ Project Directory Structure

```text
xebia project/
├── configs/
│   └── config.yaml               # Central hyperparameter & pipeline configuration
├── data/
│   ├── raw/                      # Raw MRI images organized in 4 class subfolders
│   └── processed/                # Leakage-safe train, val, and test split manifests (.csv)
├── models/
│   ├── __init__.py
│   ├── cnn_branch.py             # ResNet18/ResNet50 feature extractor & hook provider
│   ├── vit_branch.py             # ViT-B/16 backbone & self-attention weight extractor
│   ├── fusion.py                 # Cross-Attention, Gated, & Concatenation fusion blocks
│   └── hybrid_model.py           # Unified dual-branch HybridBrainTumorModel
├── explainability/
│   ├── __init__.py
│   ├── gradcam.py                # ResNet Grad-CAM implementation & heatmap overlays
│   ├── vit_attention.py          # Vision Transformer [CLS] attention spatial mapper
│   └── explainer.py              # Unified BrainTumorExplainer orchestrator
├── training/
│   ├── __init__.py
│   ├── trainer.py                # PyTorch training loop, early stopping, LR schedulers
│   └── train.py                  # CLI training entry point (full training & smoke test)
├── evaluation/
│   ├── __init__.py
│   └── evaluate.py               # Test set evaluation: Confusion Matrix, ROC-AUC, F1
├── app/
│   └── main.py                   # Streamlit medical research dashboard
├── tests/
│   ├── __init__.py
│   ├── test_dataset.py           # Data loading, transforms, validation tests
│   ├── test_models.py            # CNN, ViT, Fusion, Hybrid dimensions tests
│   ├── test_explainability.py    # Grad-CAM and ViT attention visualization tests
│   └── test_pipeline.py          # End-to-end training & evaluation integration test
├── results/
│   ├── figures/                  # Training curves, confusion matrices, ROC curves
│   └── metrics/                  # Test set metrics and training history JSONs
├── checkpoints/                  # Trained PyTorch model checkpoints (.pth)
├── docs/                         # Additional diagrams, reports, and references
├── PROJECT_REPORT.md             # Academic Project Report
├── requirements.txt              # Pinned Python dependencies
├── .gitignore                    # Git exclusions
└── README.md                     # Project documentation (this file)
```

---

## ⚙️ Environment Setup & Installation

### Prerequisites
- macOS (Apple Silicon M1/M2/M3) or Linux / Windows with NVIDIA GPU.
- Python 3.11 installed.

### 1. Clone & Navigate to Repository
```bash
git clone <repository_url>
cd "xebia project"
```

### 2. Create and Activate Virtual Environment
```bash
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
```
*(On Windows: `.venv\Scripts\activate`)*

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📦 Dataset Preparation

The project targets the standard benchmark **Brain Tumor Classification (MRI)** dataset comprising 4 classes:
1. `glioma`
2. `meningioma`
3. `pituitary`
4. `notumor`

### Option A: Download via Kaggle CLI
```bash
kaggle datasets download -d sartajbhuvaji/brain-tumor-classification-mri -p data/raw --unzip
```

### Option B: Manual Directory Organization
Place images directly into `data/raw`:
```text
data/raw/
  ├── glioma/
  ├── meningioma/
  ├── pituitary/
  └── notumor/
```

### Option C: Automated Synthetic Fallback
If you do not immediately have the Kaggle dataset downloaded, our system automatically generates a synthetic dataset with valid brain contours, ventricular morphology, and class lesions for immediate execution:
```bash
python training/train.py --smoke-test
```

---

## 🚀 Execution Guide

### 1. Model Training
To train the Hybrid CNN–ViT model:
```bash
python training/train.py --config configs/config.yaml --epochs 15 --batch-size 4
```
**Key arguments:**
- `--smoke-test`: Rapid 2-epoch pipeline verification on synthetic data.
- `--epochs N`: Override epoch count (default: 15).
- `--batch-size B`: Override batch size (default: 4, memory-safe for M1).
- `--device auto`: Automatically uses `mps` (Apple Silicon), `cuda`, or `cpu`.

Checkpoints are saved to `checkpoints/best_hybrid_model.pth`.

### 2. Held-Out Test Evaluation
To compute verifiable metrics on the test split:
```bash
python evaluation/evaluate.py --config configs/config.yaml
```
Output results:
- Metrics summary: `results/metrics/test_metrics.json`
- Confusion matrix: `results/figures/confusion_matrix.png`
- ROC curves: `results/figures/roc_curves.png`
- Per-class metrics: `results/figures/per_class_metrics.png`

### 3. Launching the Streamlit Web Application
To start the medical research and inference dashboard:
```bash
streamlit run app/main.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Automated Testing

Run the full pytest suite:
```bash
pytest tests/ -v
```
All tests verify:
- Image loading, resizing (224x224), RGB conversion, and tensor transforms
- Dataset scanning, integrity checking, and duplicate detection
- Leakage-free stratified train/val/test splitting
- ResNet CNN feature extractor output dimensions (`[B, 512]`)
- Vision Transformer ViT-B/16 output dimensions (`[B, 768]`)
- ViT Multi-Head Self-Attention matrix extraction (`[B, 12, 197, 197]`)
- Cross-Attention and Gated Fusion blocks
- End-to-end forward pass (`[B, 4]`)
- Grad-CAM heatmap extraction and colormap overlay blending
- End-to-end training and evaluation pipeline

---

## ⚕️ Medical Research Disclaimer
> **IMPORTANT:** This project is an academic research prototype developed as a final-year B.Tech CSE (AIML) project. The predictions, confidence scores, and visual saliency maps are computational interpretations generated by artificial neural networks and **MUST NOT** be used for clinical decision-making, disease diagnosis, or treatment planning. Always consult a licensed radiologist or medical oncologist.
