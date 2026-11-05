"""
Explainable Hybrid CNN–Vision Transformer Brain Tumor Classification Dashboard.
Streamlit Web Application for Research, Evaluation, and Demonstration.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image
import streamlit as st
import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, get_device
from src.dataset import ID_TO_CLASS, CLASSES, create_mock_dataset
from models.hybrid_model import HybridBrainTumorModel
from explainability.explainer import BrainTumorExplainer, MEDICAL_DISCLAIMER

# Page configuration
st.set_page_config(
    page_title="Brain Tumor AI | Explainable CNN-ViT",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .disclaimer-box {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 0.9rem;
        border-radius: 4px;
        color: #991b1b;
        font-size: 0.85rem;
        margin-top: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_app_resources(config_path: str = "configs/config.yaml") -> Tuple[Optional[torch.nn.Module], Optional[BrainTumorExplainer], dict, torch.device]:
    """Loads configuration, device, trained model, and explainability engine."""
    config = load_config(str(PROJECT_ROOT / config_path))
    device = get_device("auto")

    checkpoint_path = PROJECT_ROOT / config["training"]["checkpoint_dir"] / config["training"]["checkpoint_filename"]

    if not checkpoint_path.is_file():
        return None, None, config, device

    try:
        eval_config = dict(config)
        eval_config["model"] = dict(config.get("model", {}))
        eval_config["model"]["cnn_pretrained"] = False
        eval_config["model"]["vit_pretrained"] = False
        model = HybridBrainTumorModel.from_config(eval_config)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        explainer = BrainTumorExplainer(model=model, device=device, image_size=config["dataset"]["image_size"])
        return model, explainer, config, device
    except Exception as e:
        st.error(f"Error loading model checkpoint: {e}")
        return None, None, config, device


def render_sidebar():
    """Renders sidebar navigation and system information."""
    st.sidebar.image("https://img.icons8.com/fluency/96/brain.png", width=70)
    st.sidebar.title("Brain Tumor AI")
    st.sidebar.caption("Explainable Hybrid CNN–ViT Framework")

    st.sidebar.markdown("---")
    menu = st.sidebar.radio(
        "Navigation",
        [
            "🔬 Live MRI Inference & XAI",
            "📊 Evaluation & Metrics",
            "🧩 Architecture & Method",
            "📁 Dataset Information",
            "⚠️ Ethics & Medical Disclaimer"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("System Status")
    device_name = "Apple Silicon (MPS)" if torch.backends.mps.is_available() else ("NVIDIA CUDA" if torch.cuda.is_available() else "CPU")
    st.sidebar.info(f"**Hardware Acceleration:**\n{device_name}")
    st.sidebar.caption("PyTorch 2.15 • Torchvision 0.29")

    return menu


def page_live_inference(model, explainer, config, device):
    """Page 1: Live MRI Upload, Inference, and Multi-Scale Explainability."""
    st.markdown('<div class="main-title">Brain Tumor Classification & Multimodal Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload a Cranial MRI Scan (Axial/Coronal/Sagittal) to predict tumor class and visualize CNN Grad-CAM & ViT Self-Attention heatmaps.</div>', unsafe_allow_html=True)

    if model is None or explainer is None:
        st.warning(
            "⚠️ **Trained Model Weights Not Found**\n\n"
            "No checkpoint was detected at `checkpoints/best_hybrid_model.pth`.\n"
            "To view genuine predictions and heatmaps, please train the model using:\n\n"
            "```bash\npython training/train.py --epochs 10 --batch-size 4\n```\n"
            "*Note: This research prototype adheres to scientific integrity and does not display fake or hardcoded classifications.*"
        )

    col_upload, col_sample = st.columns([2, 1])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Choose a Brain MRI Scan",
            type=["jpg", "jpeg", "png"],
            help="Upload an axial, coronal, or sagittal T1/T2-weighted brain MRI image."
        )

    with col_sample:
        st.markdown("**Or load a sample MRI:**")
        sample_choice = st.selectbox("Select Sample Class", ["-- Choose Sample --", "Glioma", "Meningioma", "Pituitary", "No Tumor"])

    image_to_process = None
    if uploaded_file is not None:
        try:
            image_to_process = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Could not read uploaded image: {e}")
    elif sample_choice != "-- Choose Sample --":
        # Look for sample in mock or raw dataset
        sample_folder = sample_choice.lower().replace(" tumor", "").replace(" ", "")
        sample_paths = list((PROJECT_ROOT / "data").rglob(f"*{sample_folder}*.*"))
        valid_sample_paths = [p for p in sample_paths if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
        if valid_sample_paths:
            image_to_process = Image.open(valid_sample_paths[0]).convert("RGB")
            st.success(f"Loaded sample: {valid_sample_paths[0].name}")
        else:
            # Generate sample on the fly
            mock_dir = PROJECT_ROOT / "data" / "raw"
            create_mock_dataset(str(mock_dir), samples_per_class=2)
            gen_paths = list((mock_dir / sample_folder).glob("*.png")) if (mock_dir / sample_folder).exists() else []
            if gen_paths:
                image_to_process = Image.open(gen_paths[0]).convert("RGB")
                st.success(f"Generated sample: {gen_paths[0].name}")

    if image_to_process is not None and model is not None and explainer is not None:
        st.markdown("---")
        with st.spinner("Analyzing MRI with Hybrid CNN-ViT Model and extracting visual explanations..."):
            result = explainer.explain(image_to_process)

        pred_name = result["predicted_class_name"]
        conf = result["confidence"] * 100.0

        # High-level outcome banner
        badge_color = "#2563eb" if pred_name != "No Tumor" else "#059669"
        st.markdown(
            f"""
            <div style="background-color: #f1f5f9; border-radius: 8px; padding: 1.2rem; border-left: 6px solid {badge_color}; margin-bottom: 1.5rem;">
                <h3 style="margin: 0; color: #1e293b;">Predicted Diagnostic Class: <strong>{pred_name}</strong></h3>
                <p style="margin: 0.3rem 0 0 0; color: #475569; font-size: 1.05rem;">
                    Confidence Score: <strong>{conf:.2f}%</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Tabbed details: Saliency Maps vs Class Probabilities
        tab1, tab2 = st.tabs(["🔍 Multimodal Saliency Overlays", "📊 Probability Distribution"])

        with tab1:
            st.subheader("Side-by-Side Saliency Comparison")
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.image(result["original_image"], caption="1. Input MRI Scan", use_container_width=True)
                st.caption("Standardized 224x224 RGB anatomical scan.")

            with c2:
                st.image(result["cnn_gradcam_overlay"], caption="2. CNN Grad-CAM (ResNet)", use_container_width=True)
                st.caption("Local high-frequency spatial lesion localization.")

            with c3:
                st.image(result["vit_attention_overlay"], caption="3. ViT Attention (Transformer)", use_container_width=True)
                st.caption("Global anatomical contextual attention from [CLS] token.")

            with c4:
                st.image(result["hybrid_fused_overlay"], caption="4. Hybrid Multi-Scale Saliency", use_container_width=True)
                st.caption("Unified representation combining local and global features.")

            st.info(
                f"**Clinical Feature Attribution Summary:**\n"
                f"- For **{pred_name}**, the ResNet CNN branch focuses on sharp morphological boundaries and localized pixel intensity variations.\n"
                f"- The Vision Transformer branch computes self-attention across 196 non-overlapping 16x16 anatomical patches to assess structural symmetry and hemisphere balance."
            )

        with tab2:
            st.subheader("Classification Probabilities Across All 4 Diagnostic Classes")
            probs = result["class_probabilities"]
            for cls_name, prob in probs.items():
                col_name, col_bar, col_pct = st.columns([2, 5, 1])
                with col_name:
                    st.write(f"**{cls_name}**")
                with col_bar:
                    st.progress(float(prob))
                with col_pct:
                    st.write(f"{prob * 100:.2f}%")

        st.markdown(f'<div class="disclaimer-box">{result["medical_disclaimer"]}</div>', unsafe_allow_html=True)


def page_evaluation_dashboard(config):
    """Page 2: Verifiable Metrics, Confusion Matrix, and ROC Curves."""
    st.markdown('<div class="main-title">Model Performance & Evaluation Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Quantitative results evaluated on the held-out test split. All metrics reflect actual computed test results.</div>', unsafe_allow_html=True)

    metrics_file = PROJECT_ROOT / config["evaluation"].get("metrics_dir", "results/metrics") / "test_metrics.json"

    if not metrics_file.is_file():
        st.warning(
            "⚠️ **Evaluation Metrics Not Found**\n\n"
            "The model has not yet been evaluated on the test set. Run:\n\n"
            "```bash\npython evaluation/evaluate.py\n```\n"
            "to execute the evaluation suite and generate real metrics and publication figures."
        )
        return

    from src.utils import load_json
    metrics = load_json(str(metrics_file))

    # Top KPI cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Test Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    with kpi2:
        st.metric("Macro Precision", f"{metrics['macro_precision']:.4f}")
    with kpi3:
        st.metric("Macro Recall", f"{metrics['macro_recall']:.4f}")
    with kpi4:
        st.metric("Macro F1-Score", f"{metrics['macro_f1']:.4f}")

    st.markdown("---")
    st.subheader("Evaluation Visualizations")

    col_fig1, col_fig2 = st.columns(2)
    figures_dir = PROJECT_ROOT / config["evaluation"].get("figures_dir", "results/figures")

    cm_img = figures_dir / "confusion_matrix.png"
    if cm_img.is_file():
        with col_fig1:
            st.image(str(cm_img), caption="Test Confusion Matrix (Actual counts)", use_container_width=True)

    roc_img = figures_dir / "roc_curves.png"
    if roc_img.is_file():
        with col_fig2:
            st.image(str(roc_img), caption="One-vs-Rest ROC Curves", use_container_width=True)

    per_class_img = figures_dir / "per_class_metrics.png"
    if per_class_img.is_file():
        st.image(str(per_class_img), caption="Per-Class Precision, Recall, and F1-Scores", use_container_width=True)

    # Detailed Per-Class Table
    st.subheader("Detailed Classification Report")
    report_data = []
    for cls in [ID_TO_CLASS[i] for i in range(len(CLASSES))]:
        if cls in metrics.get("classification_report", {}):
            c_data = metrics["classification_report"][cls]
            report_data.append({
                "Tumor Category": cls,
                "Precision": f"{c_data['precision']:.4f}",
                "Recall": f"{c_data['recall']:.4f}",
                "F1-Score": f"{c_data['f1-score']:.4f}",
                "Support (Samples)": int(c_data['support'])
            })
    if report_data:
        st.table(report_data)


def page_architecture():
    """Page 3: Model Architecture and Methodology Deep-Dive."""
    st.markdown('<div class="main-title">Explainable Hybrid CNN–ViT Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Theoretical foundations and dual-branch design for brain MRI multi-class classification.</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Architectural Design Overview
    The framework addresses a fundamental limitation in medical neuroimaging analysis:
    - **Convolutional Neural Networks (CNNs)** excel at extracting fine, localized spatial features (lesion borders, microcalcifications, textural heterogeneities) due to their translation invariance and inductive bias, but have a constrained receptive field.
    - **Vision Transformers (ViTs)** utilize multi-head self-attention mechanisms to model global dependencies and long-range structural relationships (cross-hemisphere symmetry, ventricular alignment), but require substantial context to learn fine-grained local textures.
    - **Hybrid Fusion Framework**: By projecting both feature spaces into a shared representation and applying **Bidirectional Multi-Head Cross-Attention Fusion**, our model achieves superior representation capability.
    """)

    st.markdown("""
    ```mermaid
    flowchart LR
        Input["Brain MRI Scan (224x224x3)"]
        
        subgraph CNN_Branch ["Local Branch: ResNet"]
            C1["Residual Blocks (conv1 to layer4)"]
            GAP["Global Average Pooling"]
            F_CNN["Local Feature Vector (D=512)"]
            GradCAM["Grad-CAM Saliency Map"]
            C1 --> GAP --> F_CNN
            C1 -.-> GradCAM
        end
        
        subgraph ViT_Branch ["Global Branch: Vision Transformer"]
            P["16x16 Patch Projection"]
            CLS["[CLS] Token + Positional Embeddings"]
            ENC["12 Transformer Encoder Blocks"]
            F_ViT["Global Feature Vector (D=768)"]
            AttnMap["Self-Attention Rollout Map"]
            P --> CLS --> ENC --> F_ViT
            ENC -.-> AttnMap
        end
        
        subgraph Fusion ["Cross-Attention Fusion & Head"]
            Proj["Linear Projections to Shared D=512"]
            MHA["Multi-Head Cross-Attention Block"]
            FFN["LayerNorm + FFN + Dropout"]
            Head["Classification Head (MLP)"]
            Out["4 Diagnostic Classes"]
            Proj --> MHA --> FFN --> Head --> Out
        end
        
        Input --> C1
        Input --> P
        F_CNN --> Proj
        F_ViT --> Proj
    ```
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 1. CNN Branch (ResNet)")
        st.markdown("- **Backbone:** ResNet18 / ResNet50\n- **Receptive Field:** Hierarchical local patches\n- **Target Layer for XAI:** `layer4`\n- **Feature Dimension:** 512 / 2048")
    with col2:
        st.markdown("#### 2. ViT Branch (Transformer)")
        st.markdown("- **Backbone:** ViT-B/16\n- **Patch Size:** 16x16 ($14 \\times 14 = 196$ tokens)\n- **Self-Attention Heads:** 12 heads\n- **Embedding Dimension:** 768")
    with col3:
        st.markdown("#### 3. Cross-Attention Fusion")
        st.markdown("- **Shared Latent Dimension:** 512\n- **Fusion Mechanism:** Multi-Head Cross Attention\n- **Dropout:** 0.30\n- **Output Classes:** 4")


def page_dataset_info(config):
    """Page 4: Dataset Specifications and Download Instructions."""
    st.markdown('<div class="main-title">Dataset Specification & Setup</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Multi-class Brain Tumor MRI benchmark dataset details, licensing, and acquisition.</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Primary Benchmark Dataset: Kaggle Brain Tumor Classification (MRI)
    - **Classes:** 4 categories (`glioma`, `meningioma`, `pituitary`, `notumor`)
    - **Total Images:** ~7,023 real MRI scans (axial, coronal, sagittal views)
    - **Modality:** T1-weighted contrast-enhanced, T2-weighted, and FLAIR MRI sequences
    - **Licensing:** Creative Commons Attribution 4.0 International (CC BY 4.0)
    - **Source Reference:** *Sartaj Bhuvaji, Ankita Kadam, Prajakta Bhumkar, Sameer Dedge, "Brain Tumor Classification (MRI)", Kaggle.*

    ### Automated Dataset Preparation & Stratified Splitting
    Our pipeline enforces **leakage-safe stratified splitting**:
    - **Training Set (70%):** Utilized exclusively for parameter updates with spatial augmentations (rotations, flips, affine transformations).
    - **Validation Set (15%):** Used strictly for hyperparameter tuning and early stopping.
    - **Held-Out Test Set (15%):** Never exposed during training; used purely for final verifiable evaluation.
    """)

    st.code("""
# Dataset Setup Commands:
# 1. Download dataset via Kaggle CLI:
kaggle datasets download -d sartajbhuvaji/brain-tumor-classification-mri -p data/raw --unzip

# 2. Or place MRI image folders in:
#    data/raw/glioma/
#    data/raw/meningioma/
#    data/raw/pituitary/
#    data/raw/notumor/

# 3. Run automated preprocessing and stratified split generation:
python training/train.py --config configs/config.yaml
    """, language="bash")


def page_ethics():
    """Page 5: Ethical Guidelines and Medical Prototype Disclaimers."""
    st.markdown('<div class="main-title">Ethical Considerations & Clinical Disclaimer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Academic transparency, limitations of computational saliency, and clinical guidelines.</div>', unsafe_allow_html=True)

    st.warning(MEDICAL_DISCLAIMER)

    st.markdown("""
    ### Technical & Clinical Limitations
    1. **Not a Medical Device:** This software system is a final-year B.Tech CSE (AIML) engineering project prototype developed solely for educational and research exploration of hybrid deep learning architectures. It is neither FDA-cleared nor CE-marked as Software as a Medical Device (SaMD).
    2. **Interpretability is Not Ground Truth:** Grad-CAM and ViT attention maps visualize the mathematical gradients and self-attention weights of artificial neural networks. They highlight correlated pixel regions rather than histopathologically confirmed lesion margins.
    3. **Domain Shift & Scanner Variability:** The dataset comprises specific MRI magnet strengths (1.5T / 3.0T) and scanning protocols. Out-of-distribution scans from different hospital scanners or acquisition parameters may experience domain shift.
    4. **Human-in-the-Loop:** Clinical diagnosis of primary central nervous system (CNS) neoplasms requires multi-parametric MRI examination, patient history, histological confirmation, and molecular testing by qualified neuro-radiologists and oncologists.
    """)


def main():
    menu = render_sidebar()
    model, explainer, config, device = load_app_resources()

    if menu == "🔬 Live MRI Inference & XAI":
        page_live_inference(model, explainer, config, device)
    elif menu == "📊 Evaluation & Metrics":
        page_evaluation_dashboard(config)
    elif menu == "🧩 Architecture & Method":
        page_architecture()
    elif menu == "📁 Dataset Information":
        page_dataset_info(config)
    elif menu == "⚠️ Ethics & Medical Disclaimer":
        page_ethics()


if __name__ == "__main__":
    main()
