# Final-Year B.Tech CSE (AIML) Viva & College Demo Guide

This document provides a structured walkthrough for demonstrating this project during final-year project reviews, internal evaluations, and external examiner vivas.

---

## 🎯 Demonstration Step-by-Step

### Step 1: Environment Activation & Verification
Open Terminal in the project directory:
```bash
source .venv/bin/activate
python -c "import torch; print('PyTorch Version:', torch.__version__, '| MPS Available:', torch.backends.mps.is_available())"
```
*Key Viva Point to Mention:*
> "Our architecture is optimized for modern hardware, automatically taking advantage of Apple Silicon's unified memory and Metal Performance Shaders (MPS) as well as NVIDIA CUDA, with automated CPU fallback."

---

### Step 2: Automated Test Suite Execution
Demonstrate engineering rigor and testing:
```bash
pytest tests/ -v
```
*Examiner Explanation:*
> "We have written 13 comprehensive automated unit and integration tests covering data transforms, dataset integrity verification, CNN and ViT tensor dimensions, multi-head attention weight extraction, Grad-CAM saliency generation, and end-to-end pipeline training."

---

### Step 3: Launching the Medical Research Dashboard
Launch Streamlit:
```bash
streamlit run app/main.py
```
A browser window opens at `http://localhost:8501`.

---

### Step 4: Web Application Demonstration Walkthrough

1. **Live MRI Inference & Multimodal XAI:**
   - In the sidebar or main screen, choose a sample MRI (or upload a custom MRI scan).
   - Show the **instant prediction** with confidence score.
   - Show the **4-Panel Side-by-Side Saliency Comparison**:
     - Panel 1: Original MRI Scan (224x224 RGB).
     - Panel 2: CNN Grad-CAM (ResNet `layer4` highlighting local high-frequency lesion margins).
     - Panel 3: ViT Self-Attention (Transformer Multi-Head Attention mapping showing global anatomical context and bilateral hemisphere comparison).
     - Panel 4: Hybrid Multi-Scale Saliency Fusion.
   - Show the **Class Probabilities Bar Chart** illustrating the softmax distribution across all 4 classes.

2. **Performance & Evaluation Dashboard:**
   - Click on `📊 Evaluation & Metrics` in the sidebar.
   - Highlight the **actual test set metrics**:
     - Multi-Class Accuracy
     - Macro and Weighted Precision, Recall, and F1-Scores
     - One-vs-Rest ROC-AUC
   - Display the **Test Confusion Matrix** and **ROC Curves**.

3. **Architecture & Method Deep Dive:**
   - Click on `🧩 Architecture & Method`.
   - Explain the interactive Mermaid architectural flowchart detailing the dual-branch CNN + ViT pipeline and Cross-Attention Fusion.

4. **Ethics & Clinical Disclaimer:**
   - Point out the prominent medical research disclaimer emphasizing that this is a research prototype for computer science education and not an uncertified medical diagnostic device.

---

## 💡 Anticipated Examiner Questions & Model Answers

### Q1: Why combine a CNN with a Vision Transformer instead of using just one?
> **Answer:** "CNNs possess strong local inductive biases (spatial locality and translation invariance), making them exceptional at detecting sharp boundaries, edges, and textures of lesions. However, CNNs have localized receptive fields and struggle to capture global anatomical context. Vision Transformers, through multi-head self-attention, excel at modeling long-range dependencies and global bilateral brain symmetry, but lack inductive spatial priors. Combining both through cross-attention fusion gives us the best of both worlds: local lesion precision plus global contextual awareness."

### Q2: How does your cross-attention fusion work mathematically?
> **Answer:** "We project both the CNN feature vector ($D=512$) and the ViT $[CLS]$ feature vector ($D=768$) into a shared latent dimension of $512$ using linear layers and LayerNorm. We stack these two vectors into a sequence of length 2 and apply Multi-Head Attention where the local CNN token and global ViT token attend to one another. A residual connection and Feedforward Network (FFN) refine the representation before passing it to our 4-class classification head."

### Q3: How do you prevent data leakage between training and testing?
> **Answer:** "We implemented strict stratified splitting before training. First, the held-out test set (15%) is isolated completely. Then, the remaining data is split into 70% train and 15% validation. Data augmentations (rotations, flips, affine transforms) are applied strictly to the training split only. Validation and testing utilize deterministic resizing and normalization. We also use SHA-256 cryptographic hashing to detect and remove any duplicate scans."

### Q4: What is the clinical difference between the 4 classes?
> **Answer:**
> - **Glioma:** Infiltrative primary brain tumor arising from glial cells (astrocytes/oligodendrocytes), typically intra-axial with irregular, invasive margins.
> - **Meningioma:** Extra-axial tumor arising from the arachnoid layer of the meninges, usually well-demarcated and attached to the dura mater.
> - **Pituitary Tumor:** Sellar or suprasellar adenoma located at the base of the brain near the optic chiasm.
> - **No Tumor:** Normal brain scan showing symmetrical cerebral hemispheres, normal ventricles, and preserved gray-white matter differentiation.
