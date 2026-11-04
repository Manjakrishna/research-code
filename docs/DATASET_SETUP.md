# Dataset Acquisition & Setup Guide

## 1. Primary Benchmark Dataset
The primary dataset used for this project is the **Brain Tumor Classification (MRI)** dataset hosted on Kaggle:
- **Title:** Brain Tumor Classification (MRI)
- **Publisher:** Sartaj Bhuvaji, Ankita Kadam, Prajakta Bhumkar, Sameer Dedge
- **Licensing:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **URL:** https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri

### Target Classes
The dataset comprises 4 distinct categories:
1. `glioma`
2. `meningioma`
3. `pituitary`
4. `notumor`

---

## 2. Download Instructions

### Method A: Kaggle Official CLI (Recommended)
1. Ensure your Kaggle API key is configured at `~/.kaggle/kaggle.json`.
2. Run the following command from the project root:
```bash
kaggle datasets download -d sartajbhuvaji/brain-tumor-classification-mri -p data/raw --unzip
```

### Method B: Manual Web Download
1. Visit [Kaggle Dataset Page](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri).
2. Download the archive (`archive.zip`).
3. Unzip the contents into `data/raw/` so that the directory matches:
```text
data/raw/
  ├── glioma/
  ├── meningioma/
  ├── pituitary/
  └── notumor/
```

---

## 3. Automated Validation & Splitting

Once images are placed in `data/raw/`, run:
```bash
python -c "
from src.dataset import validate_and_clean_dataset
summary = validate_and_clean_dataset('data/raw')
print(summary)
"
```
This utility:
- Verifies every image file's header and decode integrity.
- Detects and flags any corrupted or zero-byte scans.
- Calculates SHA-256 cryptographic hashes to eliminate duplicate scans across splits.
- Reports total valid images and class distribution.

To generate the leakage-safe stratified train, val, and test splits:
```bash
python training/train.py --config configs/config.yaml
```
Splits will be saved to:
- `data/processed/train_split.csv`
- `data/processed/val_split.csv`
- `data/processed/test_split.csv`
