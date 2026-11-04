"""
Brain Tumor MRI Dataset Module.
Handles image loading, preprocessing, augmentation, stratified splitting,
image integrity verification, and mock dataset generation.
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable, Any
import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

logger = logging.getLogger("brain_tumor_project.dataset")

# Standard 4-Class Mapping
CLASSES: List[str] = ["glioma", "meningioma", "pituitary", "notumor"]
CLASS_TO_ID: Dict[str, int] = {cls: idx for idx, cls in enumerate(CLASSES)}
ID_TO_CLASS: Dict[int, str] = {
    0: "Glioma",
    1: "Meningioma",
    2: "Pituitary Tumor",
    3: "No Tumor"
}

# Standard normalization (ImageNet statistics for pretrained backbones)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(
    image_size: int = 224,
    is_training: bool = False,
    mean: List[float] = IMAGENET_MEAN,
    std: List[float] = IMAGENET_STD
) -> transforms.Compose:
    """
    Returns image transformation pipeline.
    Training uses robust spatial and color augmentations.
    Validation/Testing uses deterministic resizing and normalization.
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset for Multi-Class Brain Tumor MRI scans.
    Loads images from a list of paths or a DataFrame.
    """
    def __init__(
        self,
        samples: List[Tuple[str, int]],
        transform: Optional[Callable] = None
    ):
        """
        Args:
            samples: List of (image_path, class_id) tuples.
            transform: Optional torchvision transform pipeline.
        """
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_path, label = self.samples[idx]
        try:
            with Image.open(img_path) as img:
                # Convert to RGB to ensure 3 channels regardless of source format
                img = img.convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image at {img_path}: {e}")
            raise RuntimeError(f"Corrupted or invalid image: {img_path}") from e

        if self.transform is not None:
            tensor_img = self.transform(img)
        else:
            tensor_img = transforms.ToTensor()(img)

        return tensor_img, label, str(img_path)


def scan_dataset_directory(data_dir: str) -> List[Tuple[str, int]]:
    """
    Scans a folder containing class subdirectories:
    data_dir/
      glioma/
      meningioma/
      pituitary/
      notumor/
    Case-insensitive folder matching.
    """
    path = Path(data_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    samples: List[Tuple[str, int]] = []
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    # Map subdirectories to canonical class IDs
    found_classes = set()
    for subfolder in path.iterdir():
        if subfolder.is_dir():
            folder_clean = subfolder.name.lower().replace(" ", "").replace("_", "").replace("-", "")
            matched_cls = None
            for cls in CLASSES:
                if cls in folder_clean or folder_clean in cls:
                    matched_cls = cls
                    break
            
            if matched_cls is not None:
                class_id = CLASS_TO_ID[matched_cls]
                found_classes.add(matched_cls)
                for file_path in subfolder.rglob("*"):
                    if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                        samples.append((str(file_path), class_id))

    logger.info(f"Scanned {len(samples)} valid images across classes {list(found_classes)} from {data_dir}")
    return samples


def validate_and_clean_dataset(data_dir: str) -> Dict[str, Any]:
    """
    Examines images in data_dir:
    - Verifies file integrity (cannot open -> corrupt)
    - Checks duplicate images via SHA-256 hash
    - Reports counts per class
    """
    path = Path(data_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    total_scanned = 0
    corrupt_files = []
    duplicate_files = []
    hash_seen = {}
    class_counts: Dict[str, int] = {cls: 0 for cls in CLASSES}
    valid_samples = []

    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    for file_path in path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            total_scanned += 1
            # 1. Integrity check
            try:
                with Image.open(file_path) as img:
                    img.verify()
                # Reopen to test full pixel decode
                with Image.open(file_path) as img:
                    img.convert("RGB").load()
            except Exception as e:
                corrupt_files.append((str(file_path), str(e)))
                continue

            # 2. Duplicate check using SHA-256
            hasher = hashlib.sha256()
            with open(file_path, "rb") as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            file_hash = hasher.hexdigest()

            if file_hash in hash_seen:
                duplicate_files.append((str(file_path), hash_seen[file_hash]))
            else:
                hash_seen[file_hash] = str(file_path)
                
                # Match class
                matched_cls = None
                for cls in CLASSES:
                    if cls in str(file_path).lower():
                        matched_cls = cls
                        break
                
                if matched_cls:
                    class_counts[matched_cls] += 1
                    valid_samples.append((str(file_path), CLASS_TO_ID[matched_cls]))

    summary = {
        "total_scanned": total_scanned,
        "valid_images": len(valid_samples),
        "corrupt_count": len(corrupt_files),
        "corrupt_files": corrupt_files[:10],  # sample
        "duplicate_count": len(duplicate_files),
        "class_distribution": class_counts
    }
    logger.info(f"Dataset Quality Summary: {summary['valid_images']} valid, {summary['corrupt_count']} corrupt, {summary['duplicate_count']} duplicates.")
    return summary


def create_stratified_splits(
    samples: List[Tuple[str, int]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Splits samples into train, val, and test subsets using leakage-free stratified splitting.
    Never leaks test images into training or validation sets.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Split ratios must sum to 1.0"
    
    paths = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    # Check if stratified split is feasible based on class counts
    class_counts = {}
    for l in labels:
        class_counts[l] = class_counts.get(l, 0) + 1

    can_stratify = all(c >= 2 for c in class_counts.values()) and (len(labels) * test_ratio >= len(class_counts))

    if can_stratify:
        # Split off test set first
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            paths, labels,
            test_size=test_ratio,
            stratify=labels,
            random_state=seed
        )

        # Split train and validation
        relative_val_ratio = val_ratio / (train_ratio + val_ratio)
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths, train_val_labels,
            test_size=relative_val_ratio,
            stratify=train_val_labels,
            random_state=seed
        )
    else:
        logger.warning("Sample count too low for strict stratification; falling back to random split.")
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            paths, labels,
            test_size=test_ratio,
            random_state=seed
        )
        relative_val_ratio = val_ratio / (train_ratio + val_ratio)
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths, train_val_labels,
            test_size=relative_val_ratio,
            random_state=seed
        )

    train_samples = list(zip(train_paths, train_labels))
    val_samples = list(zip(val_paths, val_labels))
    test_samples = list(zip(test_paths, test_labels))

    logger.info(f"Stratified Splits: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
    return train_samples, val_samples, test_samples


def save_splits_to_csv(
    train_samples: List[Tuple[str, int]],
    val_samples: List[Tuple[str, int]],
    test_samples: List[Tuple[str, int]],
    output_dir: str = "data/processed"
) -> Dict[str, str]:
    """Saves splits to CSV files with columns ['image_path', 'label_id', 'label_name']."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    csv_paths = {}
    for name, data in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        df = pd.DataFrame(data, columns=["image_path", "label_id"])
        df["label_name"] = df["label_id"].map(ID_TO_CLASS)
        csv_file = out_path / f"{name}_split.csv"
        df.to_csv(csv_file, index=False)
        csv_paths[name] = str(csv_file)
        logger.info(f"Saved {name} split manifest to {csv_file} ({len(df)} rows)")

    return csv_paths


def load_split_from_csv(csv_path: str) -> List[Tuple[str, int]]:
    """Loads a split manifest CSV into a list of (image_path, label_id) tuples."""
    df = pd.read_csv(csv_path)
    return list(zip(df["image_path"].astype(str), df["label_id"].astype(int)))


def create_mock_dataset(target_dir: str = "data/raw", samples_per_class: int = 15) -> str:
    """
    Generates realistic synthetic 224x224 brain MRI-like images for rapid
    testing, CI/CD, and offline verification of the pipeline.
    """
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    for cls_name in CLASSES:
        cls_dir = target_path / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(samples_per_class):
            img_file = cls_dir / f"mock_{cls_name}_{i:03d}.png"
            if img_file.exists():
                continue

            # Create synthetic brain MRI: dark background, cranial skull ellipse, brain parenchyma
            img = Image.new("L", (224, 224), color=15)
            draw = ImageDraw.Draw(img)

            # Skull ellipse
            draw.ellipse([25, 20, 199, 204], fill=50, outline=160, width=3)
            # Brain tissue core
            draw.ellipse([40, 35, 184, 189], fill=95)

            # Class-specific tumor artifact with slight variation per sample
            offset = (i * 3) % 15
            if cls_name == "glioma":
                # Infiltrative hyperintensity with irregular edge
                draw.ellipse([60 + offset, 60, 115 + offset, 110], fill=210, outline=235)
                draw.ellipse([80 + offset, 80, 125 + offset, 120], fill=180)
            elif cls_name == "meningioma":
                # Well-circumscribed dural-based mass
                draw.chord([140, 50 + offset, 190, 120 + offset], start=0, end=180, fill=240, outline=255)
            elif cls_name == "pituitary":
                # Sellar / suprasellar mass at base of brain
                draw.ellipse([92 + offset, 140, 132 + offset, 175], fill=225, outline=250)
            elif cls_name == "notumor":
                # Normal brain ventricles (dark CSF spaces)
                draw.ellipse([100, 85 + offset, 110, 130 + offset], fill=30)
                draw.ellipse([114, 85 + offset, 124, 130 + offset], fill=30)

            # Convert to RGB and save
            rgb_img = img.convert("RGB")
            rgb_img.save(img_file)

    logger.info(f"Generated mock MRI dataset at {target_dir} ({samples_per_class} per class, 4 classes).")
    return str(target_path)
