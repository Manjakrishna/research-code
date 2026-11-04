"""
Unit tests for Dataset loading, transformations, label mappings, and validation.
"""

import os
import shutil
import tempfile
from pathlib import Path
import pytest
from PIL import Image
import torch

from src.dataset import (
    CLASSES,
    CLASS_TO_ID,
    ID_TO_CLASS,
    get_transforms,
    BrainTumorDataset,
    scan_dataset_directory,
    validate_and_clean_dataset,
    create_stratified_splits,
    create_mock_dataset
)


def test_label_mappings():
    """Verify class dictionaries are bidirectional, 4 classes, and properly ordered."""
    assert len(CLASSES) == 4
    expected_classes = ["glioma", "meningioma", "pituitary", "notumor"]
    assert CLASSES == expected_classes

    expected_id_to_class = {
        0: "Glioma",
        1: "Meningioma",
        2: "Pituitary Tumor",
        3: "No Tumor"
    }
    assert ID_TO_CLASS == expected_id_to_class


def test_transforms_output_shape():
    """Verify transforms output correct shape and tensor type."""
    img = Image.new("RGB", (300, 300), color=(100, 100, 100))
    train_transform = get_transforms(image_size=224, is_training=True)
    val_transform = get_transforms(image_size=224, is_training=False)

    tensor_train = train_transform(img)
    tensor_val = val_transform(img)

    assert tensor_train.shape == (3, 224, 224)
    assert tensor_val.shape == (3, 224, 224)
    assert isinstance(tensor_train, torch.Tensor)
    assert isinstance(tensor_val, torch.Tensor)


def test_mock_dataset_and_loader(tmp_path):
    """Test generating mock dataset, scanning directory, and loading batches."""
    mock_dir = tmp_path / "mock_mri"
    create_mock_dataset(str(mock_dir), samples_per_class=4)

    samples = scan_dataset_directory(str(mock_dir))
    assert len(samples) == 16  # 4 classes * 4 samples

    # Check validation and integrity
    summary = validate_and_clean_dataset(str(mock_dir))
    assert summary["valid_images"] == 16
    assert summary["corrupt_count"] == 0

    # Dataset loader
    transform = get_transforms(image_size=224, is_training=False)
    ds = BrainTumorDataset(samples, transform=transform)
    assert len(ds) == 16

    tensor_img, label, path = ds[0]
    assert tensor_img.shape == (3, 224, 224)
    assert 0 <= label < 4
    assert os.path.exists(path)


def test_stratified_splits(tmp_path):
    """Verify stratified splitting divides samples without leakage."""
    mock_dir = tmp_path / "mock_mri"
    create_mock_dataset(str(mock_dir), samples_per_class=10)
    samples = scan_dataset_directory(str(mock_dir))
    assert len(samples) == 40

    train, val, test = create_stratified_splits(samples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)

    # Verify no overlap / no data leakage
    train_paths = set(s[0] for s in train)
    val_paths = set(s[0] for s in val)
    test_paths = set(s[0] for s in test)

    assert len(train_paths.intersection(val_paths)) == 0
    assert len(train_paths.intersection(test_paths)) == 0
    assert len(val_paths.intersection(test_paths)) == 0
    assert len(train) + len(val) + len(test) == 40


def test_invalid_image_handling(tmp_path):
    """Verify that corrupt or invalid files raise clear errors."""
    corrupt_file = tmp_path / "corrupt.png"
    corrupt_file.write_bytes(b"NOT_A_VALID_IMAGE_BYTES_123456")

    ds = BrainTumorDataset([(str(corrupt_file), 0)])
    with pytest.raises(RuntimeError) as exc_info:
        _ = ds[0]
    assert "Corrupted or invalid image" in str(exc_info.value)
