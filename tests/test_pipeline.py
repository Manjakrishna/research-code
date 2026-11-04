"""
Integration test for end-to-end training and evaluation pipeline.
"""

import os
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import (
    BrainTumorDataset,
    create_mock_dataset,
    scan_dataset_directory,
    create_stratified_splits,
    get_transforms
)
from models.hybrid_model import HybridBrainTumorModel
from training.trainer import BrainTumorTrainer
from evaluation.evaluate import evaluate_model


def test_pipeline_smoke(tmp_path):
    """Executes a complete mini train-eval pipeline end-to-end on synthetic data."""
    # 1. Create mini mock dataset
    data_dir = tmp_path / "mock_data"
    create_mock_dataset(str(data_dir), samples_per_class=6)
    samples = scan_dataset_directory(str(data_dir))
    assert len(samples) == 24

    train_samples, val_samples, test_samples = create_stratified_splits(
        samples, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=42
    )

    transform = get_transforms(image_size=224, is_training=False)
    train_ds = BrainTumorDataset(train_samples, transform=transform)
    val_ds = BrainTumorDataset(val_samples, transform=transform)
    test_ds = BrainTumorDataset(test_samples, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False)

    # 2. Build Model
    model = HybridBrainTumorModel(
        cnn_backbone="resnet18",
        cnn_pretrained=False,
        vit_backbone="vit_b_16",
        vit_pretrained=False,
        projection_dim=128,
        num_classes=4
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    checkpoint_dir = tmp_path / "checkpoints"

    trainer = BrainTumorTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device("cpu"),
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_name="test_model.pth",
        early_stopping_patience=2
    )

    # 3. Fit 1 epoch
    history = trainer.fit(num_epochs=1)
    assert len(history["train_loss"]) == 1
    assert (checkpoint_dir / "test_model.pth").is_file()

    # 4. Evaluate
    metrics_dir = tmp_path / "metrics"
    figures_dir = tmp_path / "figures"
    eval_result = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=torch.device("cpu"),
        output_dir_metrics=str(metrics_dir),
        output_dir_figures=str(figures_dir)
    )

    assert "accuracy" in eval_result
    assert "macro_f1" in eval_result
    assert (metrics_dir / "test_metrics.json").is_file()
    assert (figures_dir / "confusion_matrix.png").is_file()
