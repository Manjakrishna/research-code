"""
Main Training Script for Explainable Hybrid CNN–ViT Brain Tumor Classification.
Usage:
    python training/train.py --config configs/config.yaml --epochs 15 --batch-size 4
    python training/train.py --smoke-test
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, set_seed, get_device, count_parameters
from src.dataset import (
    CLASSES,
    BrainTumorDataset,
    scan_dataset_directory,
    validate_and_clean_dataset,
    create_stratified_splits,
    save_splits_to_csv,
    load_split_from_csv,
    create_mock_dataset,
    get_transforms
)
from models.hybrid_model import HybridBrainTumorModel
from training.trainer import BrainTumorTrainer

logger = logging.getLogger("brain_tumor_project.train")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Hybrid CNN-ViT Brain Tumor Classifier")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="Override device: auto, mps, cuda, cpu")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid smoke test on synthetic data")
    parser.add_argument("--data-dir", type=str, default=None, help="Override raw data directory")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # Overrides
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.device is not None:
        config["training"]["device"] = args.device
    if args.data_dir is not None:
        config["dataset"]["raw_dir"] = args.data_dir

    # Set seed
    seed = config["training"].get("seed", 42)
    set_seed(seed)

    # Select device
    device = get_device(config["training"].get("device", "auto"))

    raw_dir = config["dataset"]["raw_dir"]
    processed_dir = config["dataset"]["processed_dir"]

    # Check dataset existence
    train_csv = Path(processed_dir) / "train_split.csv"
    val_csv = Path(processed_dir) / "val_split.csv"
    test_csv = Path(processed_dir) / "test_split.csv"

    if args.smoke_test:
        logger.info("Executing Smoke Test: creating synthetic mock MRI dataset...")
        mock_dir = "data/mock_dataset"
        create_mock_dataset(mock_dir, samples_per_class=6)
        samples = scan_dataset_directory(mock_dir)
        train_samples, val_samples, test_samples = create_stratified_splits(
            samples, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20, seed=seed
        )
        save_splits_to_csv(train_samples, val_samples, test_samples, output_dir=processed_dir)
        config["training"]["epochs"] = 2
        config["model"]["cnn_pretrained"] = False
        config["model"]["vit_pretrained"] = False
    elif train_csv.is_file() and val_csv.is_file():
        logger.info(f"Loading existing split manifests from {processed_dir}")
        train_samples = load_split_from_csv(str(train_csv))
        val_samples = load_split_from_csv(str(val_csv))
        test_samples = load_split_from_csv(str(test_csv))
    else:
        # Scan raw directory
        raw_path = Path(raw_dir)
        samples = []
        if raw_path.is_dir():
            samples = scan_dataset_directory(str(raw_path))

        if len(samples) < 8:
            logger.warning(f"No sufficient MRI images found in {raw_dir}. Generating synthetic dataset for demonstration and testing...")
            create_mock_dataset(raw_dir, samples_per_class=15)
            validate_and_clean_dataset(raw_dir)
            samples = scan_dataset_directory(raw_dir)

        # Create stratified splits
        train_samples, val_samples, test_samples = create_stratified_splits(
            samples,
            train_ratio=config["dataset"]["split_ratios"]["train"],
            val_ratio=config["dataset"]["split_ratios"]["val"],
            test_ratio=config["dataset"]["split_ratios"]["test"],
            seed=seed
        )
        save_splits_to_csv(train_samples, val_samples, test_samples, output_dir=processed_dir)

    # Image pipelines
    img_size = config["dataset"]["image_size"]
    train_transform = get_transforms(image_size=img_size, is_training=True)
    val_transform = get_transforms(image_size=img_size, is_training=False)

    train_dataset = BrainTumorDataset(train_samples, transform=train_transform)
    val_dataset = BrainTumorDataset(val_samples, transform=val_transform)

    batch_size = config["training"]["batch_size"]
    num_workers = config["training"].get("num_workers", 0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type != "cpu")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type != "cpu")
    )

    logger.info(f"Initialized DataLoaders: Train Batches={len(train_loader)}, Val Batches={len(val_loader)} (Batch Size={batch_size})")

    # Build Model
    model = HybridBrainTumorModel.from_config(config)
    total_p, train_p = count_parameters(model)
    logger.info(f"Hybrid CNN-ViT Model Architecture: Total Params={total_p:,}, Trainable Params={train_p:,}")

    # Optimizer, Loss, Scheduler
    lr = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"].get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    epochs = config["training"]["epochs"]
    scheduler_type = config["training"].get("scheduler", "cosine")
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    else:
        scheduler = None

    trainer = BrainTumorTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        config=config,
        checkpoint_dir=config["training"].get("checkpoint_dir", "checkpoints"),
        checkpoint_name=config["training"].get("checkpoint_filename", "best_hybrid_model.pth"),
        early_stopping_patience=config["training"].get("early_stopping_patience", 5)
    )

    history = trainer.fit(num_epochs=epochs)
    logger.info("Training workflow successfully finalized.")


if __name__ == "__main__":
    main()
