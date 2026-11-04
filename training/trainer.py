"""
Deep Learning Trainer for Hybrid Brain Tumor Classification.
Features:
- AdamW optimizer & Cosine/Plateau schedulers
- Early stopping & Best checkpoint saving
- Memory management for Apple Silicon MPS & CUDA
- Training/Validation loss & accuracy graphing
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import save_json

logger = logging.getLogger("brain_tumor_project.trainer")


class BrainTumorTrainer:
    """
    Manages end-to-end model training, validation, early stopping, and checkpointing.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[Any] = None,
        device: torch.device = torch.device("cpu"),
        config: Optional[Dict[str, Any]] = None,
        checkpoint_dir: str = "checkpoints",
        checkpoint_name: str = "best_hybrid_model.pth",
        early_stopping_patience: int = 5
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.config = config or {}
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.checkpoint_dir / checkpoint_name
        self.early_stopping_patience = early_stopping_patience

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "learning_rate": []
        }

        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.patience_counter = 0

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Runs a single training epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for batch_idx, (images, labels, _) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            current_loss = running_loss / total
            current_acc = (correct / total) * 100.0
            pbar.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.2f}%"})

            # Free cache periodically on Apple Silicon MPS
            if self.device.type == "mps" and (batch_idx + 1) % 10 == 0:
                torch.mps.empty_cache()

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100.0
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> Tuple[float, float]:
        """Runs validation evaluation."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
        for images, labels, _ in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100.0
        return epoch_loss, epoch_acc

    def save_checkpoint(self, epoch: int, val_loss: float, val_acc: float):
        """Saves best model weights and metadata."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "history": self.history,
            "config": self.config
        }
        torch.save(checkpoint, self.checkpoint_path)
        logger.info(f"==> Checkpoint saved at {self.checkpoint_path} (Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%)")

    def fit(self, num_epochs: int) -> Dict[str, Any]:
        """
        Executes full training across num_epochs with early stopping.
        """
        logger.info(f"Starting training for {num_epochs} epochs on device: {self.device.type.upper()}")
        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            # Validate
            val_loss, val_acc = self.validate_epoch(epoch)

            # Record current LR
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["learning_rate"].append(current_lr)

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch [{epoch:02d}/{num_epochs:02d}] ({epoch_time:.1f}s) - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
                f"LR: {current_lr:.6f}"
            )

            # Early stopping check (based on validation loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_loss, val_acc)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs (Patience: {self.early_stopping_patience}).")
                    break

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds. Best Val Acc: {self.best_val_acc:.2f}%")

        # Save graphs and metrics
        self.plot_training_curves()
        save_json(self.history, "results/metrics/training_history.json")
        return self.history

    def plot_training_curves(self, output_dir: str = "results/figures"):
        """Generates and saves training and validation loss and accuracy graphs."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        epochs = range(1, len(self.history["train_loss"]) + 1)

        # 1. Loss Curve
        plt.figure(figsize=(9, 5))
        plt.plot(epochs, self.history["train_loss"], "b-o", label="Training Loss", linewidth=2)
        plt.plot(epochs, self.history["val_loss"], "r--s", label="Validation Loss", linewidth=2)
        plt.title("Cross-Entropy Loss vs. Epochs", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(out_path / "loss_curve.png", dpi=300)
        plt.close()

        # 2. Accuracy Curve
        plt.figure(figsize=(9, 5))
        plt.plot(epochs, self.history["train_acc"], "g-o", label="Training Accuracy", linewidth=2)
        plt.plot(epochs, self.history["val_acc"], "m--s", label="Validation Accuracy", linewidth=2)
        plt.title("Classification Accuracy vs. Epochs", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(out_path / "accuracy_curve.png", dpi=300)
        plt.close()

        logger.info(f"Training curves saved to {output_dir}/loss_curve.png and accuracy_curve.png")
