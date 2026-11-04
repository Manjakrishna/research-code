"""
Utility functions for configuration, reproducibility, device selection,
and serialization.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TORCH_HOME", "/tmp/torch")
import random
import json
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import torch

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("brain_tumor_project")


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42) -> None:
    """Set random seed for Python, NumPy, and PyTorch across CPU, CUDA, and MPS."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS uses manual_seed
        pass
    logger.info(f"Random seed set to {seed} for reproducibility.")


def get_device(preference: str = "auto") -> torch.device:
    """
    Select the best available compute device:
    Apple Silicon MPS > NVIDIA CUDA > CPU fallback.
    """
    preference = preference.lower().strip()
    if preference == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif preference == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif preference == "cpu":
        device = torch.device("cpu")
    else:  # auto
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            
    logger.info(f"Using compute device: {device.type.upper()}")
    return device


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters of a PyTorch module."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Safely write dictionary to a JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved JSON data to {filepath}")


def load_json(filepath: str) -> Dict[str, Any]:
    """Safely load dictionary from a JSON file."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found at: {filepath}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
