import gc
import os
import random
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # torch may not be present until requirements are installed
    torch = None

# Paths
DATA_ROOT = Path(os.environ.get("LYFT_DATA_ROOT", "lyft_data"))
ZARR_CANDIDATES = (
    "scenes/train.zarr",
    "scenes/validate.zarr",
    "scenes/sample.zarr",
)

# Safety/default hyperparameters (match the notebook defaults)
SAFE_MAX_SAMPLES = 2000
SAFE_KMAX = 24
SAFE_H = 10
SAFE_T = 30
SAFE_FRAMES_PER_SC = 600
SAFE_HIDDEN = 128
SAFE_BATCH_CNN = 128
K_MODES = 3


def get_device() -> str:
    """Return cuda if available, else cpu."""
    if torch is None:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seeds(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def clear_mem() -> None:
    """Free Python + CUDA memory."""
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
