import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def fft_feature(field):
    """
    Example utility: compute Fourier features (magnitude of frequencies) of a 1D field.
    field: numpy array or tensor of shape (X,) or (X,C).
    Returns numpy array of magnitude of rfft frequencies.
    """
    arr = field.numpy() if isinstance(field, torch.Tensor) else field
    freqs = np.fft.rfft(arr, axis=0)
    mag = np.abs(freqs)
    return mag


def create_run_dir(exp_name, output_root="outputs"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"{exp_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def set_latest_link(run_dir, link_name, output_root="outputs"):
    link_path = Path(output_root) / link_name
    try:
        if link_path.exists() or link_path.is_symlink():
            if link_path.is_symlink() or link_path.is_file():
                link_path.unlink()
            elif link_path.is_dir():
                backup = link_path.with_name(
                    f"{link_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                link_path.rename(backup)
        link_path.symlink_to(run_dir.resolve(), target_is_directory=True)
        return True
    except OSError:
        return False


def setup_logger(exp_name, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"

    logger = logging.getLogger(f"{exp_name}_{output_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger, log_path
