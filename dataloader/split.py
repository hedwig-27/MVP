# Utility functions for splitting data (if needed).
# (In this project, data splits are assumed to be provided in the HDF5 file.)
import numpy as np

def train_val_test_split(indices, val_frac=0.1, test_frac=0.1, shuffle=True):
    """
    Split a list of indices into train, val, test sets.
    Returns three lists: train_indices, val_indices, test_indices.
    """
    if shuffle:
        np.random.shuffle(indices)
    n = len(indices)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    n_train = n - n_val - n_test
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    return train_idx, val_idx, test_idx

# Example usage (not called by default): 
# indices = np.arange(num_samples)
# train_idx, val_idx, test_idx = train_val_test_split(indices)
