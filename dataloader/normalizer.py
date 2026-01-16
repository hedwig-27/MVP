import numpy as np
import torch

class Normalizer:
    """
    Normalize data using mean and standard deviation.
    """
    def __init__(self, data_sample):
        """
        data_sample: a sample tensor or numpy array to compute normalization parameters.
        If data_sample is a large dataset, consider providing a subset or precomputed stats.
        """
        data = data_sample.astype('float32') if isinstance(data_sample, np.ndarray) else data_sample.float().numpy()
        self.mean = data.mean()
        self.std = data.std() + 1e-8
    
    def encode(self, x):
        # Normalize tensor or numpy array
        x_tensor = x if isinstance(x, torch.Tensor) else torch.from_numpy(x)
        return (x_tensor - self.mean) / self.std
    
    def decode(self, x):
        x_tensor = x if isinstance(x, torch.Tensor) else torch.from_numpy(x)
        return x_tensor * self.std + self.mean
