import numpy as np
from typing import List, Tuple

def merge_axes(tensor: np.ndarray, axes: List[int]) -> np.ndarray:
    """Merge multiple axes into one"""
    shape = list(tensor.shape)
    merged_size = np.prod([shape[i] for i in axes])
    new_shape = [s for i, s in enumerate(shape) if i not in axes] + [merged_size]
    return np.reshape(tensor, new_shape)

def merge_pattern(tensor: np.ndarray, pattern: str) -> np.ndarray:
    """Merge axes according to pattern"""
    source, target = pattern.split('->')
    source_dims = source.strip().split()
    target_dims = target.strip().split()
    return tensor.reshape([tensor.shape[i] for i in range(len(target_dims))])