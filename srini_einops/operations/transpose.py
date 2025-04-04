import numpy as np
from typing import List, Optional, Tuple

def transpose_tensor(tensor: np.ndarray, axes: Optional[List[int]] = None) -> np.ndarray:
    """Memory-efficient transposition"""
    return np.transpose(tensor, axes)

def optimize_transpose(tensor: np.ndarray, source_pattern: str, target_pattern: str) -> np.ndarray:
    """Optimize transposition based on pattern"""
    source_axes = source_pattern.split()
    target_axes = target_pattern.split()
    perm = [source_axes.index(ax) for ax in target_axes]
    return transpose_tensor(tensor, perm)