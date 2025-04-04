import numpy as np
from typing import List, Tuple

def split_axis(tensor: np.ndarray, axis: int, sizes: List[int]) -> np.ndarray:
    """Split single axis into multiple dimensions"""
    # Validate input
    axis_size = tensor.shape[axis]
    product_sizes = np.prod(sizes)
    
    if product_sizes != axis_size:
        raise ValueError(f"Product of sizes {product_sizes} doesn't match axis size {axis_size}")
    
    # Calculate new shape
    new_shape = list(tensor.shape)
    new_shape[axis:axis+1] = sizes
    
    return tensor.reshape(new_shape)

def split_pattern(tensor: np.ndarray, pattern: str) -> np.ndarray:
    """Split axes according to pattern"""
    source, target = pattern.split('->')
    source_dims = source.strip().split()
    target_dims = target.strip().split()
    return tensor.reshape([tensor.shape[i] for i in range(len(source_dims))])