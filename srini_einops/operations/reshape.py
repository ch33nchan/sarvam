import numpy as np
from typing import Tuple, List

def reshape_tensor(tensor: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Reshape tensor with validation"""
    if tensor.size == 0 and any(s == 0 for s in shape):
        raise ValueError("Cannot reshape empty tensor with zero dimension")
    return np.reshape(tensor, shape)

def infer_shape(tensor: np.ndarray, shape: List[int]) -> np.ndarray:
    """Auto-infer -1 dimensions"""
    total_elements = tensor.size
    known_elements = abs(np.prod([x for x in shape if x != -1]))
    unknown_dim = total_elements // known_elements
    return reshape_tensor(tensor, tuple(unknown_dim if x == -1 else x for x in shape))