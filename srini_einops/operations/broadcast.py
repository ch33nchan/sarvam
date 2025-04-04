import numpy as np
from typing import List, Tuple

def broadcast_to_shape(tensor: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Broadcast tensor to target shape"""
    return np.broadcast_to(tensor, shape)

def broadcast_tensors(*tensors: np.ndarray) -> List[np.ndarray]:
    """Broadcast multiple tensors to compatible shapes"""
    return np.broadcast_arrays(*tensors)