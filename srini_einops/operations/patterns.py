import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from .split import split_axis
from .merge import merge_axes

def rearrange_pattern(tensor: np.ndarray, pattern: str, **axes_lengths) -> np.ndarray:
    """Implement pattern-based rearrangement"""
    # Validate pattern format
    if '->' not in pattern:
        raise ValueError("Invalid pattern format: missing '->'")
    
    source, target = pattern.split('->')
    source_dims = source.strip().split()
    target_dims = target.strip().split()
    
    # Validate dimensions
    if not all(dim in source_dims for dim in target_dims if '(' not in dim and ')' not in dim):
        raise ValueError("Target pattern contains undefined dimensions")
    
    # Handle zero dimensions
    if 0 in tensor.shape:
        raise ValueError("Cannot rearrange tensor with zero-sized dimensions")
    
    # Build shape dictionary
    shape_dict = {}
    for i, dim in enumerate(source_dims):
        shape_dict[dim] = tensor.shape[i]
    
    # Parse target pattern
    target_shape = []
    current_group = []
    
    for dim in target_dims:
        if '(' in dim:
            current_group = [dim.strip('(')]
        elif ')' in dim:
            current_group.append(dim.strip(')'))
            size = np.prod([shape_dict[d] for d in current_group])
            target_shape.append(size)
            current_group = []
        elif current_group:
            current_group.append(dim)
        else:
            target_shape.append(shape_dict[dim])
    
    return tensor.reshape(target_shape)

def reduce_pattern(tensor: np.ndarray, pattern: str, reduction: str = 'mean') -> np.ndarray:
    """Implement pattern-based reduction"""
    reductions = {
        'mean': np.mean,
        'sum': np.sum,
        'max': np.max,
        'min': np.min,
        'prod': np.prod
    }
    
    if reduction not in reductions:
        raise ValueError(f"Unknown reduction: {reduction}")
    
    source, target = pattern.split('->')
    source_dims = source.strip().split()
    target_dims = target.strip().split()
    
    # Find reduction axes
    reduce_axes = [i for i, dim in enumerate(source_dims) if dim not in target_dims]
    
    # Apply reduction
    return reductions[reduction](tensor, axis=tuple(reduce_axes))

def repeat_pattern(tensor: np.ndarray, pattern: str, **axes_lengths) -> np.ndarray:
    """Implement pattern-based repetition"""
    source, target = pattern.split('->')
    source_dims = source.strip().split()
    target_dims = target.strip().split()
    
    # Find new dimensions to repeat
    new_dims = [d for d in target_dims if d not in source_dims]
    
    # Add new axes
    for dim in new_dims:
        if dim not in axes_lengths:
            raise ValueError(f"Size for repeated axis '{dim}' not provided")
        tensor = np.expand_dims(tensor, -1)
        tensor = np.repeat(tensor, axes_lengths[dim], axis=-1)
    
    return tensor