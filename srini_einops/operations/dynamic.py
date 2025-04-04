import numpy as np
from typing import Dict

def dynamic_naming(tensor: np.ndarray, pattern: str, **named_sizes: Dict[str, int]) -> np.ndarray:
    """Handle dynamic axis naming and reuse"""
    source, target = pattern.split('->')
    source_dims = source.strip().split()
    target_dims = target.strip().split()
    
    # Build dimension size dictionary
    dim_sizes = {}
    for i, dim in enumerate(source_dims):
        if dim in named_sizes:
            dim_sizes[dim] = named_sizes[dim]
        else:
            dim_sizes[dim] = tensor.shape[i]
    
    # Parse target pattern
    target_shape = []
    current_group = []
    
    for dim in target_dims:
        if '(' in dim:
            # Start of group
            current_group = [dim.strip('(')]
        elif ')' in dim:
            # End of group
            current_group.append(dim.strip(')'))
            size = np.prod([dim_sizes[d] for d in current_group])
            target_shape.append(size)
            current_group = []
        elif current_group:
            # Middle of group
            current_group.append(dim)
        else:
            # Single dimension
            target_shape.append(dim_sizes[dim])
    
    return tensor.reshape(target_shape)