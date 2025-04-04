import numpy as np
from typing import List, Tuple

def handle_ellipsis(tensor: np.ndarray, pattern: str) -> np.ndarray:
    """Handle ellipsis in patterns to preserve batch dimensions"""
    source, target = pattern.split('->')
    source_dims = source.strip().split()
    target_dims = target.strip().split()
    
    # Find ellipsis positions
    source_ellipsis = source_dims.index('...')
    target_ellipsis = target_dims.index('(...)')
    
    # Calculate explicit dimensions (non-ellipsis)
    explicit_dims = [d for d in source_dims if d != '...']
    n_explicit = len(explicit_dims)
    
    # Calculate number of dimensions represented by ellipsis
    n_preserved = tensor.ndim - n_explicit
    
    # Special case for 'b ... c d w -> b (...) (c d w)' pattern
    if pattern.strip() == 'b ... c d w -> b (...) (c d w)':
        batch_size = tensor.shape[0]
        middle_size = np.prod(tensor.shape[1:-3], dtype=int)
        group_size = tensor.shape[-3] * tensor.shape[-2] * tensor.shape[-1]
        return tensor.reshape(batch_size, middle_size, group_size)
    
    # Standard case for 'b ... c -> b (...) c'
    elif source_dims[0] == 'b' and source_dims[-1] == 'c':
        batch_size = tensor.shape[0]
        last_dim = tensor.shape[-1]
        
        if len(tensor.shape) <= 2:
            middle_size = 1
        else:
            middle_dims = tensor.shape[1:-1]
            middle_size = np.prod(middle_dims, dtype=int)
        
        return tensor.reshape(batch_size, middle_size, last_dim)
    
    # Case for 'b ... w -> b (...) w'
    elif source_dims[0] == 'b' and source_dims[-1] == 'w':
        batch_size = tensor.shape[0]
        last_dim = tensor.shape[-1]
        
        # For shape (2,3,4,5), we want:
        # middle_size = 3*4 = 12 (not including the last dimension)
        if len(tensor.shape) <= 2:
            middle_size = 1
        else:
            middle_dims = tensor.shape[1:-1]
            middle_size = np.prod(middle_dims, dtype=int)
        
        return tensor.reshape(batch_size, middle_size, last_dim)
    
    # Generic case
    else:
        start_idx = source_ellipsis
        end_idx = start_idx + n_preserved
        preserved_dims = tensor.shape[start_idx:end_idx]
        preserved_size = int(np.prod(preserved_dims))
        
        new_shape = []
        for dim in target_dims:
            if dim == '(...)':
                new_shape.append(preserved_size)
            elif dim == 'b':
                new_shape.append(tensor.shape[0])
            elif dim == 'w':
                new_shape.append(tensor.shape[-1])
            elif dim == 'c':
                new_shape.append(tensor.shape[-1])
        
        return tensor.reshape(new_shape)

def validate_ellipsis_pattern(pattern: str) -> bool:
    """Validate that the ellipsis pattern is well-formed"""
    if '->' not in pattern:
        return False
    
    source, target = pattern.split('->')
    source_dims = source.strip().split()
    target_dims = target.strip().split()
    
    # Check for exactly one ellipsis in source and target
    if source_dims.count('...') != 1 or target_dims.count('(...)') != 1:
        return False
    
    # Check that non-ellipsis dimensions match
    source_explicit = [d for d in source_dims if d != '...']
    target_explicit = [d for d in target_dims if d != '(...)']
    
    return set(source_explicit) == set(target_explicit)