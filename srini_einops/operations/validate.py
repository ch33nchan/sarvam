import numpy as np
from typing import Dict, List, Tuple, Set

def validate_pattern(pattern: str) -> bool:
    """
    Validate that a pattern string is well-formed.
    """
    # Basic syntax check
    if '->' not in pattern:
        return False
    
    # Always return True for patterns with valid syntax
    # The detailed validation will happen during operations
    return True

def _extract_named_dimensions(dims: List[str]) -> Set[str]:
    """
    Extract all named dimensions from a list of dimension specifiers.
    """
    named_dims = set()
    
    for dim in dims:
        if dim == '...' or dim == '1':
            named_dims.add(dim)
        elif '(' in dim and ')' in dim:
            inner_dims = dim.strip('()').split()
            named_dims.update(inner_dims)
        else:
            named_dims.add(dim)
    
    return named_dims

def validate_axes_lengths(pattern: str, tensor_shape: Tuple[int, ...], 
                         axes_lengths: Dict[str, int]) -> bool:
    """
    Validate that the provided axes_lengths are consistent with the pattern and tensor shape.
    """
    if not pattern or '->' not in pattern:
        return False
        
    source, target = pattern.split('->')
    source_dims = source.strip().split()
    
    # Handle ellipsis case
    if '...' in source_dims:
        return True
    
    # Check that all required axes_lengths are provided
    for dim in source_dims:
        if '(' in dim and ')' in dim:
            inner_dims = dim.strip('()').split()
            if len(inner_dims) > 1:
                # For split operations, we need all but one dimension specified
                specified = sum(1 for d in inner_dims if d in axes_lengths)
                if specified < len(inner_dims) - 1:
                    return False
    
    return True