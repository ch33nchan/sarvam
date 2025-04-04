from typing import List, Dict, Tuple, Set

def parse_pattern(pattern: str) -> Tuple[List[str], List[str]]:
    """
    Parse an einops pattern string into source and target dimensions.
    
    Args:
        pattern: String pattern in einops format
        
    Returns:
        Tuple of (source_dims, target_dims)
    
    Raises:
        ValueError: If the pattern is invalid
    """
    if '->' not in pattern:
        raise ValueError(f"Invalid pattern: {pattern}. Must contain '->'")
    
    source, target = pattern.split('->')
    source_dims = source.strip().split()
    target_dims = target.strip().split()
    
    return source_dims, target_dims

def validate_pattern(pattern: str) -> bool:
    """
    Validate that a pattern string is well-formed.
    
    Args:
        pattern: String pattern in einops format
        
    Returns:
        True if the pattern is valid, False otherwise
    """
    try:
        source_dims, target_dims = parse_pattern(pattern)
        
        # Check for balanced parentheses
        for dims in [source_dims, target_dims]:
            for dim in dims:
                if dim.count('(') != dim.count(')'):
                    return False
        
        # Check that all dimensions in target exist in source (except for repeats)
        source_dim_set = set()
        for dim in source_dims:
            if '(' in dim and ')' in dim:
                # This is a grouped dimension
                components = dim.strip('()').split()
                source_dim_set.update(components)
            elif dim != '...' and dim != '1':
                source_dim_set.add(dim)
        
        target_dim_set = set()
        for dim in target_dims:
            if '(' in dim and ')' in dim:
                # This is a grouped dimension
                components = dim.strip('()').split()
                target_dim_set.update(components)
            elif dim != '(...)':
                target_dim_set.add(dim)
        
        # All target dims should be in source dims (except for repeats)
        for dim in target_dim_set:
            if dim not in source_dim_set and dim != '1':
                return False
        
        return True
    
    except Exception:
        return False

def extract_dim_sizes(tensor_shape: Tuple[int, ...], source_dims: List[str]) -> Dict[str, int]:
    """
    Extract dimension sizes from a tensor shape.
    
    Args:
        tensor_shape: Shape of the tensor
        source_dims: List of dimension names from the source pattern
        
    Returns:
        Dictionary mapping dimension names to their sizes
    """
    dim_sizes = {}
    
    # Handle simple dimensions
    for i, dim in enumerate(source_dims):
        if dim != '...' and '(' not in dim and ')' not in dim:
            dim_sizes[dim] = tensor_shape[i]
    
    return dim_sizes