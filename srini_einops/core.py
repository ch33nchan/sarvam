import numpy as np

def rearrange(tensor, pattern, **axes_lengths):
    """Rearranges tensor dimensions according to the pattern."""
    # Parse pattern
    source, target = pattern.split('->')
    source_dims = source.strip().split()
    target_dims = target.strip().split()
    
    # Get tensor shape
    tensor_shape = list(tensor.shape)
    
    # Initialize shape dictionary with provided axes lengths
    shape_dict = axes_lengths.copy()
    
    # First pass: map non-composite dimensions directly from tensor shape
    for i, dim in enumerate(source_dims):
        if '(' not in dim and ')' not in dim:
            shape_dict[dim] = tensor_shape[i]
    
    # Second pass: handle composite dimensions
    for i, dim in enumerate(source_dims):
        if '(' in dim and ')' in dim:
            total_size = tensor_shape[i]
            inner_dims = dim.strip('()').split()
            
            # Calculate product of known dimensions
            known_product = 1
            unknown_dims = []
            for d in inner_dims:
                if d in shape_dict:
                    known_product *= shape_dict[d]
                else:
                    unknown_dims.append(d)
            
            # If only one unknown dimension, calculate its size
            if len(unknown_dims) == 1:
                shape_dict[unknown_dims[0]] = total_size // known_product
    
    # Build reshape dimensions and create flat source dimensions list
    reshape_dims = []
    flat_source_dims = []
    
    for dim in source_dims:
        if '(' in dim and ')' in dim:
            inner_dims = dim.strip('()').split()
            for d in inner_dims:
                reshape_dims.append(shape_dict[d])
                flat_source_dims.append(d)
        else:
            reshape_dims.append(shape_dict[dim])
            flat_source_dims.append(dim)
    
    # Reshape tensor
    reshaped = tensor.reshape(reshape_dims)
    
    # Build permutation if needed
    if flat_source_dims != target_dims:
        source_pos = {dim: i for i, dim in enumerate(flat_source_dims)}
        perm = [source_pos[dim] for dim in target_dims]
        reshaped = np.transpose(reshaped, perm)
    
    return reshaped