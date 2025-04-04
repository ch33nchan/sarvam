import numpy as np
from typing import Dict, List, Tuple, Optional, Union

def rearrange(tensor: np.ndarray, pattern: str, **axes_lengths) -> np.ndarray:
    """Pure NumPy implementation of tensor rearrangement"""
    # Parse input and output patterns
    in_pattern, out_pattern = pattern.split('->')
    in_axes = in_pattern.strip().split()
    out_axes = out_pattern.strip().split()
    
    # Handle composite axes (e.g., '(h w)')
    composite_map = {}
    for pattern in [in_axes, out_axes]:
        for i, ax in enumerate(pattern):
            if '(' in ax and ')' in ax:
                inner = ax[1:-1].split()
                composite_map[ax] = inner
                pattern[i] = '_'.join(inner)
    
    # Build shape dictionary
    shape_dict = {}
    for ax, size in zip(in_axes, tensor.shape):
        if ax in composite_map:
            sub_axes = composite_map[ax]
            sizes = _compute_split_sizes(size, len(sub_axes), axes_lengths)
            for sub_ax, sub_size in zip(sub_axes, sizes):
                shape_dict[sub_ax] = sub_size
        else:
            shape_dict[ax] = size
    
    # Compute intermediate shape and permutation
    flat_in = []
    reshape_shape = []
    for ax in in_axes:
        if ax in composite_map:
            flat_in.extend(composite_map[ax])
            reshape_shape.extend(shape_dict[x] for x in composite_map[ax])
        else:
            flat_in.append(ax)
            reshape_shape.append(shape_dict[ax])
    
    # Reshape to split dimensions
    tensor = tensor.reshape(reshape_shape)
    
    # Compute permutation
    flat_out = []
    for ax in out_axes:
        if ax in composite_map:
            flat_out.extend(composite_map[ax])
        else:
            flat_out.append(ax)
    
    permutation = [flat_in.index(ax) for ax in flat_out]
    
    # Apply permutation
    if permutation != list(range(len(permutation))):
        tensor = np.transpose(tensor, permutation)
    
    # Compute final shape
    final_shape = []
    for ax in out_axes:
        if ax in composite_map:
            size = np.prod([shape_dict[x] for x in composite_map[ax]])
            final_shape.append(size)
        else:
            final_shape.append(shape_dict[ax])
    
    return tensor.reshape(final_shape)

def reduce(tensor: np.ndarray, pattern: str, reduction: str = 'sum', **axes_lengths) -> np.ndarray:
    """Pure NumPy implementation of tensor reduction"""
    reduction_ops = {
        'sum': np.sum,
        'mean': np.mean,
        'max': np.max,
        'min': np.min,
        'prod': np.prod
    }
    
    if reduction not in reduction_ops:
        raise ValueError(f"Unknown reduction: {reduction}")
    
    # Parse patterns
    in_pattern, out_pattern = pattern.split('->')
    in_axes = in_pattern.strip().split()
    out_axes = out_pattern.strip().split()
    
    # Find reduction axes
    reduce_axes = []
    for i, ax in enumerate(in_axes):
        if ax not in out_axes:
            reduce_axes.append(i)
    
    # Apply reduction
    if reduce_axes:
        tensor = reduction_ops[reduction](tensor, axis=tuple(reduce_axes))
    
    # Rearrange remaining dimensions if needed
    if out_axes != [ax for ax in in_axes if ax in out_axes]:
        remaining_pattern = f"{' '.join(ax for ax in in_axes if ax in out_axes)} -> {' '.join(out_axes)}"
        tensor = rearrange(tensor, remaining_pattern)
    
    return tensor

def repeat(tensor: np.ndarray, pattern: str, **axes_lengths) -> np.ndarray:
    """Pure NumPy implementation of tensor repetition"""
    in_pattern, out_pattern = pattern.split('->')
    in_axes = in_pattern.strip().split()
    out_axes = out_pattern.strip().split()
    
    # Find new axes to repeat
    repeat_axes = [ax for ax in out_axes if ax not in in_axes]
    
    # First rearrange existing axes
    if in_axes != [ax for ax in out_axes if ax in in_axes]:
        common_pattern = (
            f"{' '.join(in_axes)} -> "
            f"{' '.join(ax for ax in out_axes if ax in in_axes)}"
        )
        tensor = rearrange(tensor, common_pattern)
    
    # Then add new axes and repeat
    for ax in repeat_axes:
        if ax not in axes_lengths:
            raise ValueError(f"Size for repeated axis '{ax}' not provided")
        repeat_size = axes_lengths[ax]
        tensor = np.expand_dims(tensor, -1)
        tensor = np.repeat(tensor, repeat_size, axis=-1)
    
    return tensor

def _compute_split_sizes(total_size: int, n_splits: int, axes_lengths: Dict[str, int]) -> List[int]:
    """Helper function to compute split sizes"""
    if n_splits == 1:
        return [total_size]
    
    sizes = [axes_lengths.get(f'_{i}', -1) for i in range(n_splits)]
    unknown_idx = sizes.index(-1) if -1 in sizes else None
    
    if unknown_idx is not None:
        known_product = np.prod([s for s in sizes if s != -1])
        if total_size % known_product != 0:
            raise ValueError(f"Cannot divide {total_size} into {n_splits} parts")
        sizes[unknown_idx] = total_size // known_product
    
    return sizes