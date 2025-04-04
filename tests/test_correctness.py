import numpy as np
import pytest
from srini_einops.operations import *

def test_basic_operations():
    x = np.arange(24).reshape(2, 3, 4)
    
    # Test reshape
    assert reshape_tensor(x, (6, 4)).shape == (6, 4)
    
    # Test transpose
    assert transpose_tensor(x, (2, 0, 1)).shape == (4, 2, 3)
    
    # Test split
    assert split_axis(x, 0, [1, 2]).shape == (1, 2, 3, 4)
    
    # Test merge
    assert merge_axes(x, [1, 2]).shape == (2, 12)

def test_pattern_operations():
    x = np.arange(24).reshape(2, 3, 4)
    
    # Test rearrange
    result = rearrange_pattern(x, 'b h w -> (b h) w')
    assert result.shape == (6, 4)
    
    # Test reduce
    result = reduce_pattern(x, 'b h w -> b w', reduction='mean')
    assert result.shape == (2, 4)
    
    # Test repeat
    result = repeat_pattern(x, 'b h w -> b h w c', c=2)
    assert result.shape == (2, 3, 4, 2)

def test_dimension_inference():
    x = np.arange(24).reshape(2, 3, 4)
    
    # Test infer shape
    result = infer_shape(x, [-1, 4, 3])
    assert result.shape == (2, 4, 3)
    
    # Test with named dimensions
    result = rearrange_pattern(x, 'b h w -> (b h) w', b=2)
    assert result.shape == (6, 4)