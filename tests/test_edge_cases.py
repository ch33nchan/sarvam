import numpy as np
import pytest
from srini_einops.operations import *

def test_empty_tensors():
    x = np.array([])
    with pytest.raises(ValueError):
        reshape_tensor(x, (0,))

def test_zero_dimensions():
    x = np.zeros((2, 0, 3))
    with pytest.raises(ValueError):
        rearrange_pattern(x, 'b h w -> (b h) w')

def test_large_dimensions():
    x = np.random.rand(1000, 1000)
    result = reshape_tensor(x, (1000000, 1))
    assert result.shape == (1000000, 1)

def test_invalid_patterns():
    x = np.random.rand(2, 3, 4)
    
    # Test invalid pattern syntax
    with pytest.raises(ValueError):
        rearrange_pattern(x, 'invalid pattern')
    
    # Test mismatched dimensions
    with pytest.raises(ValueError):
        rearrange_pattern(x, 'b h w -> b h w d')
    
    # Test invalid reduction
    with pytest.raises(ValueError):
        reduce_pattern(x, 'b h w -> h', reduction='invalid')

def test_dimension_mismatch():
    x = np.random.rand(2, 3, 4)
    with pytest.raises(ValueError):
        split_axis(x, 0, [3, 3])  # Should be 2