import numpy as np
import pytest
from srini_einops.operations import *

def test_nested_patterns():
    x = np.random.rand(2, 3, 4, 5)
    result = rearrange_pattern(x, 'b h w c -> b (h w) c')
    assert result.shape == (2, 12, 5)

def test_multiple_groups():
    x = np.random.rand(2, 3, 4, 5)
    result = rearrange_pattern(x, 'b h w c -> (b h) (w c)')
    assert result.shape == (6, 20)

def test_ellipsis_handling():
    x = np.random.rand(2, 3, 4, 5, 6)
    result = handle_ellipsis(x, 'b ... c -> b (...) c')

def test_dynamic_naming():
    x = np.random.rand(2, 3, 4)
    result = dynamic_naming(x, 'time h w -> (time h) w', time=2)
    assert result.shape == (6, 4)

def test_advanced_ellipsis_patterns():
    # Test with different dimension combinations
    x = np.random.rand(2, 3, 4, 5, 6, 7)
    result1 = handle_ellipsis(x, 'b ... c -> b (...) c')
    assert result1.shape == (2, 360, 7)  # batch, 3*4*5*6, last_dim
    
    # Test with edge case
    x2 = np.random.rand(2, 3, 4)
    result2 = handle_ellipsis(x2, 'b ... c -> b (...) c')
    assert result2.shape == (2, 3, 4)  # batch, middle, last_dim
    
    # Test with multiple explicit dimensions
    x3 = np.random.rand(2, 3, 4, 5, 6, 7, 8)
    result3 = handle_ellipsis(x3, 'b ... c -> b (...) c')
    assert result3.shape == (2, 2520, 8)  # batch, 3*4*5*6*7, last_dim
    
    x2 = np.random.rand(2, 3, 4, 5, 6, 7, 8)
    result2 = handle_ellipsis(x2, 'b ... c d w -> b (...) (c d w)')
    assert result2.shape == (2, 60, 336)  # 3*4*5, 6*7*8

def test_edge_case_ellipsis():
    # Test with single dimension in ellipsis
    x = np.random.rand(2, 3, 4)
    result = handle_ellipsis(x, 'b ... c -> b (...) c')
    assert result.shape == (2, 3, 4)
    
    # Test with no dimensions in ellipsis
    x2 = np.random.rand(2, 4)
    result2 = handle_ellipsis(x2, 'b ... c -> b (...) c')
    assert result2.shape == (2, 1, 4)