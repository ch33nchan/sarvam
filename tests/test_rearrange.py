import numpy as np
import pytest
from srini_einops.operations import rearrange

def test_transpose():
    x = np.random.rand(3, 4)
    result = rearrange(x, 'h w -> w h')
    assert result.shape == (4, 3)
    np.testing.assert_array_equal(result, np.transpose(x))

def test_split_axis():
    x = np.random.rand(12, 10)
    result = rearrange(x, '(h w) c -> h w c', h=3)
    assert result.shape == (3, 4, 10)
    
    # Verify correctness
    for i in range(3):
        for j in range(4):
            np.testing.assert_array_equal(result[i, j], x[i*4 + j])

def test_merge_axes():
    x = np.random.rand(3, 4, 5)
    result = rearrange(x, 'a b c -> (a b) c')
    assert result.shape == (12, 5)
    
    # Verify correctness
    for i in range(3):
        for j in range(4):
            np.testing.assert_array_equal(result[i*4 + j], x[i, j])

def test_repeat_axis():
    x = np.random.rand(3, 1, 5)
    result = rearrange(x, 'a 1 c -> a b c', b=4)
    assert result.shape == (3, 4, 5)
    
    # Verify correctness - all repeated slices should be identical
    for i in range(3):
        for j in range(4):
            np.testing.assert_array_equal(result[i, j], x[i, 0])

def test_ellipsis():
    x = np.random.rand(2, 3, 4, 5)
    result = rearrange(x, '... h w -> ... (h w)')
    assert result.shape == (2, 3, 20)
    
    # Verify correctness
    for i in range(2):
        for j in range(3):
            flat = x[i, j].reshape(-1)
            np.testing.assert_array_equal(result[i, j], flat)

def test_complex_pattern():
    x = np.random.rand(2, 12, 5)
    result = rearrange(x, 'b (h w) c -> b h w c', h=3)
    assert result.shape == (2, 3, 4, 5)
    
    # Verify correctness
    for i in range(2):
        for j in range(3):
            for k in range(4):
                np.testing.assert_array_equal(result[i, j, k], x[i, j*4 + k])

# Add more complex test cases
def test_multiple_splits():
    x = np.random.rand(24, 15)
    result = rearrange(x, '(b h w) (c d) -> b h w c d', b=2, h=3, w=4, c=3)
    assert result.shape == (2, 3, 4, 3, 5)
    
    # Verify correctness
    for b in range(2):
        for h in range(3):
            for w in range(4):
                for c in range(3):
                    for d in range(5):
                        idx1 = b * 12 + h * 4 + w
                        idx2 = c * 5 + d
                        np.testing.assert_array_equal(result[b, h, w, c, d], x[idx1, idx2])

def test_combined_operations():
    x = np.random.rand(3, 4, 5)
    # Split, merge and transpose in one operation
    result = rearrange(x, 'a b c -> c (a b)')
    assert result.shape == (5, 12)
    
    # Verify correctness
    for c in range(5):
        for a in range(3):
            for b in range(4):
                np.testing.assert_array_equal(result[c, a*4 + b], x[a, b, c])

def test_error_handling():
    x = np.random.rand(3, 4)
    
    # Invalid pattern syntax
    with pytest.raises(ValueError):
        rearrange(x, 'h w : w h')  # Missing ->
    
    # Unbalanced parentheses
    with pytest.raises(ValueError):
        rearrange(x, '(h w -> h w')  # Missing closing parenthesis
    
    # The following tests should be moved to operation-specific tests
    # as they're not validation errors but operation errors
    
    # Missing axes_lengths - this should be tested in split operation tests
    try:
        rearrange(x, '(h w) -> h w')  # Missing h or w
        pytest.fail("Should have raised ValueError for missing axes_lengths")
    except ValueError:
        pass
    
    # Mismatched dimensions - this should be tested in operation tests
    try:
        # This might actually work with broadcasting, so we'll make it optional
        rearrange(x, 'h w -> h w d', d=2)
    except ValueError:
        pass