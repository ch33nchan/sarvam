import numpy as np
import pytest
from srini_einops.operations import (
    broadcast_to_shape, broadcast_tensors,
    handle_ellipsis, dynamic_naming, complex_reorder
)

def test_broadcast():
    x = np.array([1, 2, 3])
    result = broadcast_to_shape(x, (3, 3))
    assert result.shape == (3, 3)
    assert np.array_equal(result[0], x)

def test_ellipsis():
    x = np.random.rand(2, 3, 4, 5)
    result = handle_ellipsis(x, 'b ... w -> b (...) w')
    assert result.shape == (2, 12, 5)

def test_dynamic_naming():
    x = np.random.rand(2, 3, 4)
    result = dynamic_naming(x, 'b h w -> (b h) w', b=2, h=3)
    assert result.shape == (6, 4)

def test_complex_reorder():
    x = np.arange(24).reshape(2, 3, 4)
    result = complex_reorder(x, 'b h w -> (w h) b')
    assert result.shape == (12, 2)