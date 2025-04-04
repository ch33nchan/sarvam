import numpy as np
import pytest
from srini_einops import reduce, repeat, rearrange

def test_basic_rearrange():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    result = rearrange(x, 'h w -> w h')
    assert result.shape == (3, 2)
    np.testing.assert_array_equal(result, x.T)

def test_basic_reduce():
    x = np.random.rand(2, 3, 4)
    result = reduce(x, 'b h w -> b h', reduction='mean')
    assert result.shape == (2, 3)
    np.testing.assert_array_almost_equal(result, np.mean(x, axis=2))

def test_basic_repeat():
    x = np.array([[1, 2], [3, 4]])
    result = repeat(x, 'h w -> h w c', c=3)
    assert result.shape == (2, 2, 3)