import numpy as np
import pytest
from srini_einops.operations import (
    reshape_tensor, infer_shape, transpose_tensor, 
    optimize_transpose, split_axis, merge_axes
)

def test_reshape():
    x = np.arange(24).reshape(4, 6)
    result = reshape_tensor(x, (6, 4))
    assert result.shape == (6, 4)
    assert np.array_equal(result.flatten(), x.flatten())

def test_infer_shape():
    x = np.arange(24).reshape(4, 6)
    result = infer_shape(x, [-1, 4])
    assert result.shape == (6, 4)
    assert np.array_equal(result.flatten(), x.flatten())

def test_transpose():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    result = transpose_tensor(x)
    assert result.shape == (3, 2)
    assert np.array_equal(result, x.T)

def test_split_axis():
    x = np.arange(24).reshape(6, 4)
    result = split_axis(x, 0, [2, 3])  # Split 6 into [2, 3]
    assert result.shape == (2, 3, 4)

def test_merge_axes():
    x = np.arange(24).reshape(2, 3, 4)
    result = merge_axes(x, [1, 2])
    assert result.shape == (2, 12)