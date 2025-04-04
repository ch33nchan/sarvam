import numpy as np
import pytest
from scipy import sparse
from srini_einops.operations import (
    optimize_reshape, handle_sparse, jit_reshape
)

def test_optimize_reshape():
    x = np.random.rand(100, 100)
    result = optimize_reshape(x, (10000,))
    assert result.shape == (10000,)
    assert np.array_equal(result, x.reshape(10000))

def test_sparse_handling():
    # Create sparse matrix
    x = sparse.csr_matrix(([1, 2, 3], ([0, 1, 2], [1, 2, 0])), shape=(3, 3))
    
    # Test reshape
    result = handle_sparse(x, 'reshape', shape=(9,))
    assert result.shape == (9,)
    
    # Test transpose
    result = handle_sparse(x, 'transpose', axes=(1, 0))
    assert result.shape == (3, 3)
    assert sparse.issparse(result)

def test_jit_reshape():
    x = np.random.rand(10, 10)
    result = jit_reshape(x, (100,))
    assert result.shape == (100,)
    assert np.array_equal(result, x.reshape(100))