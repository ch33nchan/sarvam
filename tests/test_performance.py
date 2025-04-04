import numpy as np
import pytest
import time
from srini_einops.operations import *

def test_reshape_performance():
    x = np.random.rand(1000, 1000)
    
    start_time = time.time()
    result = reshape_tensor(x, (1000000, 1))
    reshape_time = time.time() - start_time
    
    assert reshape_time < 0.1  # Should be fast

def test_pattern_compiler_cache():
    compiler = PatternCompiler()
    pattern = 'b h w -> (b h) w'
    
    # First compilation
    _ = compiler.compile_pattern(pattern)
    
    # Second compilation (should be cached)
    start_time = time.time()
    _ = compiler.compile_pattern(pattern)
    cache_time = time.time() - start_time
    
    assert cache_time < 0.001  # Should be very fast due to caching

def test_memory_efficiency():
    x = np.random.rand(1000, 1000)
    initial_memory = x.nbytes
    
    result = rearrange_pattern(x, 'h w -> w h')
    final_memory = result.nbytes
    
    assert final_memory <= initial_memory * 1.1  # Allow 10% overhead