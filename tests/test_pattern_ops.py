import numpy as np
import pytest
from srini_einops.operations import (
    rearrange_pattern, reduce_pattern, repeat_pattern,
    PatternCompiler
)

def test_rearrange_pattern():
    # Test basic rearrangement
    x = np.arange(24).reshape(2, 3, 4)
    result = rearrange_pattern(x, 'b h w -> b (h w)')
    assert result.shape == (2, 12)
    
    # Test with named dimensions
    result = rearrange_pattern(x, 'b h w -> (b h) w', b=2, h=3)
    assert result.shape == (6, 4)
    
    # Test complex rearrangement
    result = rearrange_pattern(x, 'b h w -> w (b h)')
    assert result.shape == (4, 6)

def test_reduce_pattern():
    x = np.random.rand(2, 3, 4)
    
    # Test mean reduction
    result = reduce_pattern(x, 'b h w -> b h', reduction='mean')
    assert result.shape == (2, 3)
    
    # Test sum reduction
    result = reduce_pattern(x, 'b h w -> b', reduction='sum')
    assert result.shape == (2,)
    
    # Test max reduction
    result = reduce_pattern(x, 'b h w -> w', reduction='max')
    assert result.shape == (4,)
    
    # Test invalid reduction
    with pytest.raises(ValueError):
        reduce_pattern(x, 'b h w -> b', reduction='invalid')

def test_repeat_pattern():
    x = np.array([[1, 2], [3, 4]])  # 2x2
    
    # Test basic repeat
    result = repeat_pattern(x, 'h w -> h w c', c=3)
    assert result.shape == (2, 2, 3)
    
    # Test repeat with rearrangement
    result = repeat_pattern(x, 'h w -> w h c', c=2)
    assert result.shape == (2, 2, 2)
    
    # Test missing dimension
    with pytest.raises(ValueError):
        repeat_pattern(x, 'h w -> h w c')

def test_pattern_compiler():
    compiler = PatternCompiler()
    
    # Test pattern caching
    pattern = 'b h w -> (b h) w'
    compiled1 = compiler.compile_pattern(pattern)
    compiled2 = compiler.compile_pattern(pattern)
    assert compiled1 is compiled2
    
    # Test compiled pattern structure
    assert 'source' in compiled1
    assert 'target' in compiled1
    assert compiled1['source'] == ['b', 'h', 'w']
    assert compiled1['target'] == ['(b h)', 'w']