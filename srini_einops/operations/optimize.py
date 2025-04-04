import numpy as np
from typing import Dict, List, Tuple
from numba import jit
from scipy import sparse

@jit(nopython=True)
def optimize_reshape(tensor: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """JIT-optimized reshape operation"""
    return tensor.reshape(shape)

def handle_sparse(tensor: np.ndarray, operation: str, **kwargs) -> np.ndarray:
    """Handle sparse tensor operations"""
    if sparse.issparse(tensor):
        if operation == 'reshape':
            # Convert to dense for 1D reshape
            if len(kwargs['shape']) == 1:
                return tensor.toarray().reshape(kwargs['shape'])
            return tensor.reshape(kwargs['shape'])
        elif operation == 'transpose':
            # Sparse matrices only support simple transpose without axes
            return tensor.transpose()
    return tensor

class PatternCompiler:
    """Precompile patterns for reuse"""
    def __init__(self):
        self.cache = {}
    
    def compile_pattern(self, pattern: str) -> Dict:
        if pattern not in self.cache:
            source, target = pattern.split('->')
            source_dims = source.strip().split()
            
            # Parse target pattern properly
            target_parts = target.strip().split()
            parsed_target = []
            current_group = []
            
            for part in target_parts:
                if '(' in part and ')' in part:
                    # Single group like '(b h)'
                    parsed_target.append(part)
                elif '(' in part:
                    current_group = [part.strip('(')]
                elif ')' in part:
                    current_group.append(part.strip(')'))
                    parsed_target.append('(' + ' '.join(current_group) + ')')
                    current_group = []
                else:
                    if current_group:
                        current_group.append(part)
                    else:
                        parsed_target.append(part)
            
            self.cache[pattern] = {
                'source': source_dims,
                'target': parsed_target
            }
        return self.cache[pattern]