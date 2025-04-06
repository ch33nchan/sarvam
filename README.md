# Srini Einops

A lightweight tensor manipulation library inspired by einops, providing intuitive operations for reshaping, transposing, and reducing tensors with a clean pattern-based syntax.

## Project Structure

The project is organized as follows:

```
srini_einops/
├── srini_einops/         # Main package
│   ├── __init__.py       # Package initialization
│   └── core.py           # Core implementation of operations
├── examples/             # Example notebooks and scripts
│   └── srini_einops_demo.ipynb  # Demo notebook
├── tests/                # Test suite
│   ├── test_basic_ops.py # Basic operations tests
│   ├── test_advanced_ops.py # Advanced operations tests
│   └── ...               # Other test files
└── README.md             # This file
```

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/srini_einops.git
cd srini_einops
```

Install the package in development mode:

```bash
pip install -e .
```

## Usage

Srini Einops provides three main operations:

1. `rearrange` - Reshape and transpose tensors
2. `reduce` - Reduce dimensions with operations like mean, sum, max, min
3. `repeat` - Repeat tensor along new dimensions

### Basic Examples

```python
import numpy as np
from srini_einops import rearrange, reduce, repeat

# Transpose a matrix
x = np.array([[1, 2, 3], [4, 5, 6]])
result = rearrange(x, 'h w -> w h')
# result: [[1, 4], [2, 5], [3, 6]]

# Reshape a tensor
x = np.arange(24).reshape(2, 3, 4)
result = rearrange(x, 'b h w -> b (h w)')
# result: shape (2, 12)

# Reduce dimensions
x = np.random.rand(2, 3, 4)
result = reduce(x, 'b h w -> b h', reduction='mean')
# result: shape (2, 3)

# Repeat dimensions
x = np.array([[1, 2], [3, 4]])
result = repeat(x, 'h w -> h w c', c=3)
# result: shape (2, 2, 3)
```

## Running the Demo Notebook

The demo notebook provides a comprehensive overview of the library's capabilities:

```bash
cd examples
jupyter notebook srini_einops_demo.ipynb
```

The notebook includes:
- Basic operations (transpose, reshape, split, merge)
- Advanced operations (reduction, repeat)
- Performance benchmarks

## Running Tests

Run the test suite to verify the implementation:

```bash
python -m pytest tests/ -v
```

## Implementation Approach

### Design Decisions

1. **Pattern-based syntax**: The library uses a simple pattern-based syntax (e.g., 'b h w -> b (h w)') to describe tensor operations, making complex reshaping operations more intuitive.

2. **NumPy backend**: The current implementation uses NumPy as the backend for tensor operations, focusing on simplicity and readability.

3. **Minimal dependencies**: The library has minimal dependencies, making it easy to integrate into existing projects.

### Performance

The library is designed for clarity and ease of use, with performance comparable to direct NumPy operations:

- **Transpose operations**: Typically 1-2x slower than direct NumPy transpose
- **Reshape operations**: Similar performance to NumPy reshape
- **Reduction operations**: 1-3x slower than direct NumPy reductions

The performance difference is due to the additional pattern parsing and validation that provides the improved usability.

## Future Improvements

Potential areas for enhancement:

1. Optimizing pattern parsing for better performance
2. Adding support for more backends (PyTorch, TensorFlow)
3. Implementing more advanced operations
4. Adding type hints and improved documentation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
```