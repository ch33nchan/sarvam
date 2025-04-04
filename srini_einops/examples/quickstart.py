import numpy as np
from srini_einops import rearrange, reduce, repeat

# Example 1: Basic Rearrangement
print("\n=== Example 1: Basic Rearrangement ===")
x = np.array([[1, 2, 3],
              [4, 5, 6]])
print("Original array:")
print(x)
print("\nTransposed (h w -> w h):")
print(rearrange(x, 'h w -> w h'))

# Example 2: Reduction
print("\n=== Example 2: Reduction ===")
x = np.random.rand(2, 3, 4)
print("Original shape:", x.shape)
print("Mean over last dimension (b h w -> b h):")
result = reduce(x, 'b h w -> b h', reduction='mean')
print("Result shape:", result.shape)
print(result)

# Example 3: Repeat
print("\n=== Example 3: Repeat ===")
x = np.array([[1, 2],
              [3, 4]])
print("Original array:")
print(x)
print("\nRepeated with new channel dimension (h w -> h w c):")
result = repeat(x, 'h w -> h w c', c=3)
print("Result shape:", result.shape)
print(result)