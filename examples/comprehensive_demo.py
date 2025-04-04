import numpy as np
from srini_einops import rearrange, reduce, repeat

def print_section(title):
    print(f"\n{'='*20} {title} {'='*20}")
    
# 1. Basic Operations
print_section("Basic Operations")

# Transpose
x = np.array([[1, 2, 3], [4, 5, 6]])
print("\nTranspose:")
print("Input:", x)
print("Output:", rearrange(x, 'h w -> w h'))

# Split dimensions
x = np.arange(24).reshape(6, 4)
print("\nSplit dimensions:")
print("Input shape:", x.shape)
result = rearrange(x, '(h w) c -> h w c', h=2)
print("Output shape:", result.shape)
print("Output:\n", result)

# 2. Reduction Operations
print_section("Reduction Operations")

x = np.random.rand(2, 3, 4)
print("\nInput shape:", x.shape)

print("\nMean reduction:")
print(reduce(x, 'b h w -> b h', reduction='mean'))

print("\nMax reduction:")
print(reduce(x, 'b h w -> b', reduction='max'))

print("\nSum reduction:")
print(reduce(x, 'b h w -> w', reduction='sum'))

# 3. Repeat Operations
print_section("Repeat Operations")

x = np.array([[1, 2], [3, 4]])
print("\nInput:", x)

print("\nRepeat along new axis:")
result = repeat(x, 'h w -> h w c', c=3)
print("Output shape:", result.shape)
print(result)

print("\nRepeat with rearrangement:")
result = repeat(x, 'h w -> w h c', c=2)
print("Output shape:", result.shape)
print(result)

# 4. Complex Patterns
print_section("Complex Patterns")

# Combine split and merge
x = np.arange(24).reshape(2, 3, 4)
print("\nInput shape:", x.shape)
result = rearrange(x, 'b h w -> b (h w)')
print("Merge dimensions - Output shape:", result.shape)
print(result)

# Using ellipsis
x = np.random.rand(2, 3, 4, 5)
print("\nInput shape:", x.shape)
print("\nEllipsis examples:")
# Simple ellipsis
result = rearrange(x, 'b ... -> b (...)')
print("Flatten with batch - Output shape:", result.shape)

# Preserve middle dimensions
result = rearrange(x, 'b c ... w -> b ... c w')
print("Move channel to end - Output shape:", result.shape)

# 5. Chained Operations
print_section("Chained Operations")

x = np.random.rand(2, 3, 4, 5)
print("\nInput shape:", x.shape)

# First reduce, then rearrange
result = reduce(x, 'b c h w -> b h w', reduction='mean')
result = rearrange(result, 'b h w -> b (h w)')
print("Reduce then rearrange - Output shape:", result.shape)

if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)