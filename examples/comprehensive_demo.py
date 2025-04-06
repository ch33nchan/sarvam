import numpy as np
from srini_einops import rearrange, reduce, repeat
import time
import os

def clear_screen():
    """Clear the terminal screen."""
    os.system('clear')  # For MacOS/Linux

def print_section(title, color=""):
    """Print a section title with color."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "purple": "\033[95m",
        "cyan": "\033[96m",
        "end": "\033[0m"
    }
    
    selected_color = colors.get(color, colors["cyan"])
    end_color = colors["end"]
    
    print(f"\n{selected_color}{'='*20} {title} {'='*20}{end_color}")

def print_array(arr, label="Array"):
    """Pretty print a numpy array with a label."""
    print(f"\n\033[1m{label}:\033[0m")
    
    if arr.ndim <= 2:
        print(arr)
    else:
        print(f"Shape: {arr.shape}")
        print(f"First slice:\n{arr[0]}")
    
    print(f"Data type: {arr.dtype}")

def animate_text(text):
    """Animate text typing effect."""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(0.01)
    print()

def demo_transpose():
    print_section("Transpose Operation", "green")
    animate_text("Transposing swaps the axes of a tensor.")
    
    x = np.array([[1, 2, 3], [4, 5, 6]])
    print_array(x, "Input (2×3 matrix)")
    
    result = rearrange(x, 'h w -> w h')
    print_array(result, "Output after 'h w -> w h' (3×2 matrix)")
    
    print("\nExplanation: The pattern 'h w -> w h' swaps the height and width dimensions.")
    input("\nPress Enter to continue...")

def demo_split_dimensions():
    print_section("Split Dimensions", "green")
    animate_text("Splitting breaks a dimension into multiple dimensions.")
    
    x = np.arange(24).reshape(6, 4)
    print_array(x, "Input (6×4 matrix)")
    
    result = rearrange(x, '(h w) c -> h w c', h=2)
    print_array(result, "Output after '(h w) c -> h w c' with h=2")
    
    print("\nExplanation: The pattern splits the first dimension (6) into two dimensions (2×3).")
    print("The first dimension is interpreted as (h×w) where h=2, so w=3.")
    
    # Another example
    x = np.arange(16).reshape(4, 4)
    print_array(x, "Another example - Input (4×4 matrix)")
    
    result = rearrange(x, '(h w) c -> h w c', h=2, w=2)
    print_array(result, "Output after '(h w) c -> h w c' with h=2, w=2")
    
    input("\nPress Enter to continue...")

def demo_merge_dimensions():
    print_section("Merge Dimensions", "green")
    animate_text("Merging combines multiple dimensions into one.")
    
    x = np.arange(24).reshape(2, 3, 4)
    print_array(x, "Input (2×3×4 tensor)")
    
    result = rearrange(x, 'b h w -> b (h w)')
    print_array(result, "Output after 'b h w -> b (h w)'")
    
    print("\nExplanation: The pattern merges the 'h' and 'w' dimensions (3×4) into a single dimension (12).")
    
    # Another example
    result = rearrange(x, 'b h w -> (b h) w')
    print_array(result, "Output after 'b h w -> (b h) w'")
    
    print("\nExplanation: This pattern merges the 'b' and 'h' dimensions (2×3) into a single dimension (6).")
    
    input("\nPress Enter to continue...")

def demo_reduction():
    print_section("Reduction Operations", "yellow")
    animate_text("Reduction operations collapse dimensions by applying a function.")
    
    x = np.random.rand(2, 3, 4)
    print_array(x, "Input (2×3×4 tensor with random values)")
    
    result = reduce(x, 'b h w -> b h', reduction='mean')
    print_array(result, "Mean reduction: 'b h w -> b h'")
    print("Explanation: Computes the mean along the 'w' dimension.")
    
    result = reduce(x, 'b h w -> b', reduction='max')
    print_array(result, "Max reduction: 'b h w -> b'")
    print("Explanation: Finds the maximum value across both 'h' and 'w' dimensions.")
    
    result = reduce(x, 'b h w -> w', reduction='sum')
    print_array(result, "Sum reduction: 'b h w -> w'")
    print("Explanation: Sums values across both 'b' and 'h' dimensions.")
    
    # Another example with min reduction
    result = reduce(x, 'b h w -> h w', reduction='min')
    print_array(result, "Min reduction: 'b h w -> h w'")
    print("Explanation: Finds the minimum value across the 'b' dimension.")
    
    input("\nPress Enter to continue...")

def demo_repeat():
    print_section("Repeat Operations", "purple")
    animate_text("Repeat operations duplicate data along specified dimensions.")
    
    x = np.array([[1, 2], [3, 4]])
    print_array(x, "Input (2×2 matrix)")
    
    result = repeat(x, 'h w -> h w c', c=3)
    print_array(result, "Output after 'h w -> h w c' with c=3")
    print("Explanation: Repeats the data along a new 'c' dimension 3 times.")
    
    result = repeat(x, 'h w -> w h c', c=2)
    print_array(result, "Output after 'h w -> w h c' with c=2")
    print("Explanation: Transposes 'h' and 'w', then repeats along a new 'c' dimension 2 times.")
    
    # Another example
    result = repeat(x, 'h w -> (h repeat) w', repeat=3)
    print_array(result, "Output after 'h w -> (h repeat) w' with repeat=3")
    print("Explanation: Repeats each row 3 times, expanding the height dimension.")
    
    input("\nPress Enter to continue...")

def demo_ellipsis():
    print_section("Ellipsis Notation", "blue")
    animate_text("Ellipsis (...) represents any number of dimensions.")
    
    x = np.random.rand(2, 3, 4, 5)
    print_array(x, "Input (2×3×4×5 tensor)")
    
    result = rearrange(x, 'b ... -> b (...)')
    print_array(result, "Output after 'b ... -> b (...)'")
    print("Explanation: Preserves the first dimension 'b' and flattens all other dimensions.")
    
    result = rearrange(x, 'b c ... w -> b ... c w')
    print_array(result, "Output after 'b c ... w -> b ... c w'")
    print("Explanation: Moves the 'c' dimension to before the last dimension 'w'.")
    
    # Another example
    x = np.random.rand(2, 3, 4, 5, 6)
    print_array(x, "Another example - Input (2×3×4×5×6 tensor)")
    
    result = rearrange(x, 'a b ... y z -> a ... b y z')
    print_array(result, "Output after 'a b ... y z -> a ... b y z'")
    print("Explanation: Moves the 'b' dimension to after the ellipsis.")
    
    input("\nPress Enter to continue...")

def demo_complex_patterns():
    print_section("Complex Patterns", "red")
    animate_text("Combining multiple operations for advanced transformations.")
    
    # Example 1: Image patch extraction
    img = np.arange(64).reshape(8, 8)
    print_array(img, "Input image (8×8)")
    
    patches = rearrange(img, '(h ph) (w pw) -> (h w) ph pw', ph=2, pw=2)
    print_array(patches, "Image patches after '(h ph) (w pw) -> (h w) ph pw' with ph=pw=2")
    print("Explanation: Extracts 2×2 patches from the image, resulting in 16 patches.")
    
    # Example 2: Batch processing with multiple operations
    x = np.random.rand(2, 3, 4, 5)
    print_array(x, "Input (2×3×4×5 tensor)")
    
    # First reduce, then rearrange
    result = reduce(x, 'b c h w -> b h w', reduction='mean')
    result = rearrange(result, 'b h w -> b (h w)')
    print_array(result, "After channel reduction and flattening spatial dimensions")
    print("Explanation: First averages across channels, then flattens height and width.")
    
    # Example 3: Complex reshape for attention mechanism
    x = np.random.rand(2, 8, 16)  # Batch, sequence length, features
    print_array(x, "Input for attention (2×8×16 tensor)")
    
    # Split heads
    result = rearrange(x, 'b s (h d) -> b h s d', h=4)
    print_array(result, "After splitting into attention heads: 'b s (h d) -> b h s d' with h=4")
    print("Explanation: Reshapes features into multiple attention heads, common in transformers.")
    
    input("\nPress Enter to continue...")

def main_menu():
    while True:
        clear_screen()
        print_section("Srini Einops Demo", "cyan")
        animate_text("Welcome to the Srini Einops interactive demo!")
        print("\nThis demo showcases various tensor manipulation operations.")
        print("\nPlease select an option:")
        print("1. Transpose Operations")
        print("2. Split Dimensions")
        print("3. Merge Dimensions")
        print("4. Reduction Operations")
        print("5. Repeat Operations")
        print("6. Ellipsis Notation")
        print("7. Complex Patterns")
        print("8. Run All Demos")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-8): ")
        
        if choice == '0':
            clear_screen()
            print("Thank you for exploring Srini Einops!")
            break
        elif choice == '1':
            clear_screen()
            demo_transpose()
        elif choice == '2':
            clear_screen()
            demo_split_dimensions()
        elif choice == '3':
            clear_screen()
            demo_merge_dimensions()
        elif choice == '4':
            clear_screen()
            demo_reduction()
        elif choice == '5':
            clear_screen()
            demo_repeat()
        elif choice == '6':
            clear_screen()
            demo_ellipsis()
        elif choice == '7':
            clear_screen()
            demo_complex_patterns()
        elif choice == '8':
            clear_screen()
            demo_transpose()
            demo_split_dimensions()
            demo_merge_dimensions()
            demo_reduction()
            demo_repeat()
            demo_ellipsis()
            demo_complex_patterns()
            print_section("Demo Complete", "green")
            input("Press Enter to return to the main menu...")
        else:
            print("Invalid choice. Please try again.")
            time.sleep(1)

if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)
    main_menu()