import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

@dataclass
class ParsedPattern:
    """Represents a parsed einops pattern"""
    input_axes: List[str]
    output_axes: List[str]
    composite_axes: Dict[str, List[str]]
    has_ellipsis: bool
    ellipsis_position_in: Optional[int]
    ellipsis_position_out: Optional[int]

class ParsingError(Exception):
    """Custom exception for pattern parsing errors"""
    pass

def parse_pattern(pattern: str) -> ParsedPattern:
    """
    Parse an einops pattern string into its components.
    
    Args:
        pattern: String in format 'input_axes -> output_axes'
        
    Returns:
        ParsedPattern object containing parsed components
    """
    if '->' not in pattern:
        raise ParsingError("Pattern must contain '->' to separate input and output")
    
    input_pattern, output_pattern = pattern.split('->')
    input_pattern = input_pattern.strip()
    output_pattern = output_pattern.strip()
    
    # Handle ellipsis
    has_ellipsis = '...' in input_pattern or '...' in output_pattern
    ellipsis_in = input_pattern.find('...')
    ellipsis_out = output_pattern.find('...')
    
    # Parse composite axes (e.g., '(h w)')
    def extract_composites(pattern: str) -> Tuple[str, Dict[str, List[str]]]:
        composites = {}
        comp_pattern = r'\(([^()]+)\)'
        
        def replace_match(match):
            inner = match.group(1)
            axes = inner.split()
            composite_name = '_'.join(axes)
            composites[composite_name] = axes
            return composite_name
        
        new_pattern = re.sub(comp_pattern, replace_match, pattern)
        return new_pattern, composites
    
    input_pattern, in_composites = extract_composites(input_pattern)
    output_pattern, out_composites = extract_composites(output_pattern)
    
    # Combine composite axes
    composite_axes = {**in_composites, **out_composites}
    
    # Split into individual axes
    input_axes = input_pattern.split()
    output_axes = output_pattern.split()
    
    # Validate axes
    all_axes = set()
    for axes in [input_axes, output_axes]:
        for ax in axes:
            if ax != '...' and ax not in composite_axes:
                all_axes.add(ax)
    
    return ParsedPattern(
        input_axes=input_axes,
        output_axes=output_axes,
        composite_axes=composite_axes,
        has_ellipsis=has_ellipsis,
        ellipsis_position_in=input_axes.index('...') if '...' in input_axes else None,
        ellipsis_position_out=output_axes.index('...') if '...' in output_axes else None
    )