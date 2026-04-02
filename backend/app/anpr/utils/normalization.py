"""Normalization utilities for plate text matching"""


def normalize_plate_for_matching(plate_text):
    """
    Normalize plate text for matching and comparison.
    
    Handles:
    - Case normalization
    - Whitespace trimming
    - Common character substitutions (O vs 0, etc)
    """
    if not plate_text:
        return ""
    
    # Convert to uppercase
    normalized = plate_text.upper().strip()
    
    # Replace common confusions
    normalized = normalized.replace('O', '0')  # Letter O to number 0
    normalized = normalized.replace('I', '1')  # Letter I to number 1
    
    return normalized
