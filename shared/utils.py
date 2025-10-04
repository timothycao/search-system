"""
Shared helper functions.
"""
from typing import List
from re import sub

def tokenize(text: str) -> List[str]:
    """
    Tokenize text into normalized terms.
    - Lowercase
    - Remove non-alphanumeric characters
    """
    text = text.lower()
    text = sub(r'[^a-z0-9]', ' ', text) # replace non-alphanumeric with space
    tokens = text.split()
    return tokens