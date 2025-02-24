"""
This module provides functions for ...
"""

import re
from typing import List, Tuple


def parse_timestamps(text_with_timestamps: str) -> Tuple[List[float], List[str]]:
    """
    Retrieve timestamps and text.

    Args:
        text_with_timestamps (str): The input string containing numbers and timestamp tokens.

    Returns:
        Tuple[List[float], List[str]]: A tuple containing a list of timestamps and a list of text segments.
    """
    pattern = r"<\|\d+\.\d+\|>"
    timestamps = re.findall(pattern, text_with_timestamps)
    timestamps = [float(n[2:-2]) for n in timestamps]
    text_segments = re.split(pattern, text_with_timestamps)
    text_segments = [t for t in text_segments if t.strip()]
    return timestamps, text_segments
