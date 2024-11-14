"""Translate Traditional Chinese to Simplified Chinese"""

import opencc


def trad2simp(text):
    """Translate Traditional Chinese to Simplified Chinese."""
    converter = opencc.OpenCC("t2s")

    converted_text = converter.convert(text)

    return converted_text
