import ast
import hashlib
import warnings

from src.data.constants import MAX_CHARS, MIN_CHARS


def is_valid_syntax(source: str) -> bool:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            ast.parse(source)
        except SyntaxError:
            return False
    return len(caught) == 0


def passes_char_length(source: str, min_chars: int = MIN_CHARS, max_chars: int = MAX_CHARS) -> bool:
    return min_chars <= len(source) <= max_chars


def sha256(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()
