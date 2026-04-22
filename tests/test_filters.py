import pytest

from src.data.constants import MAX_CHARS, MIN_CHARS
from src.data.filters import is_valid_syntax, passes_char_length, sha256


def test_valid_syntax():
    assert is_valid_syntax("x = 1\n") is True


def test_invalid_syntax():
    assert is_valid_syntax("def foo(\n") is False


def test_unicode_valid():
    assert is_valid_syntax("x = '日本語'\n") is True


def test_char_length_at_min():
    source = "x" * MIN_CHARS
    assert passes_char_length(source) is True


def test_char_length_below_min():
    source = "x" * (MIN_CHARS - 1)
    assert passes_char_length(source) is False


def test_char_length_at_max():
    source = "x" * MAX_CHARS
    assert passes_char_length(source) is True


def test_char_length_above_max():
    source = "x" * (MAX_CHARS + 1)
    assert passes_char_length(source) is False


def test_sha256_consistent():
    source = "def foo(): pass\n"
    assert sha256(source) == sha256(source)


def test_sha256_differs_on_different_input():
    assert sha256("a") != sha256("b")


def test_sha256_unicode():
    result = sha256("x = '日本語'\n")
    assert isinstance(result, str) and len(result) == 64
