import pytest

from src.data.metrics import CodeMetrics, soft_wrap_fn_process


# --- soft_wrap_fn_process ---

def test_wrap_adds_function_when_none():
    source = "x = 1\ny = 2\n"
    wrapped = soft_wrap_fn_process(source)
    assert wrapped.startswith("def _module():")
    assert "x = 1" in wrapped


def test_no_wrap_when_function_exists():
    source = "def foo():\n    return 1\n"
    assert soft_wrap_fn_process(source) == source


def test_no_wrap_when_async_function_exists():
    source = "async def foo():\n    return 1\n"
    assert soft_wrap_fn_process(source) == source


# --- CodeMetrics ---

SIMPLE_FN = "def foo(x, y):\n    return x + y * 2\n"

BRANCHY_FN = """\
def foo(x):
    if x > 0:
        return x
    elif x < 0:
        return -x
    else:
        return 0
"""

SCRIPT_WITH_COMMENT = """\
# this is a comment
x = 1
y = 2
"""

def test_cc_simple_function():
    m = CodeMetrics(SIMPLE_FN)
    assert m.cc == 1


def test_cc_branchy_function():
    m = CodeMetrics(BRANCHY_FN)
    assert m.cc >= 3


def test_mi_in_range():
    m = CodeMetrics(SIMPLE_FN)
    assert 0 <= m.mi <= 100


def test_sloc_counts_code_lines():
    m = CodeMetrics(SIMPLE_FN)
    assert m.sloc >= 1


def test_comment_ratio_with_comments():
    m = CodeMetrics(SCRIPT_WITH_COMMENT)
    assert m.comment_ratio > 0


def test_comment_ratio_no_comments():
    m = CodeMetrics(SIMPLE_FN)
    assert m.comment_ratio == 0.0


def test_h_volume_positive():
    m = CodeMetrics(SIMPLE_FN)
    assert m.h_volume > 0


def test_h_difficulty_positive():
    m = CodeMetrics(SIMPLE_FN)
    assert m.h_difficulty >= 0


def test_h_effort_positive():
    m = CodeMetrics(SIMPLE_FN)
    assert m.h_effort >= 0


def test_script_without_function_computes_metrics():
    source = "x = 1\ny = x + 2\n"
    m = CodeMetrics(source)
    assert m.sloc >= 1
    assert m.cc >= 1


def test_cc_is_int():
    m = CodeMetrics(SIMPLE_FN)
    assert isinstance(m.cc, int)
