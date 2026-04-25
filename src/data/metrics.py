import ast
import textwrap
from dataclasses import dataclass, field

import radon.complexity
import radon.metrics
import radon.raw


def soft_wrap_fn_process(source: str) -> str:
    tree = ast.parse(source)
    has_fn = any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in tree.body
    )
    if has_fn:
        return source
    indented = textwrap.indent(source, "    ")
    return f"def _module():\n{indented}\n"


@dataclass
class CodeMetrics:
    source: str
    cc: int = field(init=False)
    mi: float = field(init=False)
    comment_ratio: float = field(init=False)
    h_volume: float = field(init=False)
    h_difficulty: float = field(init=False)
    h_effort: float = field(init=False)
    sloc: int = field(init=False)

    def __post_init__(self):
        wrapped = soft_wrap_fn_process(self.source)

        # cc_visit returns a list of Block objects (functions/classes/methods),
        # each with a .complexity int (McCabe cyclomatic complexity).
        blocks = radon.complexity.cc_visit(wrapped)
        self.cc = int(max((b.complexity for b in blocks), default=1))

        # mi_visit returns a float 0-100 maintainability index.
        # multi=True counts multiline strings as comments.
        self.mi = float(radon.metrics.mi_visit(wrapped, multi=True))

        # analyze returns a Module namedtuple: loc, lloc, sloc, comments, multi, blank.
        # Use original source so wrapper indentation doesn't dilute comment_ratio.
        raw = radon.raw.analyze(self.source)
        self.comment_ratio = float(raw.comments / max(raw.loc + raw.comments, 1))
        self.sloc = int(raw.sloc)

        # h_visit returns a HalsteadReport; .total is a HalsteadMetrics namedtuple
        # with .volume, .difficulty, .effort, etc.
        h = radon.metrics.h_visit(wrapped).total
        self.h_volume = float(h.volume)
        self.h_difficulty = float(h.difficulty)
        self.h_effort = float(h.effort)
