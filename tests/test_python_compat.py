"""Static guards against Python-version bugs that import cleanly and then
fail at runtime.

This file exists because that exact failure reached a running server twice
in this project:

  1. `zip(..., strict=False)` - added by a lint autofix targeting a newer
     Python than the deployment ran. Every KB lookup raised
     "zip() takes no keyword arguments", the error was handed back to the
     model as a tool result, and the model reported it "couldn't find
     anything" - a hard failure wearing the costume of a plausible answer.
  2. `axes: list | None` on a pydantic model. PEP 604 unions are fine as
     deferred annotations, but pydantic resolves model annotations eagerly
     at class creation, so `from __future__ import annotations` does NOT
     save you. The server crash-looped.

Both are invisible to a test suite running on a newer interpreter, which is
why these are source-level checks rather than behavioural ones. CI also
runs the whole suite on the floor version; this catches it faster and says
why.
"""
from __future__ import annotations

import ast
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
SKIP_DIRS = {".venv", "venv", "_to_delete", "__pycache__", ".git", "build", "dist"}


def _python_files():
    for path in PROJECT_ROOT.rglob("*.py"):
        if not any(part in SKIP_DIRS for part in path.parts):
            yield path


def _uses_pep604(node: ast.AST) -> bool:
    """True if the annotation expression contains an `X | Y` union."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.BitOr):
            return True
    return False


def test_no_pep604_unions_in_pydantic_models():
    """Pydantic evaluates model annotations eagerly, so `X | None` raises
    TypeError on Python 3.9 no matter what __future__ import is in force."""
    offenders = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            base_names = {
                b.id if isinstance(b, ast.Name) else getattr(b, "attr", "")
                for b in node.bases
            }
            if not base_names & {"BaseModel", "Settings", "BaseSettings"}:
                continue
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and _uses_pep604(stmt.annotation):
                    offenders.append(
                        f"{path.relative_to(PROJECT_ROOT)}:{stmt.lineno} "
                        f"in {node.name} - use Optional[...] / Union[...]"
                    )
    assert not offenders, "PEP 604 union in an eagerly-evaluated model:\n" + "\n".join(offenders)


def test_no_zip_strict_keyword():
    """`zip(strict=)` is Python 3.10+. It imports fine and raises at the
    call site, which is the worst possible failure shape."""
    offenders = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "zip"
                    and any(kw.arg == "strict" for kw in node.keywords)):
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}")
    assert not offenders, "zip(strict=...) is 3.10+ only:\n" + "\n".join(offenders)


def test_every_module_parses_as_python_39():
    """Catches 3.10+ *syntax* (match statements, PEP 604 in runtime
    positions) across the whole project in one pass."""
    failures = []
    for path in _python_files():
        try:
            ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 9))
        except SyntaxError as e:
            failures.append(f"{path.relative_to(PROJECT_ROOT)}: {e}")
    assert not failures, "not valid Python 3.9 syntax:\n" + "\n".join(failures)
