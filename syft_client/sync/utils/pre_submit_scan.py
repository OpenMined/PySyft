from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

_RESOLVER_NAMES = {"resolve_dataset_files_path", "resolve_dataset_file_path"}
_CLIENT_KWARG_FALLBACK_RE = re.compile(
    r"resolve_dataset_files?_path\s*\([^)]*\bclient\s*=", re.DOTALL
)
_DATASET_FILES_ATTR_RE = re.compile(r"\.(mock_files|private_files)\b")


def run_pre_submit_check(code_path: Path) -> bool:
    """Scan job code for known anti-patterns. Return True if submission should proceed."""
    py_files = _collect_py_files(code_path)

    found_client_kwarg = any(_has_client_kwarg(f) for f in py_files)
    found_dataset_attr = any(_has_dataset_files_attr(f) for f in py_files)

    if found_client_kwarg:
        _print_client_kwarg_warning()
        return _confirm_continue()

    if found_dataset_attr:
        _print_dataset_files_warning()
        return _confirm_continue()

    print("✅ Pre-submit check passed.")
    return True


def _collect_py_files(code_path: Path) -> list[Path]:
    p = Path(code_path)
    if not p.exists():
        return []
    if p.is_file():
        return [p] if p.suffix == ".py" else []
    return sorted(p.rglob("*.py"))


def _read(path: Path) -> str | None:
    try:
        return path.read_text()
    except Exception:
        return None


def find_client_kwarg_in_ast(tree: ast.AST) -> bool:
    """Return True if the AST contains a resolver call with a `client=` kwarg."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_name: str | None = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        if func_name not in _RESOLVER_NAMES:
            continue
        for kw in node.keywords:
            if kw.arg == "client":
                return True
    return False


def _has_client_kwarg(path: Path) -> bool:
    content = _read(path)
    if content is None:
        return False
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return bool(_CLIENT_KWARG_FALLBACK_RE.search(content))
    return find_client_kwarg_in_ast(tree)


def _has_dataset_files_attr(path: Path) -> bool:
    content = _read(path)
    if content is None:
        return False
    return bool(_DATASET_FILES_ATTR_RE.search(content))


def _print_client_kwarg_warning() -> None:
    print(
        "🚨 Pre-submit check failed:\n\n"
        "   Your script contains 'client=client' which will fail on the DO's machine.\n"
        "   'client' is not defined there — it is only available in your local Colab session.\n\n"
        "   Fix: remove 'client=client' from resolve_dataset_files_path() and re-run %%writefile.\n\n"
        "   ✅ Correct:  sc.resolve_dataset_files_path('ocean_water_quality')\n"
        "   ❌ Wrong:    sc.resolve_dataset_files_path('ocean_water_quality', client=client)"
    )


def _print_dataset_files_warning() -> None:
    print(
        "🚨 Pre-submit check failed:\n\n"
        "   Your script references '.mock_files' or '.private_files' on a Dataset object,\n"
        "   which won't exist on the DO's machine — those attributes belong to a Dataset\n"
        "   you constructed locally.\n\n"
        "   Fix: replace dataset.mock_files / dataset.private_files with\n"
        "        sc.resolve_dataset_files_path('<dataset_name>') and re-run %%writefile.\n\n"
        "   ✅ Correct:  sc.resolve_dataset_files_path('ocean_water_quality')\n"
        "   ❌ Wrong:    dataset.mock_files  /  dataset.private_files"
    )


def _is_interactive() -> bool:
    """Check if we're in an interactive environment (terminal or Jupyter)."""
    try:
        get_ipython()  # type: ignore[name-defined]
        return True
    except NameError:
        return sys.stdin.isatty()


def _confirm_continue() -> bool:
    if not _is_interactive():
        print("(non-interactive environment; continuing without prompt)")
        return True
    response = input("\nContinue with submission anyway? (y/N): ")
    return response.strip().lower() in ("y", "yes")
