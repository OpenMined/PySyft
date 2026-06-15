"""Tests for the pre-submit script scanner."""

import ast

import pytest

from syft_client.sync.utils.pre_submit_scan import (
    find_client_kwarg_in_ast,
    run_pre_submit_check,
)


# --- AST walker tested in isolation ---


@pytest.mark.parametrize(
    "src",
    [
        'sc.resolve_dataset_files_path("foo", client=c)',
        'resolve_dataset_files_path("foo", client=c)',
        'resolve_dataset_file_path("foo", client=c)',
        'sc.resolve_dataset_file_path("foo", owner_email="x", client=c)',
        # Call buried inside other expressions still detected
        'x = [sc.resolve_dataset_files_path("foo", client=c) for _ in range(1)]',
    ],
)
def test_find_client_kwarg_in_ast_positive(src):
    tree = ast.parse(src)
    assert find_client_kwarg_in_ast(tree) is True


@pytest.mark.parametrize(
    "src",
    [
        'resolve_dataset_files_path("foo", owner_email="x")',
        'resolve_dataset_files_path("foo")',
        'sc.resolve_dataset_files_path("foo")',
        # Function definition, not a Call
        "def foo(client=client): pass",
        # Plain assignment
        "client = client",
        # Unrelated function with the same kwarg
        'some_other_func("foo", client=c)',
    ],
)
def test_find_client_kwarg_in_ast_negative(src):
    tree = ast.parse(src)
    assert find_client_kwarg_in_ast(tree) is False


# --- Full scanner via run_pre_submit_check ---


def _write_script(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content)
    return path


def test_run_pre_submit_check_clean_script(tmp_path, capsys):
    script = _write_script(
        tmp_path,
        "main.py",
        'import syft_client as sc\nsc.resolve_dataset_files_path("foo")\n',
    )
    assert run_pre_submit_check(script) is True
    out = capsys.readouterr().out
    assert "✅ Pre-submit check passed." in out


def test_run_pre_submit_check_client_kwarg(tmp_path, capsys):
    script = _write_script(
        tmp_path,
        "main.py",
        'import syft_client as sc\nsc.resolve_dataset_files_path("ds", client=client)\n',
    )
    # Non-TTY in pytest → continues without prompt
    assert run_pre_submit_check(script) is True
    out = capsys.readouterr().out
    assert "🚨 Pre-submit check failed" in out
    assert "client=client" in out
    assert "non-interactive" in out


def test_run_pre_submit_check_dataset_files_attr(tmp_path, capsys):
    script = _write_script(
        tmp_path,
        "main.py",
        "data = dataset.mock_files\n",
    )
    assert run_pre_submit_check(script) is True
    out = capsys.readouterr().out
    assert "🚨 Pre-submit check failed" in out
    assert ".mock_files" in out


def test_run_pre_submit_check_private_files_attr(tmp_path, capsys):
    script = _write_script(
        tmp_path,
        "main.py",
        "data = dataset.private_files\n",
    )
    assert run_pre_submit_check(script) is True
    out = capsys.readouterr().out
    assert "🚨 Pre-submit check failed" in out
    assert ".mock_files" in out  # warning copy lists both


def test_client_kwarg_takes_priority_over_dataset_files(tmp_path, capsys):
    _write_script(
        tmp_path,
        "a.py",
        'sc.resolve_dataset_files_path("ds", client=client)\n',
    )
    _write_script(
        tmp_path,
        "b.py",
        "data = dataset.mock_files\n",
    )
    assert run_pre_submit_check(tmp_path) is True
    out = capsys.readouterr().out
    assert "client=client" in out
    # The dataset-files-specific warning copy should NOT appear when client= fires
    assert "Your script references" not in out


def test_syntax_error_falls_back_to_regex(tmp_path, capsys):
    # Trailing unmatched paren makes ast.parse raise SyntaxError
    script = _write_script(
        tmp_path,
        "broken.py",
        'sc.resolve_dataset_files_path("ds", client=client\n# unterminated\n',
    )
    assert run_pre_submit_check(script) is True
    out = capsys.readouterr().out
    assert "🚨 Pre-submit check failed" in out
    assert "client=client" in out


def test_folder_walks_all_py_files(tmp_path, capsys):
    sub = tmp_path / "sub"
    sub.mkdir()
    _write_script(tmp_path, "main.py", "x = 1\n")
    _write_script(sub, "helper.py", "data = obj.mock_files\n")
    assert run_pre_submit_check(tmp_path) is True
    out = capsys.readouterr().out
    assert "🚨 Pre-submit check failed" in out


def test_nonexistent_path_passes_silently(tmp_path, capsys):
    # No .py files → nothing to warn about → passes
    missing = tmp_path / "does_not_exist"
    assert run_pre_submit_check(missing) is True
    out = capsys.readouterr().out
    assert "✅ Pre-submit check passed." in out
