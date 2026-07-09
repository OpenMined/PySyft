"""Tests for the display transform (approach A), exercised through run()."""

import ast
import shutil
from pathlib import Path

from syft_restrict import obfuscate, run
from syft_restrict.astutil import scan_file

FIXTURES = Path(__file__).parents[1] / "fixtures"
ALLOW_FUNCTIONS = ["jax.*", "flax.linen.*"]
ALLOW_METHODS = ["arithmetic", "indexing", "comparison"]


def _private_from_config(source: str):
    config_line = next(
        i for i, ln in enumerate(source.splitlines(), 1) if ln.startswith("CONFIG")
    )
    return [[config_line, len(source.splitlines())]], config_line


def _obfuscate_fixture(tmp_path: Path):
    src_path = tmp_path / "model.py"
    shutil.copy(FIXTURES / "compliant_model.py", src_path)
    source = src_path.read_text()
    private, config_line = _private_from_config(source)
    result = run(
        src_path,
        obfuscate=private,
        allow_functions=ALLOW_FUNCTIONS,
        allow_methods=ALLOW_METHODS,
    )
    obf = Path(result.obfuscated_path).read_text()
    return source, obf, config_line


def test_nonprivate_lines_are_byte_for_byte(tmp_path):
    source, obf, config_line = _obfuscate_fixture(tmp_path)
    src_lines = source.splitlines()
    obf_lines = obf.splitlines()
    assert len(src_lines) == len(obf_lines)
    for i in range(config_line - 1):  # lines before CONFIG are non-private
        assert src_lines[i] == obf_lines[i], f"line {i + 1} changed"
    # the import lines specifically
    assert "import jax.numpy as jnp" in obf


def test_private_region_is_mangled_and_blanked(tmp_path):
    source, obf, config_line = _obfuscate_fixture(tmp_path)
    private_text = "\n".join(obf.splitlines()[config_line - 1 :])
    assert "░" in private_text  # identifiers renamed
    assert "RMSNorm" not in private_text  # a private class name is gone
    assert "■" in private_text  # numeric/string constants blanked
    assert "dim=8" not in private_text  # the architecture dim is hidden
    # public library names stay readable
    assert "jnp" in private_text and "nn.Module" in obf


def test_hide_blanks_whole_body_keeping_indentation(tmp_path):
    """Obfuscate a function's def line but HIDE its body: the body becomes an indented marker."""
    src_path = tmp_path / "model.py"
    shutil.copy(FIXTURES / "compliant_model.py", src_path)
    lines = src_path.read_text().splitlines()
    def_line = next(
        i
        for i, ln in enumerate(lines, 1)
        if ln.lstrip().startswith("def scale_pattern")
    )
    body = [def_line + 1, def_line + 2]  # the two body lines (4-space indented)
    result = run(
        src_path,
        obfuscate=[[def_line, def_line]],
        hide=[body],
        allow_functions=ALLOW_FUNCTIONS,
        allow_methods=ALLOW_METHODS,
    )
    obf_lines = Path(result.obfuscated_path).read_text().splitlines()
    note = "# hidden/obfuscated lines can only execute restricted python"
    # the signature line survives (not hidden); the body is blanked with indentation kept
    assert obf_lines[def_line - 1].startswith("def ")
    # the block's FIRST hidden line carries the note; the rest are the bare marker
    assert (
        obf_lines[body[0] - 1].startswith("    ■■■■■■■■  #")
        and note in obf_lines[body[0] - 1]
    )
    for ln_no in range(body[0] + 1, body[1] + 1):
        assert obf_lines[ln_no - 1] == "    ■■■■■■■■", repr(obf_lines[ln_no - 1])
    hidden_text = "\n".join(obf_lines[body[0] - 1 : body[1]])
    assert "base" not in hidden_text  # the real body tokens are gone
    assert result.certificate["hide_ranges"] == [body]


def test_fstring_literal_text_is_blanked():
    # obfuscate() alone must still blank f-string text, even though verify()
    # bans f-strings now.
    source = 'def f():\n    return f"secret literal text"\n'
    scan = scan_file(ast.parse(source), [(1, 2)])
    obf = obfuscate(source, [[1, 2]], [], scan)
    assert "secret literal text" not in obf
    assert "■" in obf


def test_obfuscation_is_deterministic(tmp_path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    _, obf1, _ = _obfuscate_fixture(dir_a)
    _, obf2, _ = _obfuscate_fixture(dir_b)
    assert obf1 == obf2
