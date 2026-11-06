"""Tests for issue #9586: job stdout/stderr HTML views drop `<...>` text and show no ANSI colors.

Two bugs were present in StdoutViewer._repr_html_ (job_stdout.py) and
StderrViewer._repr_html_ (job_repr.py):

  Bug 1 – Wrong escaping order: ANSI codes were converted to
           ``<span style="color: #cd3131;">`` first, and then ``"`` was
           replaced with ``&quot;``.  Every resulting span became
           ``<span style=&quot;color: #cd3131;&quot;>``, so the browser
           ignored the style and colored output showed up without colours.

  Bug 2 – Missing HTML escaping: ``<`` and ``>`` were never escaped.
           Any output text that looks like an HTML tag (e.g.
           ``<class 'int'>`` or tracebacks containing ``in <module>``)
           was parsed as HTML and disappeared from the view.  Worse,
           output containing ``<script>`` could inject arbitrary markup.

The fix is to escape ``<``, ``>``, ``&`` and ``'`` *before* converting
ANSI sequences to ``<span>`` tags, using ``html.escape(text, quote=False)``
(``quote=False`` preserves ``"`` so the span ``style`` attributes remain
valid HTML attribute values).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Helpers – build minimal stand-ins for JobInfo so we can unit-test the
# viewers without any real syft-job infrastructure.
# ---------------------------------------------------------------------------


def _make_stdout_viewer(tmp_path: Path, content: str):
    """Return a StdoutViewer whose job_info points at *content* as stdout."""
    from syft_job.job_stdout import StdoutViewer  # type: ignore[import]

    job_dir = tmp_path / "job"
    job_dir.mkdir(exist_ok=True)
    (job_dir / "stdout.txt").write_text(content, encoding="utf-8")
    job_info = SimpleNamespace(
        status="done",
        artifact_path=lambda name: job_dir / name,
    )
    return StdoutViewer(job_info)


def _make_stderr_viewer(tmp_path: Path, content: str):
    """Return a StderrViewer whose job_info points at *content* as stderr."""
    from syft_job.job_repr import StderrViewer  # type: ignore[import]

    job_dir = tmp_path / "job"
    job_dir.mkdir(exist_ok=True)
    (job_dir / "stderr.txt").write_text(content, encoding="utf-8")
    job_info = SimpleNamespace(
        status="done",
        artifact_path=lambda name: job_dir / name,
    )
    return StderrViewer(job_info)


# ---------------------------------------------------------------------------
# Bug 2 – HTML injection / disappearing angle-bracket text (StdoutViewer)
# ---------------------------------------------------------------------------


def test_stdout_html_escapes_angle_brackets(tmp_path: Path) -> None:
    """``<class 'int'>`` must appear as ``&lt;class &#x27;int&#x27;&gt;``."""
    viewer = _make_stdout_viewer(tmp_path, "<class 'int'>\n")
    html = viewer._repr_html_()

    # The literal angle-bracket text must not appear unescaped in the output.
    assert "<class" not in html, "unescaped '<class' found – tag would be swallowed"
    # It must appear as properly escaped entities instead.
    assert "&lt;class" in html or "&#x3C;" in html.lower()


def test_stdout_html_escapes_traceback_module_tag(tmp_path: Path) -> None:
    """``in <module>`` from tracebacks must survive HTML rendering."""
    viewer = _make_stdout_viewer(tmp_path, "Traceback:\n  in <module>\n")
    html = viewer._repr_html_()

    assert "<module>" not in html, "unescaped '<module>' leaks into HTML"
    assert "&lt;module&gt;" in html


def test_stdout_html_escapes_script_injection(tmp_path: Path) -> None:
    """Malicious ``<script>`` in job output must not reach the browser."""
    payload = "<script>alert('xss')</script>"
    viewer = _make_stdout_viewer(tmp_path, payload)
    html = viewer._repr_html_()

    assert "<script>" not in html, "XSS script tag escaped into HTML output"


# ---------------------------------------------------------------------------
# Bug 1 – ANSI colour codes produce broken ``style=&quot;...&quot;`` spans (StdoutViewer)
# ---------------------------------------------------------------------------


def test_stdout_html_ansi_red_produces_valid_style_attribute(tmp_path: Path) -> None:
    """Red ANSI text must produce a ``style="color: #cd3131;"`` span."""
    viewer = _make_stdout_viewer(tmp_path, "\x1b[31mfailed\x1b[0m\n")
    html = viewer._repr_html_()

    # The double-quote must NOT be entity-escaped in the style attribute.
    assert 'style="color: #cd3131;"' in html, (
        "ANSI red colour span has broken style attribute; "
        "got: " + html[html.find("span") - 5 : html.find("span") + 80]
    )
    assert "style=&quot;" not in html, "style attribute double-quotes were entity-escaped"


def test_stdout_html_ansi_bold_produces_valid_style_attribute(tmp_path: Path) -> None:
    """Bold ANSI text must produce a ``style="font-weight: bold;"`` span."""
    viewer = _make_stdout_viewer(tmp_path, "\x1b[1mbold text\x1b[0m\n")
    html = viewer._repr_html_()

    assert 'style="font-weight: bold;"' in html
    assert "style=&quot;" not in html


# ---------------------------------------------------------------------------
# Both bugs together (StdoutViewer)
# ---------------------------------------------------------------------------


def test_stdout_html_ansi_color_and_angle_brackets_combined(tmp_path: Path) -> None:
    """Coloured output that also contains ``<...>`` text must render correctly."""
    content = "\x1b[31m<class 'int'>\x1b[0m\n"
    viewer = _make_stdout_viewer(tmp_path, content)
    html = viewer._repr_html_()

    # No unescaped angle bracket from the class repr.
    assert "<class" not in html
    assert "&lt;class" in html or "&#x3C;" in html.lower()
    # Style attribute must use real double-quotes, not entity-escaped ones.
    assert 'style="color: #cd3131;"' in html
    assert "style=&quot;" not in html


# ---------------------------------------------------------------------------
# Bug 2 – HTML injection / disappearing angle-bracket text (StderrViewer)
# ---------------------------------------------------------------------------


def test_stderr_html_escapes_angle_brackets(tmp_path: Path) -> None:
    """``<class 'int'>`` on stderr must appear as escaped entities."""
    viewer = _make_stderr_viewer(tmp_path, "<class 'int'>\n")
    html = viewer._repr_html_()

    assert "<class" not in html, "unescaped '<class' found – tag would be swallowed"
    assert "&lt;class" in html or "&#x3C;" in html.lower()


def test_stderr_html_escapes_traceback_module_tag(tmp_path: Path) -> None:
    """``in <module>`` from tracebacks must survive HTML rendering on stderr."""
    viewer = _make_stderr_viewer(tmp_path, "Traceback:\n  in <module>\n")
    html = viewer._repr_html_()

    assert "<module>" not in html
    assert "&lt;module&gt;" in html


def test_stderr_html_escapes_script_injection(tmp_path: Path) -> None:
    """Malicious ``<script>`` in stderr output must not reach the browser."""
    payload = "<script>alert('xss')</script>"
    viewer = _make_stderr_viewer(tmp_path, payload)
    html = viewer._repr_html_()

    assert "<script>" not in html


# ---------------------------------------------------------------------------
# Bug 1 – ANSI colour codes produce broken spans (StderrViewer)
# ---------------------------------------------------------------------------


def test_stderr_html_ansi_red_produces_valid_style_attribute(tmp_path: Path) -> None:
    """Red ANSI text on stderr must produce a valid ``style="..."`` span."""
    viewer = _make_stderr_viewer(tmp_path, "\x1b[31merror message\x1b[0m\n")
    html = viewer._repr_html_()

    assert 'style="color: #cd3131;"' in html
    assert "style=&quot;" not in html


def test_stderr_html_ansi_bold_produces_valid_style_attribute(tmp_path: Path) -> None:
    """Bold ANSI text on stderr must produce a valid ``style="..."`` span."""
    viewer = _make_stderr_viewer(tmp_path, "\x1b[1mWARNING\x1b[0m\n")
    html = viewer._repr_html_()

    assert 'style="font-weight: bold;"' in html
    assert "style=&quot;" not in html


# ---------------------------------------------------------------------------
# Both bugs together (StderrViewer)
# ---------------------------------------------------------------------------


def test_stderr_html_ansi_color_and_angle_brackets_combined(tmp_path: Path) -> None:
    """Coloured stderr output with ``<...>`` text must render correctly."""
    content = "\x1b[31m<class 'int'>\x1b[0m\n"
    viewer = _make_stderr_viewer(tmp_path, content)
    html = viewer._repr_html_()

    assert "<class" not in html
    assert "&lt;class" in html or "&#x3C;" in html.lower()
    assert 'style="color: #cd3131;"' in html
    assert "style=&quot;" not in html


# ---------------------------------------------------------------------------
# Regression – plain text (no ANSI, no special chars) is still displayed
# ---------------------------------------------------------------------------


def test_stdout_html_plain_text_is_displayed(tmp_path: Path) -> None:
    """Plain output without ANSI codes must still appear in the HTML."""
    viewer = _make_stdout_viewer(tmp_path, "hello from job\n")
    html = viewer._repr_html_()

    assert "hello from job" in html


def test_stderr_html_plain_text_is_displayed(tmp_path: Path) -> None:
    """Plain stderr output without ANSI codes must still appear in the HTML."""
    viewer = _make_stderr_viewer(tmp_path, "some warning\n")
    html = viewer._repr_html_()

    assert "some warning" in html
