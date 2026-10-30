"""Job stdout and stderr viewers render output as escaped, colored HTML."""

from pathlib import Path
from types import SimpleNamespace

from syft_job.job_repr import StderrViewer
from syft_job.job_stdout import StdoutViewer


def _viewer(tmp_path: Path, content: str) -> StdoutViewer:
    (tmp_path / "stdout.txt").write_text(content)
    job_info = SimpleNamespace(
        status="done", artifact_path=lambda name: tmp_path / name
    )
    return StdoutViewer(job_info)


def test_repr_html_escapes_angle_brackets(tmp_path: Path):
    html = _viewer(tmp_path, "<class 'int'>\n<script>alert(1)</script>\n")._repr_html_()

    assert "&lt;class &#x27;int&#x27;&gt;" in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>" not in html


def test_repr_html_keeps_ansi_color_spans(tmp_path: Path):
    html = _viewer(tmp_path, "\x1b[31mfailed\x1b[0m & done\n")._repr_html_()

    assert '<span style="color: #cd3131;">failed</span> &amp; done' in html


def test_stderr_repr_html_escapes_output_and_keeps_colors(tmp_path: Path):
    (tmp_path / "stderr.txt").write_text(
        'File "run.py", line 1, in <module>\n\x1b[31mValueError\x1b[0m\n'
    )
    job_info = SimpleNamespace(
        status="done", artifact_path=lambda name: tmp_path / name
    )
    html = StderrViewer(job_info)._repr_html_()

    assert "in &lt;module&gt;" in html
    assert '<span style="color: #cd3131;">ValueError</span>' in html
