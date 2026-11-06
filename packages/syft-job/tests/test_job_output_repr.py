from types import SimpleNamespace

import pytest

from syft_job.job_repr import StderrViewer
from syft_job.job_stdout import StdoutViewer


@pytest.mark.parametrize(
    ("viewer_class", "file_name"),
    [
        (StdoutViewer, "stdout.txt"),
        (StderrViewer, "stderr.txt"),
    ],
)
def test_job_output_html_escapes_markup_and_preserves_ansi(
    tmp_path, viewer_class, file_name
):
    output_file = tmp_path / file_name
    output_file.write_text(
        "<class 'int'>\n\x1b[31mfailed\x1b[0m\n<script>alert('xss')</script>",
        encoding="utf-8",
    )
    viewer = viewer_class(
        SimpleNamespace(status="done", artifact_path=lambda _: output_file)
    )

    rendered_html = viewer._repr_html_()

    assert "&lt;class &#x27;int&#x27;&gt;" in rendered_html
    assert '<span style="color: #cd3131;">failed</span>' in rendered_html
    assert "<script>" not in rendered_html
    assert "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;" in rendered_html
