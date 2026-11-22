import io
import sys
from syft.utils import bug_report


def test_bug_report_as_dict():
    report = bug_report(as_dict=True)
    assert isinstance(report, dict)
    assert "syft" in report
    assert "os_info" in report
    assert "python_env" in report


def test_bug_report_redact_paths_compatibility():
    report_default = bug_report(as_dict=True, redact_paths=False)
    report_redacted = bug_report(as_dict=True, redact_paths=True)
    assert report_default == report_redacted


def test_bug_report_stdout_safe_notes():
    captured = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = captured
        bug_report(as_dict=False)
    finally:
        sys.stdout = old_stdout
    assert "This report contains no tokens, secrets, paths, or PII." in captured.getvalue()
