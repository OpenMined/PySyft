import pytest
from syft_job.job_runner import (
    DEFAULT_JOB_TIMEOUT_SECONDS,
    _format_timeout,
    get_job_timeout_seconds,
)


def test_job_timeout_uses_default_when_environment_variable_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYFT_DEFAULT_JOB_TIMEOUT_SECONDS", raising=False)

    assert get_job_timeout_seconds() == DEFAULT_JOB_TIMEOUT_SECONDS


def test_job_timeout_accepts_positive_environment_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYFT_DEFAULT_JOB_TIMEOUT_SECONDS", "120")

    assert get_job_timeout_seconds() == 120


@pytest.mark.parametrize("value", ["invalid", "0", "-1"])
def test_job_timeout_rejects_invalid_environment_value(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("SYFT_DEFAULT_JOB_TIMEOUT_SECONDS", value)

    with pytest.raises(
        ValueError,
        match=(
            "SYFT_DEFAULT_JOB_TIMEOUT_SECONDS must be a positive integer; "
            f"got {value!r}"
        ),
    ):
        get_job_timeout_seconds()


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0, "0 seconds"),
        (30, "30 seconds"),
        (1, "1 second"),
        (60, "1 minute"),
        (90, "1 minute 30 seconds"),
        (120, "2 minutes"),
        (600, "10 minutes"),
        (3600, "1 hour"),
        (3661, "1 hour 1 minute 1 second"),
        (7384, "2 hours 3 minutes 4 seconds"),
    ],
)
def test_format_timeout_produces_human_readable_string(
    seconds: int, expected: str
) -> None:
    """Timeouts should be formatted as readable hours, minutes, and seconds."""
    assert _format_timeout(seconds) == expected


def test_format_timeout_never_reports_zero_minutes_for_sub_minute_timeout() -> None:
    """Regression test for issue #9527: sub-minute timeouts reported as 0 minutes."""
    for seconds in range(1, 60):
        result = _format_timeout(seconds)
        assert "0 minutes" not in result, (
            f"_format_timeout({seconds}) returned {result!r}, "
            "which incorrectly contains '0 minutes'"
        )
