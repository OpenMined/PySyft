import pytest
from syft_job.job_runner import (
    DEFAULT_JOB_TIMEOUT_SECONDS,
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
