import pytest

from syft_job.job_runner import format_timeout


@pytest.mark.parametrize(
    "timeout,expected",
    [
        (30, "30 seconds"),
        (1, "1 second"),
        (59, "59 seconds"),
        (60, "1 minute"),
        (90, "90 seconds"),
        (120, "2 minutes"),
        (1800, "30 minutes"),
        (2.5, "2.5 seconds"),
    ],
)
def test_format_timeout(timeout, expected):
    assert format_timeout(timeout) == expected
