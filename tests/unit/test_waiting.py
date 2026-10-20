import time

import pytest

from syft.sync.utils.waiting import poll_until, wait_for


@pytest.fixture
def clock(monkeypatch):
    """Fake monotonic clock; time.sleep() advances it and records each sleep."""
    state = {"now": 0.0, "sleeps": []}

    def sleep(seconds):
        state["sleeps"].append(seconds)
        state["now"] += seconds

    monkeypatch.setattr(time, "sleep", sleep)
    monkeypatch.setattr(time, "monotonic", lambda: state["now"])
    return state


def test_poll_until_returns_first_result_not_none(clock):
    results = iter([None, None, "found"])

    assert poll_until(lambda: next(results), deadline=100, poll_interval=10) == "found"
    assert clock["sleeps"] == [10, 10, 10]


def test_poll_until_cuts_last_sleep_at_deadline(clock):
    assert poll_until(lambda: None, deadline=25, poll_interval=10) is None
    assert clock["sleeps"] == [10, 10, 5]


def test_wait_for_returns_at_once_when_found(clock):
    assert wait_for(lambda: 1, "x", lambda: "", timeout=60, poll_interval=10) == 1
    assert clock["sleeps"] == []


def test_wait_for_polls_until_found(clock):
    results = iter([None, None, "found"])

    found = wait_for(
        lambda: next(results), "x", lambda: "", timeout=60, poll_interval=10
    )
    assert found == "found"
    assert clock["sleeps"] == [10, 10]


def test_wait_for_timeout_names_target_and_state(clock):
    with pytest.raises(TimeoutError, match="job 'j'.*job 'j' is 'pending'"):
        wait_for(
            lambda: None,
            "job 'j'",
            lambda: "job 'j' is 'pending'",
            timeout=30,
            poll_interval=10,
        )
    assert clock["sleeps"] == [10, 10, 10]


def test_wait_for_with_zero_timeout_checks_once(clock):
    calls = []

    with pytest.raises(TimeoutError):
        wait_for(lambda: calls.append(1), "x", lambda: "", timeout=0, poll_interval=10)
    assert calls == [1]
    assert clock["sleeps"] == []


def test_wait_for_raises_find_error_at_once(clock):
    def find():
        raise ValueError("ambiguous")

    with pytest.raises(ValueError, match="ambiguous"):
        wait_for(find, "x", lambda: "", timeout=60, poll_interval=10)
    assert clock["sleeps"] == []
