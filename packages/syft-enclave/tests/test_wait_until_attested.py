import os
import time

os.environ["PRE_SYNC"] = "false"

import pytest

from syft_enclaves import SyftEnclaveClient
from syft_enclaves.attestation.result import AttestationError


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


@pytest.fixture
def ds():
    enclave, _, _, ds = SyftEnclaveClient.quad_with_mock_drive_service_connection()
    ds.enclave_email = enclave.email
    return ds


def test_waits_for_evidence_then_verifies_it(ds, monkeypatch, clock):
    published = iter([None, None, "evidence"])
    monkeypatch.setattr(ds, "_peer_evidence", lambda email, quiet: next(published))
    calls = []

    def attest_peer(email, **kwargs):
        calls.append((email, kwargs))
        return "verified"

    monkeypatch.setattr(ds, "attest_peer", attest_peer)

    result = ds.wait_until_attested(ds.enclave_email, policy="the policy")
    assert result == "verified"
    assert calls == [
        (
            ds.enclave_email,
            {
                "expected_image_digest": None,
                "expected_data_owners": None,
                "expected_email": None,
                "policy": "the policy",
            },
        )
    ]
    assert clock["sleeps"] == [15, 15]


def test_times_out_quietly_when_no_evidence_arrives(ds, monkeypatch, clock, capsys):
    monkeypatch.setattr(ds, "_peer_evidence", lambda email, quiet: None)
    capsys.readouterr()

    with pytest.raises(TimeoutError, match="attestation"):
        ds.wait_until_attested(ds.enclave_email, timeout=30)
    assert "published no attestation evidence" not in capsys.readouterr().out
    assert clock["sleeps"] == [15, 15]


def test_failed_verification_raises_at_once(ds, monkeypatch, clock):
    monkeypatch.setattr(ds, "_peer_evidence", lambda email, quiet: "evidence")

    def attest_peer(email, **kwargs):
        raise AttestationError("digest mismatch")

    monkeypatch.setattr(ds, "attest_peer", attest_peer)

    with pytest.raises(AttestationError, match="digest mismatch"):
        ds.wait_until_attested(ds.enclave_email)
    assert clock["sleeps"] == []


def test_bad_arguments_fail_before_wait(ds, monkeypatch, clock):
    def no_lookup(email, quiet):
        raise AssertionError("must not look for evidence")

    monkeypatch.setattr(ds, "_peer_evidence", no_lookup)

    with pytest.raises(ValueError, match="not both"):
        ds.wait_until_attested(
            ds.enclave_email, expected_email=ds.enclave_email, policy="the policy"
        )
    assert clock["sleeps"] == []
