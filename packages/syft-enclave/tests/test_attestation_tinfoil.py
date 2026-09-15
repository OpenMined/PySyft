"""Tests for Tinfoil attestation verification.

The real ``tinfoil`` SDK is an optional dependency and is not installed in the
test environment, so it is stubbed into ``sys.modules``. These tests cover the
syft-specific appraisal: which release we compare against, that the embedded
config is only trusted once it hashes to the signed digest, and that the
checklist reports the same way the Confidential Space verifier does.
"""

import base64
import hashlib
import json
import sys
import types
from unittest.mock import MagicMock

import pytest

from syft_enclaves.attestation import AttestationError
from syft_enclaves.attestation_envelope import tinfoil_evidence
from syft_enclaves.attestation_tinfoil import DEPLOYMENT_ASSET, HASH_ASSET
from syft_enclaves.optional_deps import MissingOptionalDependency

TINFOIL_DOC = {
    "format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
    "body": "H4sIAAAAAAAA/2JmgAEEixBg",
}
TLS_FP = "ff" * 32
HOST = "enclave.example"


def _real_keys():
    """A genuine syft keypair, so nonce signatures are really verified."""
    import syft_crypto_python as syc

    keys = syc.SyftRecoveryKey.generate().derive_keys()
    bundle = keys.to_public_bundle().to_did_document("did:syft:enclave@openmined.org")
    bundle["identity"] = "enclave@openmined.org"
    return keys, bundle
IMAGE_DIGEST = "sha256:" + "ab" * 32
IMAGE = f"docker.io/openminedreleasebot/syft-enclave@{IMAGE_DIGEST}"
SYFT_VERSION_IN_CONFIG = "9.9.9"
RELEASE_TAG = "v0.0.1"


def _config(image=IMAGE, name="syft-enclave", syft_version=SYFT_VERSION_IN_CONFIG):
    env = [{"SYFT_ENCLAVE_ATTESTATION_PROVIDER": "tinfoil"}]
    if syft_version:
        env.append({"SYFT_VERSION": syft_version})
    return {"containers": [{"name": name, "image": image, "env": env}]}


def _deployment_bytes(config=None):
    """A release's tinfoil-deployment.json, with the config embedded base64."""
    payload = {
        "snp_measurement": "44c6" * 24,
        "config": base64.b64encode(
            json.dumps(config if config is not None else _config()).encode()
        ).decode(),
    }
    return json.dumps(payload).encode()


class _FakeMeasurement:
    """Stands in for tinfoil's Measurement, whose only job here is comparison."""

    def __init__(self, registers, matches=True):
        self.registers = registers
        self._matches = matches

    def assert_equal(self, other):
        if not self._matches or self.registers != other.registers:
            raise ValueError("measurement mismatch")


@pytest.fixture
def sdk(monkeypatch):
    """Stub the tinfoil SDK submodules and the release-asset HTTP fetch."""
    measurement = _FakeMeasurement(["deadbeef"])

    attestation = types.ModuleType("tinfoil.attestation")
    attestation.verify_attestation_json = MagicMock(
        return_value=types.SimpleNamespace(
            measurement=measurement,
            public_key_fp=TLS_FP,
            hpke_public_key="ee" * 32,
        )
    )

    github = types.ModuleType("tinfoil.github")
    github.GITHUB_PROXY = "https://github-proxy.example"
    deployment = _deployment_bytes()
    digest = hashlib.sha256(deployment).hexdigest()
    github.fetch_latest_release = MagicMock(
        return_value=types.SimpleNamespace(tag=RELEASE_TAG, digest=digest)
    )
    github.fetch_attestation_bundle = MagicMock(return_value=b"{}")

    sigstore = types.ModuleType("tinfoil.sigstore")
    sigstore.verify_attestation = MagicMock(return_value=_FakeMeasurement(["deadbeef"]))

    root = types.ModuleType("tinfoil")
    root.attestation, root.github, root.sigstore = attestation, github, sigstore
    for name, module in {
        "tinfoil": root,
        "tinfoil.attestation": attestation,
        "tinfoil.github": github,
        "tinfoil.sigstore": sigstore,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    assets = {"tinfoil-deployment.json": deployment, "tinfoil.hash": digest.encode()}
    state = {"digest": digest}

    def set_deployment(raw):
        """Publish a different config, re-signing it (digest follows content)."""
        assets["tinfoil-deployment.json"] = raw
        state["digest"] = hashlib.sha256(raw).hexdigest()
        assets["tinfoil.hash"] = state["digest"].encode()
        github.fetch_latest_release.return_value = types.SimpleNamespace(
            tag=RELEASE_TAG, digest=state["digest"]
        )

    def tamper_deployment(raw):
        """Swap the file but leave the signed digest alone."""
        assets["tinfoil-deployment.json"] = raw

    # Mirrors the real world: tinfoil's proxy serves tinfoil.hash but 400s on
    # tinfoil-deployment.json, which only github.com has.
    proxy_blocks = {"tinfoil-deployment.json"}
    fetched = []

    def fake_get(url, timeout=None):
        name = url.rsplit("/", 1)[-1]
        fetched.append(url)
        response = MagicMock()
        if url.startswith(github.GITHUB_PROXY) and name in proxy_blocks:
            response.raise_for_status.side_effect = RuntimeError("400 Bad Request")
            return response
        response.content = assets[name]
        return response

    requests_stub = types.ModuleType("requests")
    requests_stub.get = fake_get
    monkeypatch.setitem(sys.modules, "requests", requests_stub)

    return types.SimpleNamespace(
        attestation=attestation,
        github=github,
        sigstore=sigstore,
        digest=digest,
        assets=assets,
        set_deployment=set_deployment,
        tamper_deployment=tamper_deployment,
        proxy_blocks=proxy_blocks,
        fetched=fetched,
    )


@pytest.fixture
def pinned(monkeypatch):
    """Patch the pinned fetch with a payload the enclave would really serve."""
    from syft_enclaves.attestation_https import AttestedPayload
    from syft_enclaves.nonce_challenge import new_nonce, sign_challenge

    def _install(
        *,
        document=None,
        bundle="real",
        tls_fp=TLS_FP,
        sign_with=None,
        signature="valid",
        unreachable=False,
    ):
        keys, real_bundle = _real_keys()
        served = real_bundle if bundle == "real" else bundle
        nonce = new_nonce()
        if signature == "valid":
            signer = sign_with or keys
            nonce_signature = sign_challenge(signer.to_jwks(), nonce)
        else:
            nonce_signature = signature

        payload = AttestedPayload(
            document=document or TINFOIL_DOC,
            key_bundle=served,
            tls_public_key_fp=tls_fp,
            host=HOST,
            nonce=nonce,
            nonce_signature=nonce_signature,
        )

        def fetch(host, *a, **kw):
            if unreachable:
                from syft_enclaves.attestation_https import AttestationFetchError

                raise AttestationFetchError("connection refused")
            return payload

        monkeypatch.setattr(
            "syft_enclaves.attestation_tinfoil.fetch_attested_payload", fetch
        )
        return payload

    return _install


@pytest.fixture
def verify(sdk, pinned):
    """verify_tinfoil_evidence with the SDK and a pinned fetch stubbed, quiet."""
    from syft_enclaves.attestation_tinfoil import (
        TinfoilAppraisalPolicy,
        verify_tinfoil_evidence,
    )

    def _run(policy=None, evidence=None, install_pinned=True, **policy_kwargs):
        if install_pinned:
            pinned()
        policy_kwargs.setdefault("expected_syft_version", SYFT_VERSION_IN_CONFIG)
        policy_kwargs.setdefault("host", HOST)
        return verify_tinfoil_evidence(
            evidence or tinfoil_evidence(TINFOIL_DOC),
            policy=policy or TinfoilAppraisalPolicy(**policy_kwargs),
            verbose=False,
        )

    return _run


def _names(result):
    return [c.name for c in result.checks]


def _check(result, name):
    return next(c for c in result.checks if c.name == name)


class TestHappyPath:
    def test_all_checks_pass(self, verify):
        result = verify(expected_image_digest=IMAGE_DIGEST)
        assert _names(result) == [
            "hardware_report",
            "key_binding",
            "nonce_freshness",
            "release_lookup",
            "sigstore_bundle",
            "measurement_match",
            "image_digest",
            "version_match",
        ]

    def test_unpinned_release_uses_the_latest(self, verify, sdk):
        verify(expected_image_digest=IMAGE_DIGEST)
        sdk.github.fetch_latest_release.assert_called_once_with(
            "OpenMined/syft-enclave-tinfoil"
        )

    def test_pinned_release_tag_is_enforced_in_the_signature_policy(self, verify, sdk):
        verify(release_tag=RELEASE_TAG, expected_image_digest=IMAGE_DIGEST)
        # A pinned tag must reach sigstore verification, else any tag's
        # signature would be accepted.
        assert (
            sdk.sigstore.verify_attestation.call_args.kwargs["expected_release_tag"]
            == RELEASE_TAG
        )
        sdk.github.fetch_latest_release.assert_not_called()


class TestPinnedChannel:
    """Tinfoil evidence is always appraised from the live enclave.

    The report alone says what code is running. Only a pinned connection binds
    the enclave's syft keys to it, and only a nonce shows the answer is fresh —
    so the Drive copy is provenance, never a fallback.
    """

    def test_no_host_anywhere_is_refused(self, verify):
        with pytest.raises(AttestationError, match="no host to reach it on"):
            verify(host=None, evidence=tinfoil_evidence(TINFOIL_DOC))

    def test_an_unreachable_enclave_is_refused(self, verify, pinned):
        # Downgrading to Drive here would silently lose key binding.
        pinned(unreachable=True)
        with pytest.raises(AttestationError, match="unreachable"):
            verify(install_pinned=False)

    def test_the_host_can_come_from_the_enclaves_own_evidence(self, verify, sdk):
        result = verify(
            host=None, evidence=tinfoil_evidence(TINFOIL_DOC, host="advertised.example")
        )
        assert _check(result, "hardware_report").passed is True

    def test_the_fetched_document_is_used_not_the_drive_copy(self, verify, sdk, pinned):
        fresh = {"format": TINFOIL_DOC["format"], "body": "FRESHER-THAN-DRIVE"}
        pinned(document=fresh)
        verify(install_pinned=False, expected_image_digest=IMAGE_DIGEST)
        passed = json.loads(sdk.attestation.verify_attestation_json.call_args.args[0])
        assert passed == fresh


class TestKeyBinding:
    """The report commits to the TLS key, so matching it proves where we are."""

    def test_passes_when_the_served_key_matches_the_report(self, verify):
        result = verify(expected_image_digest=IMAGE_DIGEST)
        assert _check(result, "key_binding").passed is True

    def test_fails_when_the_served_key_is_not_the_attested_one(self, verify, pinned):
        # A MITM would serve a replayed report plus its own certificate.
        pinned(tls_fp="aa" * 32)
        with pytest.raises(AttestationError) as excinfo:
            verify(install_pinned=False, expected_image_digest=IMAGE_DIGEST)
        check = _check(excinfo.value.result, "key_binding")
        assert check.passed is False
        assert "does not end in this enclave" in check.detail
        assert excinfo.value.result.verified_key_bundle is None

    def test_a_bound_channel_without_a_bundle_still_binds(self, verify, pinned):
        pinned(bundle=None)
        result = verify(install_pinned=False, expected_image_digest=IMAGE_DIGEST)
        assert _check(result, "key_binding").passed is True
        # Nothing to adopt, and the nonce proof has no key to check.
        assert _check(result, "nonce_freshness").passed is None
        assert result.verified_key_bundle is None


class TestNonceFreshness:
    """The enclave must prove it holds the key it served, for this exchange.

    The hardware report cannot carry a caller nonce, so this is what rules out
    a bundle the responder cannot use and an answer produced earlier.
    """

    def test_a_valid_signature_over_our_nonce_passes(self, verify):
        result = verify(expected_image_digest=IMAGE_DIGEST)
        assert _check(result, "nonce_freshness").passed is True

    def test_the_bundle_is_adopted_only_once_possession_is_proven(self, verify, pinned):
        payload = pinned()
        result = verify(install_pinned=False, expected_image_digest=IMAGE_DIGEST)
        assert result.verified_key_bundle == payload.key_bundle

    def test_a_signature_by_some_other_key_fails(self, verify, pinned):
        import syft_crypto_python as syc

        # Signed with a key that is not the one in the served bundle.
        pinned(sign_with=syc.SyftRecoveryKey.generate().derive_keys())
        with pytest.raises(AttestationError) as excinfo:
            verify(install_pinned=False, expected_image_digest=IMAGE_DIGEST)
        check = _check(excinfo.value.result, "nonce_freshness")
        assert check.passed is False
        assert "does not hold the private half" in check.detail
        # An unproven bundle must never be adopted.
        assert excinfo.value.result.verified_key_bundle is None

    def test_a_missing_signature_fails(self, verify, pinned):
        pinned(signature=None)
        with pytest.raises(AttestationError) as excinfo:
            verify(install_pinned=False, expected_image_digest=IMAGE_DIGEST)
        assert _check(excinfo.value.result, "nonce_freshness").passed is False

    def test_a_garbage_signature_fails(self, verify, pinned):
        pinned(signature="not-base64-at-all!!")
        with pytest.raises(AttestationError) as excinfo:
            verify(install_pinned=False, expected_image_digest=IMAGE_DIGEST)
        assert _check(excinfo.value.result, "nonce_freshness").passed is False


class TestHardwareReport:
    def test_failure_fails_fast(self, verify, sdk):
        sdk.attestation.verify_attestation_json.side_effect = ValueError("bad quote")
        with pytest.raises(AttestationError) as excinfo:
            verify()
        # No later check may run: without a verified report there are no
        # measurements to compare.
        assert _names(excinfo.value.result) == ["hardware_report"]

    def test_the_document_is_passed_through_verbatim(self, verify, sdk):
        verify(expected_image_digest=IMAGE_DIGEST)
        passed = json.loads(sdk.attestation.verify_attestation_json.call_args.args[0])
        assert passed == TINFOIL_DOC


class TestMeasurementAndSignature:
    def test_measurement_mismatch_fails(self, verify, sdk):
        sdk.sigstore.verify_attestation.return_value = _FakeMeasurement(["other"])
        with pytest.raises(AttestationError, match="measurement_match"):
            verify()

    def test_sigstore_failure_skips_the_comparison_rather_than_passing_it(
        self, verify, sdk
    ):
        sdk.sigstore.verify_attestation.side_effect = ValueError("wrong repo")
        with pytest.raises(AttestationError) as excinfo:
            verify()
        result = excinfo.value.result
        assert _check(result, "sigstore_bundle").passed is False
        assert _check(result, "measurement_match").passed is None

    def test_release_lookup_failure_is_reported(self, verify, sdk):
        sdk.github.fetch_latest_release.side_effect = RuntimeError("404")
        with pytest.raises(AttestationError) as excinfo:
            verify()
        assert _check(excinfo.value.result, "release_lookup").passed is False

    def test_verifier_policy_repo_wins_over_the_peers_claim(self, verify, sdk):
        # The enclave writes its own evidence, so its claimed repo must not
        # decide which releases are trusted.
        evidence = tinfoil_evidence(TINFOIL_DOC, repo="attacker/repo")
        verify(evidence=evidence, expected_image_digest=IMAGE_DIGEST)
        assert (
            sdk.sigstore.verify_attestation.call_args.args[2]
            == "OpenMined/syft-enclave-tinfoil"
        )


class TestConfigDerivedChecks:
    def test_image_digest_skipped_when_not_pinned(self, verify):
        result = verify()
        assert _check(result, "image_digest").passed is None
        assert "pass one via" in _check(result, "image_digest").detail

    def test_image_digest_mismatch_fails(self, verify):
        with pytest.raises(AttestationError) as excinfo:
            verify(expected_image_digest="sha256:" + "cd" * 32)
        assert _check(excinfo.value.result, "image_digest").passed is False

    def test_a_tampered_deployment_json_is_not_trusted(self, verify, sdk):
        # The embedded config is only trustworthy because the file hashes to
        # the digest the Sigstore DSSE signed.
        sdk.tamper_deployment(
            _deployment_bytes(_config(image="docker.io/evil@sha256:00"))
        )
        with pytest.raises(AttestationError) as excinfo:
            verify(expected_image_digest=IMAGE_DIGEST)
        result = excinfo.value.result
        assert _check(result, "image_digest").passed is False
        assert _check(result, "version_match").passed is None

    def test_falls_back_to_github_when_the_proxy_rejects_the_asset(
        self, verify, sdk
    ):
        # Regression: tinfoil's proxy allowlists tinfoil.hash only, so reading
        # the config must fall through to github.com.
        result = verify(expected_image_digest=IMAGE_DIGEST)
        assert _check(result, "image_digest").passed is True
        assert any(
            url.startswith("https://github.com") and url.endswith(DEPLOYMENT_ASSET)
            for url in sdk.fetched
        )

    def test_unreachable_from_every_source_fails_the_check(self, verify, sdk):
        sdk.proxy_blocks.add(HASH_ASSET)
        sdk.assets.pop(DEPLOYMENT_ASSET)
        with pytest.raises(AttestationError) as excinfo:
            verify(expected_image_digest=IMAGE_DIGEST)
        assert _check(excinfo.value.result, "image_digest").passed is False

    def test_missing_container_fails_the_digest_check(self, verify):
        with pytest.raises(AttestationError) as excinfo:
            verify(
                expected_image_digest=IMAGE_DIGEST, container_name="not-in-the-config"
            )
        assert _check(excinfo.value.result, "image_digest").passed is False

    def test_version_skipped_when_the_config_pins_none(self, verify, sdk):
        sdk.set_deployment(_deployment_bytes(_config(syft_version=None)))
        result = verify(expected_image_digest=IMAGE_DIGEST)
        assert _check(result, "version_match").passed is None

    def test_version_skipped_when_the_policy_pins_none(self, verify):
        result = verify(expected_image_digest=IMAGE_DIGEST, expected_syft_version=None)
        assert _check(result, "version_match").passed is None

    def test_version_mismatch_fails(self, verify):
        with pytest.raises(AttestationError) as excinfo:
            verify(expected_image_digest=IMAGE_DIGEST, expected_syft_version="0.0.1")
        assert _check(excinfo.value.result, "version_match").passed is False


class TestChecklistBehaviour:
    def test_every_check_runs_after_a_non_fatal_failure(self, verify):
        with pytest.raises(AttestationError) as excinfo:
            verify(expected_image_digest="sha256:" + "cd" * 32)
        # image_digest failed, but version_match must still have been appraised.
        assert _check(excinfo.value.result, "version_match").passed is True

    def test_every_failure_is_named(self, verify, sdk):
        sdk.set_deployment(_deployment_bytes(_config(image="docker.io/x@sha256:0bad")))
        with pytest.raises(AttestationError) as excinfo:
            verify(expected_image_digest=IMAGE_DIGEST, expected_syft_version="0.0.1")
        message = str(excinfo.value)
        assert "image_digest" in message and "version_match" in message

    def test_error_carries_the_result(self, verify, sdk):
        sdk.attestation.verify_attestation_json.side_effect = ValueError("nope")
        with pytest.raises(AttestationError) as excinfo:
            verify()
        assert excinfo.value.result is not None


class TestOptionalDependency:
    def test_missing_tinfoil_explains_how_to_install_it(self, monkeypatch, pinned):
        from syft_enclaves.attestation_tinfoil import (
            TinfoilAppraisalPolicy,
            verify_tinfoil_evidence,
        )

        pinned()
        for name in list(sys.modules):
            if name == "tinfoil" or name.startswith("tinfoil."):
                monkeypatch.delitem(sys.modules, name, raising=False)
        monkeypatch.setattr(
            "syft_enclaves.optional_deps.importlib.import_module",
            MagicMock(side_effect=ImportError("No module named 'tinfoil'")),
        )
        with pytest.raises(MissingOptionalDependency) as excinfo:
            verify_tinfoil_evidence(
                tinfoil_evidence(TINFOIL_DOC),
                policy=TinfoilAppraisalPolicy(host=HOST),
                verbose=False,
            )
        message = str(excinfo.value)
        assert 'pip install "syft-enclave[tinfoil]"' in message
        assert "docs/tinfoil.md" in message

    def test_the_policy_is_usable_without_the_sdk(self):
        # Importing the module for its policy must not need the extra.
        from syft_enclaves.attestation_tinfoil import TinfoilAppraisalPolicy

        assert TinfoilAppraisalPolicy().repo == "OpenMined/syft-enclave-tinfoil"
