"""Tests for provider selection and the two evidence providers."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from syft_enclaves.attestation.envelope import AttestationKind
from syft_enclaves.evidence import (
    PROVIDERS,
    probed_locations,
    select_provider,
)
from syft_enclaves.evidence.confidential_space import (
    ConfidentialSpaceProvider,
    decode_jwt_payload,
    structure_claims,
)
from syft_enclaves.evidence.tinfoil import TinfoilProvider

TINFOIL_DOC = {
    "format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
    "body": "H4sIAAAAAAAA/2JmgAEEixBg",
}


@pytest.fixture
def tinfoil_mount(tmp_path, monkeypatch):
    """A fake /tinfoil mount, as Tinfoil presents it to the container."""
    (tmp_path / "attestation.json").write_text(json.dumps(TINFOIL_DOC))
    (tmp_path / "config.yml").write_text("cpus: 2\n")
    (tmp_path / "container-status.json").write_text('{"syft-enclave": "running"}')
    for name, attr in [
        ("attestation.json", "TINFOIL_ATTESTATION_PATH"),
        ("config.yml", "TINFOIL_CONFIG_PATH"),
        ("container-status.json", "TINFOIL_STATUS_PATH"),
    ]:
        monkeypatch.setattr(f"syft_enclaves.evidence.tinfoil.{attr}", tmp_path / name)
    return tmp_path


class TestSelection:
    def test_none_disables_attestation(self):
        assert select_provider("none") is None

    def test_auto_finds_nothing_outside_a_tee(self):
        assert select_provider("auto") is None

    def test_unknown_name_raises_and_lists_the_options(self):
        with pytest.raises(ValueError, match="Unknown attestation provider"):
            select_provider("nitro-enclave")

    def test_explicit_provider_that_does_not_detect_returns_none(self):
        # A misconfigured deployment publishes nothing loudly rather than
        # pretending some other provider applies.
        assert select_provider("tinfoil") is None

    def test_auto_prefers_whichever_tee_is_present(self, tinfoil_mount):
        provider = select_provider("auto")
        assert isinstance(provider, TinfoilProvider)
        assert provider.kind is AttestationKind.TINFOIL

    def test_probed_locations_names_both_targets(self):
        assert "/run/container_launcher/teeserver.sock" in probed_locations()
        assert "/tinfoil/attestation.json" in probed_locations()

    def test_every_registered_provider_implements_the_seam(self):
        for provider_cls in PROVIDERS.values():
            assert hasattr(provider_cls, "kind")
            assert hasattr(provider_cls, "probe_path")
            for method in ("detect", "from_settings", "collect", "describe"):
                assert callable(getattr(provider_cls, method))


class TestTinfoilProvider:
    def test_collects_the_mounted_document(self, tinfoil_mount):
        evidence = TinfoilProvider(repo="OpenMined/x", release_tag="v0.0.1").collect()
        assert evidence.kind is AttestationKind.TINFOIL
        assert evidence.format == TINFOIL_DOC["format"]
        assert evidence.body == TINFOIL_DOC["body"]
        assert evidence.metadata == {"repo": "OpenMined/x", "release_tag": "v0.0.1"}

    def test_reads_repo_and_tag_from_settings(self, tinfoil_mount):
        class Settings:
            tinfoil_repo = "OpenMined/syft-enclave-tinfoil"
            tinfoil_release_tag = "v1.2.3"

        provider = TinfoilProvider.from_settings(Settings())
        assert provider.repo == "OpenMined/syft-enclave-tinfoil"
        assert provider.release_tag == "v1.2.3"

    def test_settings_are_optional(self):
        provider = TinfoilProvider.from_settings(None)
        assert provider.repo is None and provider.release_tag is None

    def test_a_caller_nonce_is_refused(self, tinfoil_mount):
        # Accepting and ignoring it would imply a freshness guarantee that
        # Tinfoil cannot provide.
        with pytest.raises(ValueError, match="cannot carry a caller nonce"):
            TinfoilProvider().collect(caller_nonce="abc123")

    def test_missing_document_explains_why(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "syft_enclaves.evidence.tinfoil.TINFOIL_ATTESTATION_PATH",
            tmp_path / "absent.json",
        )
        with pytest.raises(RuntimeError, match="not running inside a Tinfoil enclave"):
            TinfoilProvider().collect()

    @pytest.mark.parametrize("content", ["not json at all", '["a", "list"]'])
    def test_malformed_document_raises(self, tmp_path, monkeypatch, content):
        path = tmp_path / "attestation.json"
        path.write_text(content)
        monkeypatch.setattr(
            "syft_enclaves.evidence.tinfoil.TINFOIL_ATTESTATION_PATH", path
        )
        with pytest.raises(RuntimeError, match="Malformed"):
            TinfoilProvider().collect()

    def test_document_without_format_or_body_raises(self, tmp_path, monkeypatch):
        path = tmp_path / "attestation.json"
        path.write_text('{"something": "else"}')
        monkeypatch.setattr(
            "syft_enclaves.evidence.tinfoil.TINFOIL_ATTESTATION_PATH", path
        )
        with pytest.raises(ValueError, match="'format' and 'body'"):
            TinfoilProvider().collect()

    def test_describe_includes_the_booted_config(self, tinfoil_mount):
        provider = TinfoilProvider(repo="OpenMined/x")
        described = provider.describe(provider.collect())
        assert described["config"] == "cpus: 2\n"
        assert described["container_status"] == {"syft-enclave": "running"}
        assert described["repo"] == "OpenMined/x"

    def test_describe_tolerates_a_partial_mount(self, tmp_path, monkeypatch):
        path = tmp_path / "attestation.json"
        path.write_text(json.dumps(TINFOIL_DOC))
        monkeypatch.setattr(
            "syft_enclaves.evidence.tinfoil.TINFOIL_ATTESTATION_PATH", path
        )
        monkeypatch.setattr(
            "syft_enclaves.evidence.tinfoil.TINFOIL_CONFIG_PATH", tmp_path / "gone.yml"
        )
        provider = TinfoilProvider()
        assert provider.describe(provider.collect())["config"] is None


class TestConfidentialSpaceProvider:
    def test_detect_follows_the_launcher_socket(self, tmp_path, monkeypatch):
        socket_path = tmp_path / "teeserver.sock"
        monkeypatch.setattr(
            "syft_enclaves.evidence.confidential_space.TEE_SOCKET_PATH", socket_path
        )
        assert ConfidentialSpaceProvider.detect() is False
        socket_path.write_text("")
        assert ConfidentialSpaceProvider.detect() is True

    def test_collect_wraps_the_launcher_token(self):
        with patch(
            "syft_enclaves.evidence.confidential_space.fetch_attestation_token",
            return_value="header.payload.signature",
        ) as fetch:
            evidence = ConfidentialSpaceProvider().collect()

        assert evidence.kind is AttestationKind.CONFIDENTIAL_SPACE
        assert evidence.body == "header.payload.signature"
        assert evidence.metadata == {"audience": "syft-attestation"}
        # The version nonce is still what the launcher is asked for.
        assert fetch.call_args.kwargs["eat_nonce"][0].startswith("syft-")

    def test_collect_passes_a_caller_nonce_through(self):
        with patch(
            "syft_enclaves.evidence.confidential_space.fetch_attestation_token",
            return_value="a.b.c",
        ) as fetch:
            ConfidentialSpaceProvider().collect(caller_nonce="freshness")
        assert fetch.call_args.kwargs["eat_nonce"][1] == "freshness"

    def test_decode_jwt_payload_rejects_a_non_jwt(self):
        with pytest.raises(ValueError, match="expected 3 parts"):
            decode_jwt_payload("not-a-jwt")

    def test_structure_claims_sections(self):
        structured = structure_claims(
            {
                "secboot": True,
                "dbgstat": "disabled-since-boot",
                "eat_nonce": ["syft-0.1.0"],
                "submods": {
                    "container": {"image_digest": "sha256:abc"},
                    "confidential_space": {"support_attributes": ["X"]},
                },
                "nvidia_gpu": {"mode": "on"},
            }
        )
        assert structured["hardware"]["secboot"] is True
        assert structured["container"]["image_digest"] == "sha256:abc"
        assert structured["gpu"] == {"mode": "on"}
        assert structured["confidential_space"]
        assert structured["eat_nonce"] == ["syft-0.1.0"]

    def test_structure_claims_omits_absent_optional_sections(self):
        structured = structure_claims({"secboot": True})
        for absent in ("gpu", "confidential_space", "eat_nonce"):
            assert absent not in structured


class TestAttestationServerWiring:
    """The /attestation endpoint must configure providers like the runner does.

    Regression: the endpoint called select_provider without settings, so a
    Tinfoil enclave published evidence with an empty metadata block while the
    runner published the repo and release tag.
    """

    def _app(self, monkeypatch):
        monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "docker"))
        import attestation_server

        return attestation_server

    def test_endpoint_passes_settings_through_to_the_provider(
        self, tinfoil_mount, monkeypatch
    ):
        monkeypatch.setenv("SYFT_ENCLAVE_ATTESTATION_PROVIDER", "tinfoil")
        monkeypatch.setenv("SYFT_ENCLAVE_TINFOIL_REPO", "OpenMined/example")
        monkeypatch.setenv("SYFT_ENCLAVE_TINFOIL_RELEASE_TAG", "v9.9.9")

        server = self._app(monkeypatch)
        provider = server._detect_provider()

        assert provider.repo == "OpenMined/example"
        assert provider.release_tag == "v9.9.9"
        assert provider.collect().metadata == {
            "repo": "OpenMined/example",
            "release_tag": "v9.9.9",
        }

    def test_endpoint_reports_no_provider_outside_a_tee(self, monkeypatch):
        monkeypatch.setenv("SYFT_ENCLAVE_ATTESTATION_PROVIDER", "none")
        server = self._app(monkeypatch)
        assert server._detect_provider() is None
