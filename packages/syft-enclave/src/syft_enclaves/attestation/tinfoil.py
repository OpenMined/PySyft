"""Appraising Tinfoil attestation evidence.

The hardware report is verified by the ``tinfoil`` SDK, which owns the
SEV-SNP/TDX parsing and the AMD KDS / Intel PCS trust chains — we never
reimplement that. What this module adds is the syft-specific appraisal: that
the measurement matches the one a *named config repo* published in a
Sigstore-signed release, and that the config it committed to pins the container
image (and optionally the syft version) we expect.

How this differs from Confidential Space
----------------------------------------
Stronger: the reference values come from a signed, publicly auditable release
rather than from claims the workload asked for. The image digest is committed
to by the measurement itself, not asserted by the token.

A workload cannot inject a nonce: the report's 64 bytes of user data are the
sha256 of the shim's TLS public key followed by its HPKE public key. That is
also what makes key binding possible, because the report commits to the key
terminating a TLS connection to the enclave — see
``syft_enclaves.attestation.https``. Freshness for the syft key comes from a nonce the
enclave signs with that bundle (``attestation.nonce``), which the report cannot
carry. Evidence stays published to Drive as provenance, but it is never a
fallback here: appraising it instead would mean unbound keys and a replayable
report, so an unreachable enclave is an error.
"""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any, NoReturn, Optional

from pydantic import BaseModel

from syft.version import SYFT_VERSION

from syft_enclaves.attestation.result import AttestationError, AttestationResult
from syft_enclaves.attestation.envelope import AttestationEvidence
from syft_enclaves.attestation.https import (
    AttestationFetchError,
    AttestedPayload,
    fetch_attested_payload,
)
from syft_enclaves.attestation.nonce import NonceVerificationError, verify_challenge
from syft_enclaves.optional_deps import MissingOptionalDependency, require

#: The config repo whose signed releases publish the expected measurement for
#: OpenMined's enclave image. Shipped as a constant because it is the trust
#: anchor: it decides *who was allowed* to produce a measurement we accept.
DEFAULT_TINFOIL_CONFIG_REPO = "OpenMined/syft-enclave-tinfoil"

DEPLOYMENT_ASSET = "tinfoil-deployment.json"
HASH_ASSET = "tinfoil.hash"
#: Tinfoil's GitHub proxy only serves some asset paths (``tinfoil.hash`` yes,
#: ``tinfoil-deployment.json`` no), so fall through to GitHub itself.
GITHUB_RELEASES = "https://github.com"
DOCS = "packages/syft-enclave/docs/tinfoil.md"
REQUEST_TIMEOUT_SECONDS = 30


class TinfoilAppraisalPolicy(BaseModel):
    """Reference values a Tinfoil enclave's evidence is appraised against.

    ``repo`` deliberately has a shipped default and is never taken from the
    peer: the enclave (and whoever controls its transport) writes its own
    evidence, so letting it name the repo would let it choose which releases
    are trusted.
    """

    model_config = {"frozen": True}

    repo: str = DEFAULT_TINFOIL_CONFIG_REPO
    # None -> appraise against the repo's latest release.
    release_tag: Optional[str] = None
    # None -> the image-digest check is skipped and the image is not pinned.
    expected_image_digest: Optional[str] = None
    # None -> skipped. Only meaningful when the config pins SYFT_VERSION.
    expected_syft_version: Optional[str] = SYFT_VERSION
    # Which container in the config carries the enclave.
    container_name: str = "syft-enclave"
    # Where to fetch the report over a connection pinned to the key the report
    # commits to. None -> fall back to the host the enclave advertised in its
    # evidence; still None -> Drive-only, with no key binding.
    host: Optional[str] = None


def verify_tinfoil_evidence(
    evidence: AttestationEvidence,
    policy: Optional[TinfoilAppraisalPolicy] = None,
    verbose: bool = True,
) -> AttestationResult:
    """Verify Tinfoil evidence and return the check checklist.

    Prefers a pinned HTTPS fetch from the enclave: that yields a fresh report
    and binds the enclave's syft key bundle to it (see ``attestation.https``).
    Falls back to the evidence the enclave published to Drive, which still
    proves what code is running but gives no key binding and no freshness.

    Runs every check before raising, so one failure does not hide later ones.
    The hardware report is the exception: without a verified report there are
    no measurements to compare, so it fails fast.
    """
    policy = policy or TinfoilAppraisalPolicy()
    return _TinfoilVerifier(
        evidence, policy, verbose, _fetch_pinned(evidence, policy)
    ).run()


def _fetch_pinned(
    evidence: AttestationEvidence, policy: TinfoilAppraisalPolicy
) -> AttestedPayload:
    """Fetch the report over a pinned connection. Never optional.

    Tinfoil evidence is always appraised from the live enclave: the report on
    its own says what code is running, but only a pinned connection binds the
    enclave's syft keys to it and only a nonce proves the answer is fresh. The
    Drive copy stays published as provenance, and as something to fall back on
    if this ever needs to become optional again — but accepting it here would
    mean silently downgrading to unbound keys and a replayable report.

    The host may come from the verifier's policy or from the enclave's own
    evidence; either way it is untrusted, because a wrong host either fails
    the fingerprint check or is the enclave we wanted.
    """
    host = policy.host or evidence.metadata.get("host")
    if not host:
        raise AttestationError(
            "Cannot verify a Tinfoil enclave: no host to reach it on. The "
            "enclave publishes one in its evidence metadata (set "
            "SYFT_ENCLAVE_TINFOIL_HOST when deploying), or pass "
            "TinfoilAppraisalPolicy(host=...)."
        )
    try:
        return fetch_attested_payload(host)
    except AttestationFetchError as e:
        raise AttestationError(
            f"Cannot verify a Tinfoil enclave: {host} is unreachable ({e}). "
            "Attestation is not downgraded to the Drive-published copy, which "
            "would bind no keys and be replayable."
        ) from e


class _TinfoilVerifier:
    """One verification run. Holds the state the checks hand to each other."""

    def __init__(
        self,
        evidence: AttestationEvidence,
        policy: TinfoilAppraisalPolicy,
        verbose: bool,
        payload: AttestedPayload,
    ) -> None:
        self.evidence = evidence
        self.policy = policy
        self.verbose = verbose
        # Always set: we reached the enclave directly. Its document is used
        # rather than the Drive copy, because it answers our nonce.
        self.payload = payload
        self.result = AttestationResult()
        self.sdk = _TinfoilSDK()
        self.actual_measurement: Any = None
        self.verification: Any = None
        self.expected_measurement: Any = None
        self.release_tag: Optional[str] = policy.release_tag
        self.release_digest: Optional[str] = None
        self.config: Optional[dict[str, Any]] = None
        self.config_error: Optional[str] = None

    def run(self) -> AttestationResult:
        if self.verbose:
            print("🔒 Verifying enclave attestation (tinfoil)...")
        self._warn_on_repo_mismatch()
        self._check_hardware_report()
        self._check_key_binding()
        self._check_nonce_freshness()
        self._check_release_lookup()
        self._check_sigstore_bundle()
        self._check_measurement_match()
        self._check_image_digest()
        self._check_version_match()
        return self._finish()

    # -- checks -----------------------------------------------------------

    def _check_hardware_report(self) -> None:
        """The quote is genuine SEV-SNP/TDX, chained to the CPU vendor.

        The SDK's defaults also enforce debug-disabled, minimum TCB and
        firmware versions, so this subsumes the secure-boot and debug checks
        the Confidential Space verifier makes separately.
        """
        self._progress("Hardware report")
        document = json.dumps(self.payload.document).encode()
        try:
            verification = self.sdk.attestation.verify_attestation_json(document)
        except MissingOptionalDependency:
            raise
        except Exception as e:
            self._fail_fast("hardware_report", "Hardware report", f"invalid: {e}")
        self.actual_measurement = verification.measurement
        self.verification = verification
        self.result.add(
            "hardware_report",
            "Hardware report",
            True,
            "genuine TEE quote, debug disabled, TCB at minimum "
            f"(fetched from {self.payload.host})",
        )

    def _check_key_binding(self) -> None:
        """The channel ends inside the attested enclave, so its keys are its own.

        The report's user data is the sha256 of the shim's TLS public key, so
        comparing it to the certificate we were actually served proves the
        connection terminates in the enclave the report describes. Only then is
        the key bundle that came down the same connection trustworthy.
        """
        self._progress("Key binding")
        expected = getattr(self.verification, "public_key_fp", None)
        if not expected:
            self.result.add(
                "key_binding", "Key binding", False, "report commits to no TLS key"
            )
            return
        if expected != self.payload.tls_public_key_fp:
            self.result.add(
                "key_binding",
                "Key binding",
                False,
                "the certificate served is not the key the report commits to "
                f"(served {self.payload.tls_public_key_fp[:16]}…, report "
                f"{expected[:16]}…) — the connection does not end in this enclave",
            )
            return
        self._record_verified_bundle()

    def _record_verified_bundle(self) -> None:
        """Note that the channel is bound. The bundle is adopted in _finish.

        Adoption waits for the nonce proof: a bundle that arrived over a bound
        channel is authentic, but until the enclave signs our challenge we have
        no evidence it can actually use the key.
        """
        bundle = self.payload.key_bundle
        detail = (
            "channel ends in the attested enclave; its key bundle is bound to "
            "the report"
            if bundle
            else "channel ends in the attested enclave, but it served no key "
            "bundle (encryption off, or an older enclave image)"
        )
        self.result.add("key_binding", "Key binding", True, detail)

    def _check_nonce_freshness(self) -> None:
        """The enclave holds the key it served, and answered *this* exchange.

        The hardware report cannot carry a caller nonce, so freshness for the
        syft key comes from a challenge: the enclave signs our nonce with the
        identity key from the bundle it served. Failing this means the
        responder produced a bundle it cannot use, or replayed an older answer.
        """
        self._progress("Nonce freshness")
        bundle = self.payload.key_bundle
        if not bundle:
            self.result.add(
                "nonce_freshness",
                "Nonce freshness",
                None,
                "the enclave served no key bundle, so there is no key to prove "
                "possession of (skipped)",
            )
            return
        try:
            verify_challenge(bundle, self.payload.nonce, self.payload.nonce_signature)
        except NonceVerificationError as e:
            self.result.add("nonce_freshness", "Nonce freshness", False, str(e))
            return
        self.result.add(
            "nonce_freshness",
            "Nonce freshness",
            True,
            "the enclave signed our nonce with the key it served",
        )

    def _check_release_lookup(self) -> None:
        """Resolve which published release we appraise against."""
        self._progress("Release lookup")
        try:
            self.release_tag, self.release_digest = self._resolve_release()
        except MissingOptionalDependency:
            raise
        except Exception as e:
            self.result.add(
                "release_lookup", "Release lookup", False, f"could not resolve: {e}"
            )
            return
        self.result.add(
            "release_lookup",
            "Release lookup",
            True,
            f"{self.policy.repo} @ {self.release_tag}",
        )

    def _check_sigstore_bundle(self) -> None:
        """The measurement was signed by that repo's own release workflow."""
        self._progress("Code transparency")
        if self.release_digest is None:
            self.result.add(
                "sigstore_bundle", "Code transparency", None, "no release (skipped)"
            )
            return
        try:
            self.expected_measurement = self._verified_expected_measurement()
        except MissingOptionalDependency:
            raise
        except Exception as e:
            self.result.add(
                "sigstore_bundle", "Code transparency", False, f"unverified: {e}"
            )
            return
        self.result.add(
            "sigstore_bundle",
            "Code transparency",
            True,
            f"signed by {self.policy.repo} on refs/tags/{self.release_tag}",
        )

    def _check_measurement_match(self) -> None:
        """What is running equals what was published."""
        self._progress("Measurement match")
        if self.expected_measurement is None or self.actual_measurement is None:
            self.result.add(
                "measurement_match",
                "Measurement match",
                None,
                "no published measurement to compare (skipped)",
            )
            return
        try:
            self.expected_measurement.assert_equal(self.actual_measurement)
        except Exception as e:
            self.result.add(
                "measurement_match",
                "Measurement match",
                False,
                f"enclave does not match the published release: {e}",
            )
            return
        self.result.add(
            "measurement_match",
            "Measurement match",
            True,
            "enclave matches the published measurement",
        )

    def _check_image_digest(self) -> None:
        """The measured config pins the container image we expect."""
        self._progress("Image digest")
        if not self.policy.expected_image_digest:
            self.result.add(
                "image_digest",
                "Image digest",
                None,
                "no expected image digest supplied — pass one via "
                "TinfoilAppraisalPolicy to pin the image (skipped)",
            )
            return
        digest = self._config_image_digest()
        if digest is None:
            self.result.add(
                "image_digest",
                "Image digest",
                False,
                "could not read the image digest from the verified config"
                + (f": {self.config_error}" if self.config_error else ""),
            )
            return
        passed = digest == self.policy.expected_image_digest
        detail = (
            "container matches expected image"
            if passed
            else f"digest mismatch (got {digest}, expected "
            f"{self.policy.expected_image_digest})"
        )
        self.result.add("image_digest", "Image digest", passed, detail)

    def _check_version_match(self) -> None:
        """The measured config pins the syft version we expect, if it pins one."""
        self._progress("Version match")
        actual = self._config_env().get("SYFT_VERSION")
        expected = self.policy.expected_syft_version
        if not expected:
            self.result.add(
                "version_match", "Version match", None, "no expected version (skipped)"
            )
            return
        if not actual:
            self.result.add(
                "version_match",
                "Version match",
                None,
                "config pins no SYFT_VERSION (skipped)",
            )
            return
        passed = actual == expected
        detail = (
            f"enclave runs expected syft {expected}"
            if passed
            else f"version mismatch (enclave={actual!r}, expected={expected!r})"
        )
        self.result.add("version_match", "Version match", passed, detail)

    # -- reference values -------------------------------------------------

    def _resolve_release(self) -> tuple[str, str]:
        """The (tag, digest) to appraise against.

        The digest is the sha256 of the release's ``tinfoil-deployment.json``,
        and it is what the Sigstore DSSE commits to as its subject.
        """
        if self.policy.release_tag is None:
            release = self.sdk.github.fetch_latest_release(self.policy.repo)
            return release.tag, release.digest
        tag = self.policy.release_tag
        # No SDK helper takes a tag, so read the release's own hash asset.
        return tag, self._fetch_asset(tag, HASH_ASSET).decode().strip()

    def _verified_expected_measurement(self) -> Any:
        bundle = self.sdk.github.fetch_attestation_bundle(
            self.policy.repo, self.release_digest
        )
        return self.sdk.sigstore.verify_attestation(
            bundle,
            self.release_digest,
            self.policy.repo,
            expected_release_tag=self.release_tag,
        )

    def _config_env(self) -> dict[str, Any]:
        container = self._config_container()
        if container is None:
            return {}
        return _env_mapping(container.get("env") or [])

    def _config_image_digest(self) -> Optional[str]:
        container = self._config_container()
        if container is None:
            return None
        image = container.get("image", "")
        _, _, digest = image.partition("@")
        return digest or None

    def _config_container(self) -> Optional[dict[str, Any]]:
        try:
            config = self._verified_config()
        except MissingOptionalDependency:
            raise
        except Exception as e:
            self.config_error = str(e)
            return None
        if config is None:
            return None
        for container in config.get("containers") or []:
            if container.get("name") == self.policy.container_name:
                return container
        return None

    def _verified_config(self) -> Optional[dict[str, Any]]:
        """The ``tinfoil-config.yml`` the enclave booted with.

        Trustworthy only because the deployment JSON that embeds it hashes to
        the digest the Sigstore DSSE signed as its subject — so we check that
        before reading anything out of it.
        """
        if self.config is not None or self.release_digest is None:
            return self.config
        raw = self._fetch_asset(self.release_tag, DEPLOYMENT_ASSET)
        if hashlib.sha256(raw).hexdigest() != self.release_digest:
            raise AttestationError(
                f"{DEPLOYMENT_ASSET} does not hash to the signed release digest "
                f"{self.release_digest}"
            )
        deployment = json.loads(raw)
        yaml = require(
            "yaml", extra="tinfoil", feature="Tinfoil attestation", docs=DOCS
        )
        self.config = yaml.safe_load(base64.b64decode(deployment["config"]))
        return self.config

    def _fetch_asset(self, tag: Optional[str], name: str) -> bytes:
        """Download a release asset, trying each host in turn.

        The deployment JSON needs no trusted source: it is accepted only once
        it hashes to the digest the Sigstore DSSE signed. The digest itself is
        a trust input — a hostile source could substitute the digest of another
        *legitimately signed* release of the same repo, i.e. roll us back. Pin
        ``policy.release_tag`` to rule that out; the signature policy then
        requires that exact tag.
        """
        requests = require(
            "requests", extra="tinfoil", feature="Tinfoil attestation", docs=DOCS
        )
        errors = []
        for host in (self.sdk.github.GITHUB_PROXY, GITHUB_RELEASES):
            url = f"{host}/{self.policy.repo}/releases/download/{tag}/{name}"
            try:
                response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
                response.raise_for_status()
                return response.content
            except Exception as e:
                errors.append(f"{url}: {e}")
        raise AttestationError(f"Could not fetch {name} — tried {'; '.join(errors)}")

    # -- plumbing ---------------------------------------------------------

    def _warn_on_repo_mismatch(self) -> None:
        """The peer's own claim about its repo is a hint, never a trust input."""
        claimed = self.evidence.metadata.get("repo")
        if claimed and claimed != self.policy.repo:
            print(
                f"⚠️  Enclave claims config repo {claimed!r} but verifying "
                f"against {self.policy.repo!r} (the verifier's policy wins)."
            )

    def _progress(self, label: str) -> None:
        if self.verbose:
            print(f"  ⏳ {label} ...")

    def _fail_fast(self, name: str, label: str, detail: str) -> NoReturn:
        self.result.add(name, label, False, detail)
        if self.verbose:
            self.result.print_checklist()
            print("❌ Attestation failed — no verified report, cannot inspect further")
        raise AttestationError(f"Attestation failed: {name}", self.result)

    def _adopt_bundle_if_proven(self) -> None:
        """Hand the bundle back only if both binding and possession hold."""
        proven = {
            check.name: check.passed
            for check in self.result.checks
            if check.name in ("key_binding", "nonce_freshness")
        }
        if proven.get("key_binding") and proven.get("nonce_freshness"):
            self.result.verified_key_bundle = self.payload.key_bundle

    def _finish(self) -> AttestationResult:
        if self.verbose:
            self.result.print_checklist()
        self._adopt_bundle_if_proven()
        failed = [c for c in self.result.checks if c.passed is False]
        if failed:
            names = ", ".join(c.name for c in failed)
            if self.verbose:
                print(
                    f"❌ Attestation failed — {len(failed)} check(s) did not pass: "
                    f"{names}"
                )
            raise AttestationError(f"Attestation failed: {names}", self.result)
        if self.verbose:
            print("🔒 Attestation verified — enclave is trusted")
        return self.result


class _TinfoilSDK:
    """The tinfoil SDK submodules, imported on first use.

    Lazy so that importing this module — for ``TinfoilAppraisalPolicy``, say —
    never requires the optional dependency.
    """

    def __init__(self) -> None:
        self._modules: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        if name not in self._modules:
            self._modules[name] = require(
                f"tinfoil.{name}",
                extra="tinfoil",
                feature="Tinfoil attestation",
                docs=DOCS,
            )
        return self._modules[name]


def _env_mapping(entries: Any) -> dict[str, Any]:
    """Flatten a tinfoil config ``env`` block into a mapping.

    Entries are either ``{"KEY": "value"}`` maps or bare ``"KEY"`` strings
    (inherited from the deploy environment, so no value is pinned).
    """
    env: dict[str, Any] = {}
    if isinstance(entries, dict):
        return dict(entries)
    for entry in entries:
        if isinstance(entry, dict):
            env.update(entry)
    return env
