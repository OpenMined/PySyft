"""Enclave runner — drives the enclave lifecycle.

Two ways to use this:

  * ``run()`` — long-running daemon mode. Installs signal handlers, calls
    ``init()``, then loops ``tick()`` on ``poll_interval`` until interrupted.
  * ``init()`` + ``tick()`` — manual mode (e.g. inside a notebook). Call
    ``init()`` once, then call ``tick()`` whenever the enclave needs to act.

``init()`` runs three startup phases in order: initialize → attest → peer.
``tick()`` runs one iteration: sync → receive_jobs → run_jobs → distribute_results.
"""

import logging
import signal
import time
from typing import Callable, Optional

from syft.sync.peers.peer_store import datasite_crypto_keys_path
from syft.version import SYFT_VERSION

from syft_enclaves.attestation.claims import build_claims

from syft_enclaves.client import SyftEnclaveClient
from syft_enclaves.evidence.key_bundle import write_public_bundle
from syft_enclaves.evidence import (
    AUTO,
    EvidenceProvider,
    probed_locations,
    select_provider,
)

logger = logging.getLogger(__name__)


class EnclaveRunner:
    """Drives the enclave lifecycle. Use ``run()`` for daemon mode or
    ``init()`` + ``tick()`` to drive it manually (e.g. from a notebook)."""

    def __init__(
        self,
        client: SyftEnclaveClient,
        poll_interval: int = 1,
        require_tee: bool = False,
        fresh_state: bool = True,
        post_init: Optional[Callable[[], None]] = None,
        attestation_provider: str = AUTO,
        settings: Optional[object] = None,
    ) -> None:
        self.client = client
        self.poll_interval = poll_interval
        self.require_tee = require_tee
        self.fresh_state = fresh_state
        self.post_init = post_init
        # Which deployment target to collect evidence from; see
        # syft_enclaves.evidence. ``settings`` is passed through so each
        # provider can read its own configuration.
        self.attestation_provider = attestation_provider
        self.settings = settings
        self._shutdown_requested = False

    # -- public API -------------------------------------------------------

    def init(self) -> None:
        """Run the startup phases: initialize → attest → peer."""
        logger.info(
            "Enclave runner initializing — email=%s syftbox_folder=%s",
            self.client.email,
            self.client.syftbox_folder,
        )

        logger.info("init step 1/3: initializing")
        self._on_initializing()
        logger.info("init step 1/3: initializing complete")

        logger.info("init step 2/3: attesting")
        self._on_attesting()
        logger.info("init step 2/3: attesting complete")

        logger.info("init step 3/3: peering")
        self._on_peering()
        logger.info("init step 3/3: peering complete")

        if self.post_init is not None:
            logger.info("running post_init hook")
            self.post_init()
            logger.info("post_init hook complete")

        logger.info("Enclave runner init complete")

    def tick(self) -> None:
        """One iteration: accept peers, sync, receive_jobs, run_jobs, distribute_results."""
        logger.info("tick start")
        started = time.monotonic()
        try:
            self._accept_peers()
            self.client.sync()
            self.client.receive_jobs()
            self.client.run_jobs()
            self.client.distribute_results()
        except Exception:
            logger.exception("Error during tick")
            # Don't crash — log and retry next cycle
        logger.info("tick complete (%.2fs)", time.monotonic() - started)

    def run(self) -> None:
        """Daemon mode: install signal handlers, init, then loop tick()."""
        self._install_signal_handlers()
        logger.info("Enclave runner starting — poll=%ds", self.poll_interval)
        try:
            self.init()
            self._loop()
        except Exception:
            logger.exception("Fatal error in enclave runner")
            raise
        finally:
            self._on_shutting_down()

    # -- phase handlers ---------------------------------------------------

    def _on_initializing(self) -> None:
        """Validate configuration; optionally wipe state for a clean slate."""
        logger.info("Initializing enclave for %s", self.client.email)
        if self.fresh_state:
            logger.warning(
                "fresh_state=true — wiping ALL SyftBox state "
                "(local folder + Google Drive files) before init"
            )
            self.client.delete_syftbox()
            logger.info("State wipe complete — enclave starts with a clean slate")

    def _on_attesting(self) -> None:
        """Collect attestation evidence and publish it to the version file."""
        provider = select_provider(self.attestation_provider, self.settings)
        if provider is None:
            if self.require_tee:
                raise RuntimeError(
                    "No TEE detected. Probed: "
                    f"{probed_locations()}. "
                    "Set require_tee=False for local testing."
                )
            logger.warning("Running outside TEE — attestation unavailable")
            return
        logger.info(
            "TEE detected (%s) — collecting attestation evidence", provider.kind.value
        )
        # The key bundle first: on a target that can bind claims, the token
        # commits to a digest covering it, so it has to exist before minting.
        self._publish_key_bundle()
        self._publish_attestation(provider)

    def _publish_attestation(self, provider: EvidenceProvider) -> None:
        """Write the provider's evidence into the peer-visible version file."""
        evidence = provider.collect(**self._binding_for(provider))
        peer_manager = self.client._rds.peer_manager
        evidence.publish_to(peer_manager.get_own_version())
        peer_manager.write_own_version()
        logger.info(
            "Attestation evidence (%s) published to SYFT_version.json",
            evidence.kind.value,
        )

    def _binding_for(self, provider: EvidenceProvider) -> dict:
        """The runtime facts to commit to, on targets that can commit to any.

        Confidential Space lets the workload put a digest into the signed
        token, which is the only way a peer can trust the enclave's email, its
        configured data owners or its key bundle — all runtime values outside
        the measurement. Tinfoil has no such channel and binds over a pinned
        connection instead, so it gets nothing here.
        """
        if not getattr(provider, "accepts_caller_nonce", False):
            return {}
        peer_store = self.client._rds.peer_manager.peer_store
        bundle = (
            peer_store.get_public_bundle()
            if peer_store.use_encryption and peer_store.has_my_keys()
            else None
        )
        claims = build_claims(
            email=self.client.email,
            data_owners=self.client.data_owners,
            syft_version=SYFT_VERSION,
            key_bundle=bundle,
        )
        logger.info(
            "Binding runtime claims into the attestation token: email, "
            "%d data owner(s), key bundle %s",
            len(claims["data_owners"]),
            "included" if bundle else "absent",
        )
        return {"claims": claims}

    def _publish_key_bundle(self) -> None:
        """Expose our public key bundle on the attestation endpoint.

        A peer fetching the report over a pinned connection gets the bundle
        from the same channel, which binds it to the hardware report. Skipped
        when encryption is off, since then there is no bundle.
        """
        peer_store = self.client._rds.peer_manager.peer_store
        if not peer_store.use_encryption or not peer_store.has_my_keys():
            logger.info("Encryption disabled — no key bundle to publish")
            return
        peer_manager = self.client._rds.peer_manager
        # The per-datasite default location; the enclave never overrides
        # crypto_keys_path, and PeerManager does not carry the resolved value.
        keys_path = datasite_crypto_keys_path(
            peer_manager.syftbox_folder, peer_store.email
        )
        try:
            # fresh_state wiped the syftbox folder a moment ago, taking the key
            # file with it — the keys live on in memory, so persist them again
            # before advertising where they are. Without this the endpoint
            # serves a bundle it cannot sign a nonce with.
            peer_store.save_keys(keys_path)
            write_public_bundle(peer_store.get_public_bundle(), keys_path=keys_path)
        except OSError as e:
            # Not fatal: peers can still fall back to the Drive-published
            # bundle, they just lose the attestation binding.
            logger.warning("Could not publish the public key bundle: %s", e)

    def _on_peering(self) -> None:
        """Load peers and accept pending peer requests."""
        self.client.load_peers()
        self._accept_peers()
        self.client.sync()
        logger.info("Initial peer sync complete — %d peers", len(self.client.peers))

    def _on_shutting_down(self) -> None:
        logger.info("Enclave runner shut down")

    # -- main loop --------------------------------------------------------

    def _loop(self) -> None:
        """Core poll loop — runs until shutdown is requested."""
        logger.info("Entering main loop (interval=%ds)", self.poll_interval)
        while not self._shutdown_requested:
            self.tick()
            self._sleep()

    def _accept_peers(self) -> None:
        """Accept any pending peer requests."""
        self.client.load_peers()
        for peer in self.client.peers:
            if getattr(peer, "state", None) == "requested_by_peer":
                try:
                    self.client.approve_peer_request(peer.email)
                    logger.info("Accepted peer: %s", peer.email)
                except Exception:
                    logger.warning(
                        "Failed to accept peer: %s", peer.email, exc_info=True
                    )

    # -- utilities --------------------------------------------------------

    def _sleep(self) -> None:
        """Interruptible sleep — exits early on shutdown."""
        end = time.monotonic() + self.poll_interval
        while time.monotonic() < end and not self._shutdown_requested:
            time.sleep(0.5)

    def _install_signal_handlers(self) -> None:
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum: int, _frame: Optional[object]) -> None:
        name = signal.Signals(signum).name
        logger.info("Received %s — requesting shutdown", name)
        self._shutdown_requested = True
