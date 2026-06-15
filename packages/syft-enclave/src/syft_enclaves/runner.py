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
from typing import Optional

from syft_enclaves.client import SyftEnclaveClient
from syft_enclaves.tee_token import (
    TEE_SOCKET_PATH,
    build_eat_nonce,
    fetch_attestation_token,
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
    ) -> None:
        self.client = client
        self.poll_interval = poll_interval
        self.require_tee = require_tee
        self.fresh_state = fresh_state
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
            self.client._manager.delete_syftbox()
            logger.info("State wipe complete — enclave starts with a clean slate")

    def _on_attesting(self) -> None:
        """Verify TEE environment and publish attestation token to version file."""
        in_tee = TEE_SOCKET_PATH.exists()
        if self.require_tee and not in_tee:
            raise RuntimeError(
                f"TEE socket not found at {TEE_SOCKET_PATH}. "
                "Set require_tee=False for local testing."
            )
        if in_tee:
            logger.info("Confidential Spaces TEE detected — fetching attestation token")
            self._publish_attestation()
        else:
            logger.warning("Running outside TEE — attestation unavailable")

    def _publish_attestation(self) -> None:
        """Fetch attestation JWT from the TEE and write it into the version file."""
        eat_nonce = build_eat_nonce()
        token = fetch_attestation_token(eat_nonce=eat_nonce)
        self.client._manager.peer_manager.get_own_version().attestation_token = token
        self.client._manager.peer_manager.write_own_version()
        logger.info("Attestation token published to SYFT_version.json")

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
