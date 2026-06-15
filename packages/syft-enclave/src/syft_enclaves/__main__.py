"""Entry point for the Syft enclave runner: ``python -m syft_enclaves``.

Configuration is read entirely from ``SYFT_ENCLAVE_*`` environment variables
(see :class:`syft_enclaves.settings.EnclaveSettings`).
"""

import logging
import sys

from pydantic import ValidationError

from syft_enclaves.client import SyftEnclaveClient
from syft_enclaves.inference.logs_dataset import ensure_logs_dataset
from syft_enclaves.runner import EnclaveRunner
from syft_enclaves.settings import EnclaveSettings

logger = logging.getLogger(__name__)


def _configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_settings() -> EnclaveSettings:
    """Load settings, exiting with an actionable message on misconfiguration."""
    try:
        return EnclaveSettings()
    except ValidationError as exc:
        # A missing or malformed SYFT_ENCLAVE_* variable is an operator error,
        # not a bug — fail fast with the validation report, not a traceback.
        print(f"Invalid enclave configuration:\n{exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def main() -> None:
    settings = _load_settings()
    _configure_logging(settings.log_level)
    logger.info("python -m syft_enclaves starting")
    logger.info(
        f"Enclave settings — email={settings.email} data_owners={settings.data_owners} "
        f"token_path={settings.token_path} poll_interval={settings.poll_interval}s "
        f"require_tee={settings.require_tee} fresh_state={settings.fresh_state} "
        f"use_encryption={settings.use_encryption}"
    )

    logger.info("Building SyftEnclaveClient...")
    client = SyftEnclaveClient.for_enclave(
        email=settings.email,
        token_path=settings.token_path,
        data_owners=settings.data_owners,
        encryption=settings.use_encryption,
    )
    logger.info("SyftEnclaveClient ready")

    post_init = None
    if settings.model_owner is not None:
        logger.info(
            f"Inference mode — model_owner={settings.model_owner} "
            f"model_dataset={settings.model_dataset} model_size={settings.model_size} "
            f"logs_dataset={settings.logs_dataset}"
        )
        post_init = lambda: ensure_logs_dataset(client, settings.logs_dataset)  # noqa: E731

    logger.info("Building EnclaveRunner...")
    runner = EnclaveRunner(
        client=client,
        poll_interval=settings.poll_interval,
        require_tee=settings.require_tee,
        fresh_state=settings.fresh_state,
        post_init=post_init,
    )
    logger.info("EnclaveRunner ready — calling runner.run()")
    runner.run()


if __name__ == "__main__":
    main()
