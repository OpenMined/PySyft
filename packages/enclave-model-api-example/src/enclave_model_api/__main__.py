"""Entry point for the inference enclave runner: ``python -m enclave_model_api``.

Runs the generic enclave runner (from ``syft-enclave``) with an inference-specific
``post_init`` hook that creates the logs dataset on the enclave's own datasite.
Configuration is read from ``SYFT_ENCLAVE_*`` environment variables — base fields
via :class:`syft_enclaves.settings.EnclaveSettings`, model fields via
:class:`enclave_model_api.settings.InferenceSettings`.
"""

import logging
import sys

from pydantic import ValidationError
from syft_enclaves.client import SyftEnclaveClient
from syft_enclaves.runner import EnclaveRunner
from syft_enclaves.settings import EnclaveSettings

from enclave_model_api.logs_dataset import ensure_logs_dataset
from enclave_model_api.settings import InferenceSettings

logger = logging.getLogger(__name__)


def _configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_settings() -> tuple[EnclaveSettings, InferenceSettings]:
    """Load settings, exiting with an actionable message on misconfiguration."""
    try:
        return EnclaveSettings(), InferenceSettings()
    except ValidationError as exc:
        # A missing or malformed SYFT_ENCLAVE_* variable is an operator error,
        # not a bug — fail fast with the validation report, not a traceback.
        print(f"Invalid enclave configuration:\n{exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def main() -> None:
    settings, inference = _load_settings()
    _configure_logging(settings.log_level)
    logger.info("python -m enclave_model_api starting")
    logger.info(
        f"Enclave settings — email={settings.email} data_owners={settings.data_owners} "
        f"token_path={settings.token_path} poll_interval={settings.poll_interval}s "
        f"require_tee={settings.require_tee} fresh_state={settings.fresh_state} "
        f"use_encryption={settings.use_encryption}"
    )
    logger.info(
        f"Inference settings — model_owner={inference.model_owner} "
        f"model_dataset={inference.model_dataset} model_size={inference.model_size} "
        f"logs_dataset={inference.logs_dataset}"
    )

    logger.info("Building SyftEnclaveClient...")
    client = SyftEnclaveClient.for_enclave(
        email=settings.email,
        token_path=settings.token_path,
        data_owners=settings.data_owners,
        encryption=settings.use_encryption,
    )
    logger.info("SyftEnclaveClient ready")

    logger.info("Building EnclaveRunner...")
    runner = EnclaveRunner(
        client=client,
        poll_interval=settings.poll_interval,
        require_tee=settings.require_tee,
        fresh_state=settings.fresh_state,
        post_init=lambda: ensure_logs_dataset(client, inference.logs_dataset),
    )
    logger.info("EnclaveRunner ready — calling runner.run()")
    runner.run()


if __name__ == "__main__":
    main()
