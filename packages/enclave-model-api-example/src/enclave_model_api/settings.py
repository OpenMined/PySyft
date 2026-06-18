"""Inference-only settings, read from the same ``SYFT_ENCLAVE_*`` env vars.

The generic enclave runtime config lives in
:class:`syft_enclaves.settings.EnclaveSettings`. These extra fields are specific
to the inference image and are kept out of that class so ``syft-enclave`` stays
free of any inference concepts. Both classes share the ``SYFT_ENCLAVE_`` prefix
and ignore each other's fields (``extra="ignore"``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class InferenceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SYFT_ENCLAVE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    use_mock_model: bool = Field(
        default=True,
        description=(
            "Serve canned mock responses instead of loading a real model. "
            "Enabled by default so the demo runs without uploading weights; "
            "set false to load the real Gemma model from the weights dataset."
        ),
    )
    model_owner: str = Field(
        description=(
            "Email of the data owner who uploads the model weights dataset. "
            "Required for the inference image."
        ),
    )
    model_dataset: str = Field(
        default="gemma3_model",
        description="Name of the model-weights dataset shared by model_owner.",
    )
    model_size: Literal["270m", "1b", "4b"] = Field(
        default="270m",
        description="Gemma 3 model size to load from the weights dataset.",
    )
    logs_dataset: str = Field(
        default="inference_logs",
        description=(
            "Name of the dataset on the enclave's own datasite that collects "
            "inference request logs. Its private data never leaves the enclave."
        ),
    )
