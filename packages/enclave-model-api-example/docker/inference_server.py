"""Combined FastAPI app for the inference image: attestation + inference.

Reuses the attestation app from the base image (/, /health, /attestation —
attestation_server.py sits next to this file in /app) and mounts the
inference routes on it, so everything serves on the single Confidential
Spaces port (8080).
"""

from attestation_server import app
from syft_enclaves.settings import EnclaveSettings

from enclave_model_api.paths import default_syftbox_folder, private_dataset_dir
from enclave_model_api.server import build_router
from enclave_model_api.service import InferenceService
from enclave_model_api.settings import InferenceSettings

settings = EnclaveSettings()
inference = InferenceSettings()

# Mock by default — no weights, no jax/gemma import. Import the real Gemma
# backend lazily only when explicitly enabled.
if inference.use_mock_model:
    from enclave_model_api.mock_backend import MockBackend

    backend = MockBackend()
else:
    from enclave_model_api.backend import GemmaBackend

    backend = GemmaBackend()

syftbox_folder = default_syftbox_folder(settings.email)
service = InferenceService(
    backend=backend,
    model_size=inference.model_size,
    weights_dir=private_dataset_dir(
        syftbox_folder, inference.model_owner, inference.model_dataset
    ),
    logs_dir=private_dataset_dir(
        syftbox_folder, settings.email, inference.logs_dataset
    ),
)
service.start_polling()
app.include_router(build_router(service))
