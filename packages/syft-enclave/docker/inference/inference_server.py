"""Combined FastAPI app for the inference image: attestation + inference.

Reuses the attestation app from the base image (/, /health, /attestation —
attestation_server.py sits next to this file in /app) and mounts the
inference routes on it, so everything serves on the single Confidential
Spaces port (8080).
"""

from attestation_server import app
from syft_enclaves.inference.backend import GemmaBackend
from syft_enclaves.inference.paths import default_syftbox_folder, private_dataset_dir
from syft_enclaves.inference.server import build_router
from syft_enclaves.inference.service import InferenceService
from syft_enclaves.settings import EnclaveSettings

settings = EnclaveSettings()
if settings.model_owner is None:
    raise RuntimeError("SYFT_ENCLAVE_MODEL_OWNER is required for the inference image.")

syftbox_folder = default_syftbox_folder(settings.email)
service = InferenceService(
    backend=GemmaBackend(),
    model_size=settings.model_size,
    weights_dir=private_dataset_dir(
        syftbox_folder, settings.model_owner, settings.model_dataset
    ),
    logs_dir=private_dataset_dir(syftbox_folder, settings.email, settings.logs_dataset),
)
service.start_polling()
app.include_router(build_router(service))
