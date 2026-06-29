"""Run the inference server locally (no docker) — uses create_app, no attestation.

The combined docker app (docker/inference_server.py) mounts onto the base
image's attestation_server; locally we just serve the inference routes via
create_app on the same port (8080). Backend (mock vs real) follows
SYFT_ENCLAVE_USE_MOCK_MODEL, exactly like the docker image.
"""

import os

os.environ.setdefault("PRE_SYNC", "false")

import uvicorn
from syft_enclaves.settings import EnclaveSettings

from enclave_model_api.paths import default_syftbox_folder, private_dataset_dir
from enclave_model_api.server import create_app
from enclave_model_api.service import InferenceService
from enclave_model_api.settings import InferenceSettings

settings = EnclaveSettings()
inf = InferenceSettings()
if inf.use_mock_model:
    from enclave_model_api.mock_backend import MockBackend

    backend = MockBackend()
else:
    from enclave_model_api.backend import GemmaBackend

    backend = GemmaBackend()

folder = default_syftbox_folder(settings.email)
service = InferenceService(
    backend=backend,
    model_size=inf.model_size,
    weights_dir=private_dataset_dir(folder, inf.model_owner, inf.model_dataset),
    logs_dir=private_dataset_dir(folder, settings.email, inf.logs_dataset),
)
service.start_polling()
uvicorn.run(
    create_app(service, use_encryption=settings.use_encryption),
    host="0.0.0.0",
    port=8080,
)
