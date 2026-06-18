# enclave-model-api-example

An example LLM inference API packaged on top of a [`syft-enclave`](../syft-enclave).
It runs a small FastAPI server (`POST /infer`, `GET /model-status`) alongside the
regular enclave runner inside a Confidential Spaces TEE.

This package is a **consumer** of `syft-enclave`, not part of it: the generic enclave
runtime knows nothing about inference. Everything inference-specific — the Gemma backend,
the FastAPI surface, the docker image, and the build/deploy recipes — lives here.

## Layout

- `src/enclave_model_api/` — inference service: `backend.py` (Gemma/JAX, docker-only),
  `service.py`, `server.py`, `paths.py`, `log_writer.py`, `logs_dataset.py`,
  `settings.py` (`InferenceSettings`), and `__main__.py` (the runner entry point).
- `docker/` — the inference image: `Dockerfile` (built on the base enclave image),
  `entrypoint.sh`, `inference_server.py` (combined attestation + inference app),
  `requirements.txt` (the docker-only `gemma` dependency).
- `tests/` — unit/integration tests using a stub backend (no real weights needed).
- `Justfile` — build / run / push / deploy recipes (imports shared helpers from
  `../syft-enclave/Justfile`).

## Build & run locally

```bash
just inference-local-build
just inference-local-run <email> <data_owners> <model_owner> <token_path>
```

The model weights arrive through syftbox as a private dataset shared by `model_owner`;
inference request logs are written to a dataset on the enclave's own datasite whose
private data never leaves the enclave.
