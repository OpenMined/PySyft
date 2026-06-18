# enclave-model-api-example

An example LLM inference API packaged on top of a [`syft-enclave`](../syft-enclave).
It runs a small FastAPI server (`POST /infer`, `GET /model-status`) alongside the
regular enclave runner inside a Confidential Spaces TEE.

This package is a **consumer** of `syft-enclave`, not part of it: the generic enclave
runtime knows nothing about inference. Everything inference-specific — the Gemma backend,
the FastAPI surface, the docker image, and the build/deploy recipes — lives here.

## Layout

- `src/enclave_model_api/` — inference service: `backend.py` (Gemma/JAX, docker-only),
  `mock_backend.py` (canned responses, the default), `service.py`, `server.py`, `paths.py`,
  `log_writer.py`, `logs_dataset.py`, `settings.py` (`InferenceSettings`), and `__main__.py`
  (the runner entry point).
- `docker/` — the inference image: `Dockerfile` (built on the base enclave image),
  `entrypoint.sh`, `inference_server.py` (combined attestation + inference app),
  `requirements.txt` (the docker-only `gemma` dependency).
- `scripts/` — `reset_state.py` (wipe datasites) and `local_server.py` (run the server
  without docker).
- `notebooks/demo.ipynb` — the end-to-end demo (see below).
- `tests/` — unit/integration tests using a stub backend (no real weights needed).
- `Justfile` — build / run / push / deploy recipes (imports shared helpers from
  `../syft-enclave/Justfile`).

## Demo

The demo (`notebooks/demo.ipynb`) drives three datasites: the **enclave** (runs the model
API), **DO1** (model + log owner), and **DO2** (submits an analysis job). DO2 calls `/infer`
a few times, then submits a job that counts how often a banned topic ("bio-weapon") appeared —
which **both** data owners must approve before it runs on the enclave's private logs.

By default the model is **mocked** (`SYFT_ENCLAVE_USE_MOCK_MODEL=true`) so no weights are
needed; set `USE_MOCK = False` in the notebook for the real Gemma 3 model (DO1 then uploads
flax weights via syftbox).

> All commands below are run from this directory (`packages/enclave-model-api-example`).

### Step 1 — Reset state

Wipe all three datasites so everyone starts clean and re-publishes a fresh version file
(this avoids a peer-version handshake race during peering):

```bash
just inference-reset ../../credentials/token_enclave.json \
                     ../../credentials/token_do.json \
                     ../../credentials/token_ds.json
```

### Step 2 — Start the enclave (pick one)

**Option A — Local, no docker** (fastest for development). Runs the enclave runner +
inference server as local background processes on `localhost:8080`:

```bash
just inference-local-serve ../../credentials/token_enclave.json
# ... when done: just inference-local-stop
```

Then in the notebook, **skip cell 1** (the `docker run`) — the enclave is already running.

**Option B — Local docker** (what the notebook does by default). Build the image once, then
the notebook's first cell starts the container on `localhost:8080`:

```bash
just inference-local-build
# the notebook's "1. (re)start the enclave container" cell runs `docker run` for you
```

**Option C — Deployed enclave (GCP Confidential Space)** — a real TEE. Build + push the
image, then launch the hardened VM:

```bash
just inference-build-push
just inference-start <enclave_email> <model_owner_email>
```

Point the notebook's inference URL at the VM's IP (`http://<vm-ip>:8080`) instead of
`localhost`, and skip the local `docker run` cell. See the `[inference]` recipes in the
`Justfile` (`inference-start`, `inference-start-debug`) for options.

### Step 3 — Run the notebook

Open `notebooks/demo.ipynb` and run it top-to-bottom (skipping cell 1 for Options A/C as
noted). It logs in the data owners, waits until peering is fully established, calls `/infer`,
submits the analysis job, has both DOs approve it, and prints the result. The final cell
stops the container (Option B) and wipes all state.
