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
image (only needed when the inference code/image changed — `latest` on Docker Hub is reused
otherwise), provision a secret + service account once, then launch the hardened VM under a
distinct name so it never collides with other enclaves in the project:

```bash
just inference-build-push                                  # skip if the image is already current
just init <project_id> ../../credentials/token_enclave.json \
          "<do1_email>,<do2_email>"                        # provisions the shared secret + SA
just inference-start <enclave_email> <model_owner_email> enclave-inference-demo-vm
just get-ip enclave-inference-demo-vm                      # print the VM's external IP
```

The deployed enclave serves the **mock** model by default (`SYFT_ENCLAVE_USE_MOCK_MODEL`
defaults to `true`), so no weights upload is needed. Point the inference URL at the VM's IP
(`http://<vm-ip>:8080`) and skip the local `docker run` cell. See the `[inference]` recipes
in the `Justfile` (`inference-start`, `inference-start-debug`) for options.

> **Isolated resources (recommended on a shared project):** to keep teardown from touching
> a secret/SA shared with other enclaves, provision demo-scoped ones instead of `just init`:
>
> ```bash
> just provision-secret-sa ../../credentials/token_enclave.json \
>      syft-enclave-token-inference-demo syft-enclave-sa-inference-demo
> ```
>
> (`data_owners` still has to be in `~/.syft-enclaves/settings.json` — set it once with
> `just init` or by editing that file.) `inference-destroy` defaults to deleting exactly
> these demo-scoped names.

### Step 3 — Run the notebook

Open `notebooks/demo.ipynb` (for demo) or `scripts/run_demo.py` (for testing) and run it top-to-bottom (skipping cell 1 for Options A/C as
noted). It logs in the data owners, waits until peering is fully established, calls `/infer`,
submits the analysis job, has both DOs approve it, and prints the result. The final cell
stops the container (Option B) and wipes all state.

For Option C, point `run_demo.py` at the VM by setting `ENCLAVE_URL` (it defaults to
`http://localhost:8080`):

```bash
ENCLAVE_URL="http://$(just get-ip enclave-inference-demo-vm):8080" uv run python scripts/run_demo.py
```

### Step 4 — Tear down the deployed enclave (Option C)

When done, delete the VM **and** all its gcloud state — the demo-scoped Secret Manager
secret and service account, including IAM bindings — in one command:

```bash
just inference-destroy
# or, if you used non-default names:
just inference-destroy <vm_name> <secret_name> <sa_name>
```

This does not touch syftbox/Drive state — wipe that separately with `just inference-reset`
(or `just delete-syftbox <enclave_email>`).
