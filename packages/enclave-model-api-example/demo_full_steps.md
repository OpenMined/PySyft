# Demo — Option C (deployed enclave) with the **real** Gemma model

End-to-end steps to run `notebooks/demo.ipynb` against a deployed Confidential-Space VM
serving the **real** Gemma 3 270m model (not the mock).

All terminal commands run from this directory (`packages/enclave-model-api-example`).

Fixed values used below:

| role                    | email                             | token                                  |
| ----------------------- | --------------------------------- | -------------------------------------- |
| enclave                 | `beach.do.008@gmail.com`          | `../../credentials/token_enclave.json` |
| DO1 (model + log owner) | `koenlennartvanderveen@gmail.com` | `../../credentials/token_do.json`      |
| DO2 (researcher)        | `koen@openmined.org`              | `../../credentials/token_ds.json`      |

VM name: `enclave-inference-demo-vm` · `project_id`/`zone`/`data_owners` are in
`~/.syft-enclaves/settings.json` (from `just init`); the SA + secret are provisioned in step 2b.

---

## 0. Prerequisite — get the real weights (once)

DO1 needs the Gemma 3 270m **flax** weights on this machine. Download them from Kaggle:

```bash
uv run python -c "import kagglehub; print(kagglehub.model_download('google/gemma-3/flax/gemma-3-270m-it'))"
```

`kagglehub` caches under `~/.cache/kagglehub/` and prints the download path, which lands at:

```
~/.cache/kagglehub/models/google/gemma-3/flax/gemma-3-270m-it/1/
├── tokenizer.model            # SentencePiece tokenizer (root)
└── gemma-3-270m-it/           # the single Orbax checkpoint subdir
```

That top-level `/1` directory is exactly the layout the backend expects — pass it to the
notebook as `GEMMA_WEIGHTS_DIR`.

## 1. Reset state

```bash
just inference-reset ../../credentials/token_enclave.json \
                     ../../credentials/token_do.json \
                     ../../credentials/token_ds.json
```

## 2. Build + push the inference image (skip if already current on Docker Hub)

```bash
just inference-build-push
```

## 2b. Provision the service account + secret (once)

`inference-start` reads `sa_email`/`secret_resource` from settings.json. Provision them with
**demo-scoped** names so teardown (`inference-destroy`, which defaults to these names) never
touches a secret/SA shared with other enclaves:

```bash
just provision-secret-sa ../../credentials/token_enclave.json \
     syft-enclave-token-inference-demo syft-enclave-sa-inference-demo
```

(Skip if `~/.syft-enclaves/settings.json` already has `sa_email` + `secret_resource`.)

## 3. Start the deployed enclave with the real model

`use_mock=false` is the new last argument — it flips the VM to load real weights instead of
serving canned responses:

```bash
just inference-start beach.do.008@gmail.com koenlennartvanderveen@gmail.com \
     enclave-inference-demo-vm n2d-standard-4 gemma3_model 270m inference_logs "" false
```

Get the VM's external IP:

```bash
just get-ip enclave-inference-demo-vm
```

> The enclave starts **before** the weights exist — its inference service comes up and polls for
> the `gemma3_model` dataset. DO1 uploads the weights from inside the notebook (step 4), the
> enclave syncs them over Drive, and only then loads the model. That's why we start the enclave
> first and upload second.

## 4. Run the notebook

Point the notebook at the VM and your local weights, then open it:

```bash
export ENCLAVE_URL="http://$(just get-ip enclave-inference-demo-vm):8080"
export GEMMA_WEIGHTS_DIR="$HOME/.cache/kagglehub/models/google/gemma-3/flax/gemma-3-270m-it/1"   # from step 0
jupyter lab notebooks/demo.ipynb               # or: code notebooks/demo.ipynb
```

Run the cells top to bottom (the notebook also reads these two env vars). It will:

1. Connect to the enclave and confirm `model_loaded: false` (weights not uploaded yet).
2. Log in DO1 + DO2 and peer them with the enclave.
3. **DO1 uploads + privately shares the real Gemma weights** with the enclave (section 3).
4. Re-run the status cell (section 4) until `model_loaded` is `true` — the enclave has synced
   and loaded the real model. (This can take a few minutes for real weights.)
5. Call `/infer` with 3 prompts, submit the bio-weapon-count job, both DOs approve, result prints.

## 5. Tear down

```bash
# in the notebook: the last cell deletes all syftbox/Drive state
just inference-destroy        # deletes the VM + demo-scoped secret/SA
```

---

### Notes

- **The mock variant** (local docker, no weights) is preserved as `notebooks/demo_mock_model.ipynb`.
- `use_mock` is also available on `inference-start-debug` (as the last argument) if you need an
  SSH-enabled VM to inspect model loading via serial logs.
- If `/infer` returns `503 Model not loaded yet`, the weights are still syncing — wait and re-run
  the status cell.
