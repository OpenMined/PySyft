# Running the double-blind eval demo

A benchmark owner and a model owner each upload half of an evaluation into a Tinfoil enclave. Both
approve the job, the enclave runs it, and only the benchmark owner reads the results. The two
notebooks in this folder are the two parties; everything else on this page is the operator's job,
done once before either notebook is opened.

## What you need

Three Google accounts: one for the enclave, one per party. The enclave reaches Drive with a token
you register as a Tinfoil secret, and each party authenticates through Colab.

You also need the `tinfoil` CLI and an admin API key from the Tinfoil dashboard under **Settings →
API Keys → Admin keys**. Log in once with `tinfoil login --api-key admin_...` and check it with
`just tinfoil-whoami`. Deploying needs that key; publishing a release does not.

Run every `just` command below from `packages/syft-enclave/`.

## 1. Reset both parties' state

Wipe each data owner's SyftBox before you deploy, not after. The enclave caches its peers' Drive
folders when it boots, so clearing them afterwards leaves it peered with folders that no longer
exist.

```bash
just delete-syftbox benchmark_owner@openmined.org ../../credentials/token_benchmark_owner.json
just delete-syftbox model_owner@openmined.org ../../credentials/token_model_owner.json
```

Without a token file to hand, each notebook has a commented cell under Step 0 that calls
`delete_syftbox()` from Colab instead.

The enclave needs no reset: Tinfoil Containers have no persistent disk, and the enclave wipes its
own state on every boot.

## 2. Deploy the published release

Three releases of [`OpenMined/syft-enclave-tinfoil`](https://github.com/OpenMined/syft-enclave-tinfoil)
are already published. They run the same image:

| Release   | Config                        | Receipts | Enclave                                 |
| --------- | ----------------------------- | -------- | --------------------------------------- |
| `v0.1.27` | `tinfoil-config.yml`          | off      | 2 CPUs, 16 GB                           |
| `v0.1.28` | `tinfoil-config-receipts.yml` | on       | 2 CPUs, 16 GB                           |
| `v0.1.29` | `tinfoil-config-gpu.yml`      | on       | 16 CPUs, 64 GB, 1 GPU, Gemma 4 12B pack |

These notebooks run Gemma 4 12B on the GPU, so deploy `v0.1.29` as it stands. `v0.1.28` is the CPU
release, on which the earlier TinyLlama-1.1B version of this demo ran. Skip to step 5 only if you
changed the enclave image or its config.

```bash
just --set tinfoil_container syft-enclave-gpu tinfoil-deploy v0.1.29 enclave@openmined.org "--mark-latest false"
```

The container is named `syft-enclave-gpu`, so it sits next to the CPU one instead of replacing it.
`--mark-latest false` leaves `v0.1.28` marked as the repository's latest release.

All three configs pin `benchmark_owner@openmined.org` and `model_owner@openmined.org` as
`SYFT_ENCLAVE_DATA_OWNERS`, which is what makes a job wait for two approvals instead of one. The
value is measured, so each party's attestation checks it against the release. Other parties need a
config with their emails, and so a new release.

## 3. Check the enclave is attesting

```bash
just tinfoil-attest syft-enclave-gpu.openmined.containers.tinfoil.dev
```

If it fails, run `just tinfoil-why` first — the control plane's `error_message` has named the cause
in every failure so far. [`docs/tinfoil_troubleshooting.md`](../../../packages/syft-enclave/docs/tinfoil_troubleshooting.md)
covers the rest.

## 4. Run the two notebooks

Open both in Colab, one per account:

- [`1. DO-benchmark-owner-dbe.ipynb`](1.%20DO-benchmark-owner-dbe.ipynb) — uploads the prompts,
  submits the job, reads the results, and logs the receipt on Rekor
- [`2. DO-model-owner-dbe.ipynb`](2.%20DO-model-owner-dbe.ipynb) — uploads the adapter, approves the
  job, sees no results

Set the same five constants in both, in the cell under **Setup**:

```python
ENCLAVE_EMAIL         = "enclave@openmined.org"
BENCHMARK_OWNER_EMAIL = "benchmark_owner@openmined.org"
MODEL_OWNER_EMAIL     = "model_owner@openmined.org"
TINFOIL_REPO = "OpenMined/syft-enclave-tinfoil"
TINFOIL_TAG  = "v0.1.29"
IMAGE_DIGEST = "sha256:ea890c9b82dbf3c80c703d717dabe2c3db0c48b53bd269801984807732446864"
```

`TINFOIL_TAG` and `IMAGE_DIGEST` are what the parties check the enclave against, so they must match
the release you deployed. The digest above is the one all three releases pin; after a republish,
use the one `just tinfoil-build` printed.

Then run both notebooks top to bottom. They wait on each other four times, and a card in the
notebook says so each time. Cells that wait print a 🟠 line and tell you to re-run them.

The evaluation takes about three and a half minutes on the enclave's H200 (209 seconds in a test
run of this config): the job installs PyTorch's CUDA build, loads the base model from the mounted
pack, and generates. The GPU config raises the job limit from 600 to 1,800 seconds, so there is
room to spare.

## 5. Republish, after changing the image or config

Everything in the configs under `tinfoil/` is measured, so any edit to one — or to the enclave image
— needs a new release before it can be deployed.

```bash
just tinfoil-build v0.1.23                                          # build, push, pin the digest in both CPU configs
just tinfoil-release v0.1.23                                        # receipts off
just tinfoil-release v0.1.24 tinfoil/tinfoil-config-receipts.yml    # receipts on
just tinfoil-release v0.1.25 tinfoil/tinfoil-config-gpu.yml         # receipts on, GPU
just tinfoil-deploy v0.1.24 enclave@openmined.org
```

`tinfoil-build` pins the new digest in `tinfoil-config.yml` and `tinfoil-config-receipts.yml` only;
put it in `tinfoil-config-gpu.yml` by hand before releasing that one.

Each `tinfoil-release` opens a pull request on the config repo and waits until you merge it. Then it
publishes, which takes about a minute to compute the measurement. Keep the digest `tinfoil-build`
printed, and put it in both notebooks along with the tag you deployed.

A release is a signed GitHub release and a transparency-log entry, so it cannot be unpublished.
Number versions with that in mind. To go back to an earlier one without moving "latest":

```bash
just tinfoil-relaunch v0.1.21 --promote-release=false
```

## The GPU and the model

The demo runs Gemma 4 12B (`google/gemma-4-12B-it`) on one GPU. `tinfoil-config-gpu.yml` gives the
enclave the GPU and mounts the base model's weights read-only as a Tinfoil model pack, which Tinfoil
checks against the root hash in the config at boot. The job finds the pack by that hash and does not
download the weights. The model owner uploads a rank-16 LoRA adapter for it
(`sandepaAI/sandepaAI_gemma4_coder_12b`, pinned to one commit). The CPU releases (`v0.1.27`,
`v0.1.28`) cannot run this job: it stops at once if it sees no GPU or no pack.

## Further reading

- [`docs/tinfoil_deployment.md`](../../../packages/syft-enclave/docs/tinfoil_deployment.md) — the
  deployment mechanics in full
- [`docs/security.md`](../../../packages/syft-enclave/docs/security.md) §6 — what the attestation
  proves
- [`OpenMined/double-blind-eval-bench`](https://github.com/OpenMined/double-blind-eval-bench) — the
  prompt set the benchmark owner downloads
