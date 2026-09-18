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
just delete-syftbox bench@openmined.org ../../credentials/token_bench.json
just delete-syftbox model@openmined.org ../../credentials/token_model.json
```

Without a token file to hand, each notebook has a commented cell under Step 0 that calls
`delete_syftbox()` from Colab instead.

The enclave needs no reset: Tinfoil Containers have no persistent disk, and the enclave wipes its
own state on every boot.

## 2. Deploy the published release

Release `v0.1.14` of [`OpenMined/syft-enclave-tinfoil`](https://github.com/OpenMined/syft-enclave-tinfoil)
is already published, so deploy it as it stands. Skip to step 5 only if you changed the enclave
image or its config.

```bash
just tinfoil-deploy v0.1.14 enclave@openmined.org bench@openmined.org,model@openmined.org
```

Both party emails have to appear in that comma-separated list. It becomes
`SYFT_ENCLAVE_DATA_OWNERS`, which is what makes a job wait for two approvals instead of one.

## 3. Check the enclave is attesting

```bash
just tinfoil-attest syft-enclave.openmined.containers.tinfoil.dev
```

If it fails, run `just tinfoil-why` first — the control plane's `error_message` has named the cause
in every failure so far. [`docs/tinfoil_troubleshooting.md`](../../../packages/syft-enclave/docs/tinfoil_troubleshooting.md)
covers the rest.

## 4. Run the two notebooks

Open both in Colab, one per account:

- [`1. DO-benchmark-owner-dbe.ipynb`](1.%20DO-benchmark-owner-dbe.ipynb) — uploads the prompts,
  submits the job, reads the results
- [`2. DO-model-owner-dbe.ipynb`](2.%20DO-model-owner-dbe.ipynb) — uploads the adapter, approves the
  job, sees no results

Set the same five constants in both, in the cell under **Setup**:

```python
ENCLAVE_EMAIL         = "enclave@openmined.org"
BENCHMARK_OWNER_EMAIL = "bench@openmined.org"
MODEL_OWNER_EMAIL     = "model@openmined.org"
TINFOIL_REPO = "OpenMined/syft-enclave-tinfoil"
TINFOIL_TAG  = "v0.1.14"
IMAGE_DIGEST = "sha256:d0bd57f22af80b9dcd0dc151fb68d89cca65b65fcbbd1d2e4586cdfa9d7daebc"
```

`TINFOIL_TAG` and `IMAGE_DIGEST` are what the parties check the enclave against, so they must match
the release you deployed. The digest above is the one `v0.1.14` pins; after a republish, use the one
`just tinfoil-release` printed.

Then run both notebooks top to bottom. They wait on each other four times, and a card in the
notebook says so each time. Cells that wait print a 🟠 line and tell you to re-run them.

The evaluation itself takes four to six minutes. It installs PyTorch, downloads a base model and
generates on two CPUs, and a job is killed at 600 seconds.

## 5. Republish, after changing the image or config

Everything in `tinfoil/tinfoil-config.yml` is measured, so any edit to it — or to the enclave image
— needs a new release before it can be deployed.

```bash
just tinfoil-build-info              # the latest tag, and a suggested next one
just tinfoil-release v0.1.15         # build, push, pin the digest, open the config PR
```

Merge that pull request, then publish and redeploy:

```bash
just tinfoil-publish v0.1.15         # about a minute to compute the measurement
just tinfoil-deploy v0.1.15 enclave@openmined.org bench@openmined.org,model@openmined.org
```

Keep the digest `tinfoil-release` printed, and put it in both notebooks along with the new tag.

A release is a signed GitHub release and a transparency-log entry, so it cannot be unpublished.
Number versions with that in mind. To go back to an earlier one without moving "latest":

```bash
just tinfoil-relaunch v0.1.14 --promote-release=false
```

## Giving the model a GPU

The demo runs TinyLlama-1.1B on CPU because `tinfoil-config.yml` sets `gpus: 0`. The model owner
notebook carries a commented `ADAPTER_REPO` line for the adapter the
[double-blind-eval](https://github.com/tinfoilsh/double-blind-eval) demo uses, which targets
gemma-4-31B-it and needs a GPU. Switching to it means raising `gpus:` in the config, which is a
config change, so follow step 5.

## Further reading

- [`docs/tinfoil_deployment.md`](../../../packages/syft-enclave/docs/tinfoil_deployment.md) — the
  deployment mechanics in full
- [`docs/security.md`](../../../packages/syft-enclave/docs/security.md) §6 — what the attestation
  proves
- [`OpenMined/double-blind-eval-bench`](https://github.com/OpenMined/double-blind-eval-bench) — the
  prompt set the benchmark owner downloads
