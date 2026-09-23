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

Two releases of [`OpenMined/syft-enclave-tinfoil`](https://github.com/OpenMined/syft-enclave-tinfoil)
are already published. They run the same image, and differ only in whether each job's outputs carry a
signed receipt:

| Release   | Config                        | Receipts |
| --------- | ----------------------------- | -------- |
| `v0.1.17` | `tinfoil-config.yml`          | off      |
| `v0.1.18` | `tinfoil-config-receipts.yml` | on       |

Deploy one of them as it stands. Skip to step 5 only if you changed the enclave image or its config.

```bash
just tinfoil-deploy v0.1.18 enclave@openmined.org
```

Both configs pin `benchmark_owner@openmined.org` and `model_owner@openmined.org` as
`SYFT_ENCLAVE_DATA_OWNERS`, which is what makes a job wait for two approvals instead of one. The
value is measured, so each party's attestation checks it against the release. Other parties need a
config with their emails, and so a new release.

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
  submits the job, reads the results, and logs the receipt on Rekor
- [`2. DO-model-owner-dbe.ipynb`](2.%20DO-model-owner-dbe.ipynb) — uploads the adapter, approves the
  job, sees no results

Set the same five constants in both, in the cell under **Setup**:

```python
ENCLAVE_EMAIL         = "enclave@openmined.org"
BENCHMARK_OWNER_EMAIL = "benchmark_owner@openmined.org"
MODEL_OWNER_EMAIL     = "model_owner@openmined.org"
TINFOIL_REPO = "OpenMined/syft-enclave-tinfoil"
TINFOIL_TAG  = "v0.1.18"
IMAGE_DIGEST = "sha256:49314ceb202d47e48a851670e2e8aaaa470eca95cb4496bb79cfab672901a0c2"
```

`TINFOIL_TAG` and `IMAGE_DIGEST` are what the parties check the enclave against, so they must match
the release you deployed. The digest above is the one both `v0.1.17` and `v0.1.18` pin; after a
republish, use the one `just tinfoil-build` printed.

Then run both notebooks top to bottom. They wait on each other four times, and a card in the
notebook says so each time. Cells that wait print a 🟠 line and tell you to re-run them.

The evaluation takes a little over two minutes, measured on a run against this release: the enclave
installs PyTorch, downloads the base model, and generates on two CPUs. A job is killed at 600
seconds, so there is room to spare.

## 5. Republish, after changing the image or config

Everything in `tinfoil/tinfoil-config.yml` is measured, so any edit to it — or to the enclave image
— needs a new release before it can be deployed.

```bash
just tinfoil-build v0.1.19                                          # build, push, pin the digest in both configs
just tinfoil-release v0.1.19                                        # receipts off
just tinfoil-release v0.1.20 tinfoil/tinfoil-config-receipts.yml    # receipts on
just tinfoil-deploy v0.1.20 enclave@openmined.org
```

Each `tinfoil-release` opens a pull request on the config repo and waits until you merge it. Then it
publishes, which takes about a minute to compute the measurement. Keep the digest `tinfoil-build`
printed, and put it in both notebooks along with the tag you deployed.

A release is a signed GitHub release and a transparency-log entry, so it cannot be unpublished.
Number versions with that in mind. To go back to an earlier one without moving "latest":

```bash
just tinfoil-relaunch v0.1.17 --promote-release=false
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
