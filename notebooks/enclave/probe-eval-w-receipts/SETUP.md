# Running the probe evaluation demo

A probe owner trains a linear probe on an open base model, then applies it inside a Tinfoil enclave
to that base model plus a model owner's private LoRA adapter. Both approve the job, the enclave runs
it, and only the probe owner reads the results: one verdict per prompt, and a signed receipt. The
probe owner never sees the adapter, and the model owner never sees the probe or the prompts.

This is a variant of [`double-blind-eval-w-receipts`](../double-blind-eval-w-receipts/SETUP.md),
which carries out the [tinfoilsh/double-blind-eval](https://github.com/tinfoilsh/double-blind-eval)
flow over PySyft: the same enclave, release and flow, with a job that reads the model's hidden state
and applies a probe instead of generating completions and sending them to a safety classifier. The notebooks in
this folder are the two parties; everything else on this page is the operator's job, done once
before either party opens one.

| Notebook                             | Run by      | Where                                           |
| ------------------------------------ | ----------- | ----------------------------------------------- |
| `0. probe-owner-train-on-base.ipynb` | probe owner | locally or in Colab, with no account            |
| `1. DO-probe-owner.ipynb`            | probe owner | Colab, logged in as the benchmark-owner account |
| `2. DO-model-owner-probe.ipynb`      | model owner | Colab, logged in as the model-owner account     |

The probe owner uses the benchmark-owner account, `benchmark_owner@openmined.org`. The release below
pins that account and `model_owner@openmined.org` as the enclave's data owners, and the pin is
measured, so this variant runs on the release unchanged only by reusing the account. A probe owner
with their own email needs a config with that email, and so a new release (step 6).

## What you need

Three Google accounts: one for the enclave, one per party. The enclave reaches Drive with a token
you register as a Tinfoil secret, and each party authenticates through Colab.

No inference key. The job makes no call outside the enclave except two downloads: the base model
at its pinned commit, and the pinned tools that hash it.

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

Without a token file to hand, notebooks 1 and 2 each have a commented cell under Step 0 that calls
`delete_syftbox()` from Colab instead.

The enclave needs no reset: Tinfoil Containers have no persistent disk, and the enclave wipes its
own state on every boot.

## 2. Deploy the published release

Two releases of [`OpenMined/syft-enclave-tinfoil`](https://github.com/OpenMined/syft-enclave-tinfoil)
are already published. They run the same image, and differ only in whether each job's outputs carry a
signed receipt:

| Release   | Config                        | Receipts |
| --------- | ----------------------------- | -------- |
| `v0.1.27` | `tinfoil-config.yml`          | off      |
| `v0.1.28` | `tinfoil-config-receipts.yml` | on       |

Deploy `v0.1.28` as it stands: the probe owner's notebook ends by checking the receipt and logging it
on Rekor. On `v0.1.27` the job runs the same, but there is no receipt to check.

```bash
just tinfoil-deploy v0.1.28 enclave@openmined.org
```

Both configs pin `benchmark_owner@openmined.org` and `model_owner@openmined.org` as
`SYFT_ENCLAVE_DATA_OWNERS`, which is what makes a job wait for two approvals instead of one. The
value is measured, so each party's attestation checks it against the release.

## 3. Check the enclave is attesting

```bash
just tinfoil-attest syft-enclave.openmined.containers.tinfoil.dev
```

If it fails, run `just tinfoil-why` first — the control plane's `error_message` has named the cause
in every failure so far. [`docs/tinfoil_troubleshooting.md`](../../../packages/syft-enclave/docs/tinfoil_troubleshooting.md)
covers the rest.

## 4. Train the probe

The probe owner runs [`0. probe-owner-train-on-base.ipynb`](0.%20probe-owner-train-on-base.ipynb)
first, on their own machine or in Colab. It needs no account and never touches the enclave. It
downloads the [MLCommons AILuminate](https://github.com/mlcommons/ailuminate) demo prompt set,
leaves out the ten prompts the evaluation uses, reads the open base model's hidden state on a sample
of the rest, and fits a logistic-regression probe. It writes three
files next to itself:

- `probe.json` — the probe. This is the private half: it goes to the enclave and nowhere else. The
  folder's `.gitignore` keeps it out of git.
- `probe_mock.json` — the same fields with random weights, which is all the model owner sees of the
  probe.
- `probe_readout.py` — the code that reads the hidden state and applies the probe. Notebook 1 runs
  it for a dry run on the base model and puts it in the job verbatim.

Notebook 1 reads all three from the folder it runs in. Run both notebooks in the same folder and
there is nothing to do; in Colab, each notebook has a commented cell to download the files from
notebook 0 and upload them to notebook 1.

The concept the probe tracks is a placeholder: a _specialised-advice request_, with the AILuminate
`spc_*` hazards as positives. A real probe owner brings their own concept and prompts by changing
`CONCEPT`, `POSITIVE_HAZARDS` and the download in notebook 0.

Notebook 0 runs the base model in bfloat16 on CPU, as the enclave does: one forward pass per
prompt. The demo set has 100 `spc_*` prompts, 97 once the evaluation prompts are left out, so the
default sample is those 97 and three negatives for each: 388 prompts. On an Apple-silicon laptop
they took about 12 minutes, 1.9 seconds each; Colab's two CPUs are slower. The notebook times the
first five prompts and prints an estimate before starting the rest. Set `MAX_PROMPTS` for a quicker
run on a stratified subsample. The dry run in notebook 1 loads the same 2.2 GB base model and took
about 13 seconds for its five prompts.

## 5. Run the two party notebooks

Open both in Colab, one per account:

- [`1. DO-probe-owner.ipynb`](1.%20DO-probe-owner.ipynb) — uploads the prompts and the probe, dry
  runs the probe on the base model, submits the job, reads the verdicts, compares them with the dry
  run, and logs the receipt on Rekor
- [`2. DO-model-owner-probe.ipynb`](2.%20DO-model-owner-probe.ipynb) — uploads the adapter, reviews
  the job and the probe's public card (concept, layer, position, threshold; no weights), approves,
  sees no results

Set the same constants in both, in the cell under **Setup**:

```python
ENCLAVE_EMAIL         = "enclave@openmined.org"
BENCHMARK_OWNER_EMAIL = "benchmark_owner@openmined.org"
MODEL_OWNER_EMAIL     = "model_owner@openmined.org"
TINFOIL_REPO = "OpenMined/syft-enclave-tinfoil"
TINFOIL_TAG  = "v0.1.28"
IMAGE_DIGEST = "sha256:ea890c9b82dbf3c80c703d717dabe2c3db0c48b53bd269801984807732446864"
```

These are the values of the double-blind-eval notebooks, unchanged: `BENCHMARK_OWNER_EMAIL` is the
probe owner's account. `TINFOIL_TAG` and `IMAGE_DIGEST` are what the parties check the enclave
against, so they must match the release you deployed. The digest above is the one both `v0.1.27`
and `v0.1.28` pin; after a republish, use the one `just tinfoil-build` printed.

Then run both notebooks top to bottom. They wait on each other four times, and a card in the
notebook says so each time. Cells that wait print a 🟠 line and tell you to re-run them.

### Runtime, and the enclave's limits

The enclave has 2 CPUs, 16 GB of RAM shared with its ramdisk, and no GPU, and it kills a job at 600
seconds. The job is built for that: it installs the CPU-only PyTorch wheel, loads the base model in
bfloat16, and runs one forward pass per prompt, with no generation. Its dependency list is the
double-blind-eval job's, unchanged.

We have not yet timed this job in an enclave. The double-blind-eval job on `v0.1.28` took 102
seconds from start to finish, by its receipt's timestamps. About 58 of those were generation. The
other 44 installed PyTorch, downloaded, hashed and loaded the base model, and called the safety
classifier. This job keeps all of that but the classifier, and replaces generation with one forward
pass per prompt, which costs about what the time to first token measured there: 20 seconds for the
five private prompts (on the laptop above, the job's readout of the same five took 13). So we expect
about 65 seconds, well inside the 600-second limit.

## 6. Republish, after changing the image or config

This variant needs none of this on `v0.1.28`. Everything in `tinfoil/tinfoil-config.yml` is
measured, so any edit to it — or to the enclave image — needs a new release before it can be
deployed.

```bash
just tinfoil-build v0.1.23                                          # build, push, pin the digest in both configs
just tinfoil-release v0.1.23                                        # receipts off
just tinfoil-release v0.1.24 tinfoil/tinfoil-config-receipts.yml    # receipts on
just tinfoil-deploy v0.1.24 enclave@openmined.org
```

Each `tinfoil-release` opens a pull request on the config repo and waits until you merge it. Then it
publishes, which takes about a minute to compute the measurement. Keep the digest `tinfoil-build`
printed, and put it in both notebooks along with the tag you deployed.

A release is a signed GitHub release and a transparency-log entry, so it cannot be unpublished.
Number versions with that in mind. To go back to an earlier one without moving "latest":

```bash
just tinfoil-relaunch v0.1.21 --promote-release=false
```

## A larger base model

The demo runs TinyLlama-1.1B on CPU because `tinfoil-config.yml` sets `gpus: 0`. A larger base model
needs a GPU, which means raising `gpus:` in the config, a config change, so step 6. The probe has to
be trained again on that base model, in notebook 0, and its commit pinned as `BASE_MODEL_REVISION`
in notebooks 0 and 1: the job refuses a probe trained on a different base model or commit.

## Further reading

- [`docs/tinfoil_deployment.md`](../../../packages/syft-enclave/docs/tinfoil_deployment.md) — the
  deployment mechanics in full
- [`docs/security.md`](../../../packages/syft-enclave/docs/security.md) §6 — what the attestation
  proves
- [`OpenMined/double-blind-eval-bench`](https://github.com/OpenMined/double-blind-eval-bench) — the
  prompt set the probe owner downloads
