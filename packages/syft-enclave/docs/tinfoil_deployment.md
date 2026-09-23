# Tinfoil Deployment

An alternative to [Confidential Spaces](./terraform_cs.md) with no GCP involved. The enclave runs in an AMD SEV-SNP or Intel TDX confidential VM managed by [Tinfoil](https://docs.tinfoil.sh), and a data owner verifies it against a measurement published in a Sigstore-signed GitHub release.

Two repositories are in play:

- **this one** builds and pushes the enclave image to Docker Hub;
- **[`OpenMined/syft-enclave-tinfoil`](https://github.com/OpenMined/syft-enclave-tinfoil)** holds the measured `tinfoil-config.yml`, and its GitHub releases publish the expected launch measurement.

The canonical copy of that config lives here at [`tinfoil/tinfoil-config.yml`](../tinfoil/tinfoil-config.yml), so the image and the config that pins it are reviewed together; `just tinfoil-release` syncs it over. A second copy, [`tinfoil/tinfoil-config-receipts.yml`](../tinfoil/tinfoil-config-receipts.yml), differs only in turning signed receipts on (see [Receipts](#receipts)). The release workflows live only in the config repo — nothing in PySyft runs them.

Run all commands from `packages/syft-enclave/`.

When something fails, see [Tinfoil Troubleshooting](./tinfoil_troubleshooting.md). For what the
attestation proves — and how Tinfoil's route to it differs from Confidential Spaces' — see
[Security Overview §6](./security.md#6-what-attestation-proves-on-each-target). This doc covers the
mechanics, not the guarantees.

## Prerequisites

- The `tinfoil` CLI:
  ```bash
  curl -fsSL https://github.com/tinfoilsh/tinfoil-cli/raw/main/install.sh | sh
  ```
  The installer writes to `/usr/local/bin` and so wants `sudo`. To avoid that,
  grab the release tarball and `install -m 0755 tinfoil ~/.local/bin/tinfoil`.
- A Tinfoil organisation account, with the Tinfoil GitHub App installed on the config repo (that App, not your PAT, is what dispatches the release workflow)
- Docker with buildx, and push access to `docker.io/openminedreleasebot`
- [`just`](https://github.com/casey/just) and `jq`
- For verifying: the optional extra, `uv pip install "syft-enclave[tinfoil]"`

No `gcloud`, no terraform, no GCP project.

## Authentication (read this first)

`tinfoil login` needs an **admin API key**, created in the Tinfoil dashboard under
**Settings → API Keys → Admin keys**. Keys are scoped to a single organisation, so you need an
account in the org that will own the container — there is no way to get one from the CLI.

```bash
tinfoil login --api-key admin_...   # or omit --api-key to be prompted
just tinfoil-whoami
tinfoil logout
```

The key is stored in `~/.tinfoil/config.json` (mode 0600). `TINFOIL_API_KEY` and
`TINFOIL_CONTROLPLANE_URL` override it per command, which is what to use in CI. Every
`just tinfoil-*` recipe refuses to run without one or the other.

Publishing a release does **not** need the key — it is a GitHub Actions run, so
`gh workflow run tinfoil-release.yml -f version=vX.Y.Z` works with nothing but your GitHub auth.
The key is needed for the control-plane operations: `container create`, `deployment update`,
`relaunch`, and the secret store.

## Configure

[`tinfoil/tinfoil-config.yml`](../tinfoil/tinfoil-config.yml) declares everything inside the enclave. Two things about it matter more than the rest.

**Everything in the file is measured.** Its sha256 goes into the CVM's kernel command line, so any edit changes the measurement and needs a new config release. So the data owners are pinned in it, because they are the approval gate and a data owner must be able to check them. The enclave email is not pinned — see the table below.

**Egress defaults to `closed`.** Without the `allowlist` the enclave cannot reach Google Drive at all and will sit there doing nothing. Only exact hostnames work — wildcards and IP literals are both rejected by the schema. The two the code actually needs are `www.googleapis.com` (Drive v3) and `oauth2.googleapis.com` (refreshing the OAuth token). `accounts.google.com` is deliberately absent so an accidental interactive OAuth flow fails loudly.

Note that `networks` is a **map keyed by network name**, not a list of objects — the canonical Go
schema is `map[string]*NetworkSpec`. A container may attach to several networks but at most one of
them may have egress other than `closed`.

| Setting                                                   | Where it lives                        | Verifiable by a data owner? |
| --------------------------------------------------------- | ------------------------------------- | --------------------------- |
| Container image digest                                    | measured config                       | yes                         |
| `SYFT_ENCLAVE_ATTESTATION_PROVIDER`, `SYFT_BOOTSTRAP`     | measured config                       | yes                         |
| CPU / memory / GPU shape                                  | measured config                       | yes                         |
| Egress allowlist, exposed paths                           | measured config                       | yes                         |
| `SYFT_ENCLAVE_DATA_OWNERS`, `SYFT_ENCLAVE_RECEIPTS`       | measured config                       | yes                         |
| `SYFT_ENCLAVE_EMAIL`                                      | `--variable` at deploy                | **no**                      |
| `SYFT_ENCLAVE_REQUIRE_TEE`, `SYFT_ENCLAVE_USE_ENCRYPTION` | `--variable` at deploy                | **no**                      |
| Drive OAuth token                                         | `--secret` (name measured, value not) | n/a                         |

To make any of the deploy-time values verifiable, move them into the config's `env` block — at the cost of one config release per combination.

The Drive token reaches the enclave as `--secret SYFT_ENCLAVE_TOKEN_CONTENT`; register it with your Tinfoil org first (`tinfoil secret --help`). Tinfoil has no Secret Manager equivalent, so `SYFT_BOOTSTRAP=tinfoil` reads that injected env var and writes it to `SYFT_ENCLAVE_TOKEN_PATH` at 0600 before the runner starts.

## GPU deployments

Set `gpus:` in the config and publish a new release. Which shapes are available, and whether GPU changes the evidence format (NVIDIA confidential computing adds its own claims on Confidential Spaces), is not yet confirmed with Tinfoil — check before relying on it.

`cvm-version` pins the CVM base image the measurement is computed against; we track the version
[`tinfoilsh/tinfoil-containers-template`](https://github.com/tinfoilsh/tinfoil-containers-template)
uses, currently `0.14.7`. Its deprecation policy is not documented.

## Quickstart: production

```bash
# 1. Build + push the image and pin its digest in both configs.
just tinfoil-build vX.Y.Z            # prints the digest — keep it

# 2. Release a config: open its PR, wait for you to merge it, then publish.
just tinfoil-build-info              # latest tag + suggested next version
just tinfoil-release vX.Y.Z          # receipts off; ~1 min for the measurement
just tinfoil-release vX.Y.Z+1 tinfoil/tinfoil-config-receipts.yml   # receipts on

# 3. Deploy one of those releases.
just tinfoil-deploy vX.Y.Z enclave@openmined.org

# 4. Check it is attesting at all.
just tinfoil-attest syft-enclave.openmined.containers.tinfoil.dev

# 5. Verify it the way a data owner would.
just tinfoil-verify syft-enclave.openmined.containers.tinfoil.dev \
  --expected-image-digest sha256:... --tag vX.Y.Z
```

Steps 1 and 2 need no Tinfoil API key. `tinfoil-release` works on a local clone of the config repo
(`~/workspace/syft-enclave-tinfoil`, or `SYFT_TINFOIL_REPO_DIR`) with `git` and `gh`, so it needs
only your GitHub auth and not the Tinfoil GitHub App. It dispatches the release workflow, so check
that both runs succeed:

```bash
gh run list --repo OpenMined/syft-enclave-tinfoil --limit 2   # "Release" and "Publish release"
```

Step 3 onwards is where the admin API key becomes mandatory.

Then from a data owner's client, against the evidence the enclave published to Drive:

```python
do.attest_peer(
    ENCLAVE_EMAIL,
    expected_image_digest="sha256:...",
    expected_data_owners=["do1@openmined.org", "do2@openmined.org"],
    expected_email=ENCLAVE_EMAIL,
)
```

Pass the digest `tinfoil-build` printed. All three arguments are required: without them the attestation would prove a genuine enclave booted a signed config, but not that the config pinned the image you reviewed, which datasite the enclave runs as, or who has to approve a job. To skip them on purpose, pass a policy with `allow_unpinned=True`.

The whole data-owner side (peer, attest over Drive, upload a dataset) is scripted:

```bash
# Reset the data owners FIRST, then redeploy — the enclave caches peer Drive
# folders at boot, so wiping state afterwards breaks peering.
uv run python ../enclave-model-api-example/scripts/reset_state.py \
    model_owner@openmined.org=../../credentials/token_model_owner.json
just tinfoil-deploy vX.Y.Z enclave@openmined.org

uv run --project ../.. python scripts/tinfoil_e2e_check.py \
    --enclave-email enclave@openmined.org \
    --do-email model_owner@openmined.org \
    --token ../../credentials/token_model_owner.json \
    --tag vX.Y.Z --expected-image-digest sha256:...
```

That is the check that matters: `just tinfoil-verify` fetches the report over HTTPS, while
`attest_peer` reads it from `SYFT_version.json` on Drive, which is the path the syft flow
actually uses.

Prefer `--tag` over the default "latest release" where you can. Unpinned, the release digest is fetched over the network, and a hostile source could substitute the digest of another _legitimately signed_ release of the same repo — a rollback. A pinned tag makes the signature policy require that exact tag.

## Receipts

A release of `tinfoil-config-receipts.yml` writes a signed `receipt.dsse.json` into every finished
job's outputs. The receipt names the code, the dataset file hashes, the data owners, the results and
the run. The enclave signs it with its attested identity key, and the submitter can log it on Rekor
with `upload_to_rekor`. [`tinfoil/CLAUDE.md`](../tinfoil/CLAUDE.md) explains why the receipt can
be trusted.

## Teardown

```bash
just tinfoil-relaunch vX.Y.Z-1 --promote-release=false   # roll back without changing "latest"
tinfoil container stop syft-enclave                      # stop, keep the record
tinfoil container delete syft-enclave                    # remove it
```

A **config release cannot be unpublished** — it is a signed GitHub release and a transparency-log
entry, permanent by design. Plan version numbers accordingly; there is no undo. Nothing is lost by
deleting a container: Tinfoil Containers have no persistent disk, so enclave state is gone on every
restart regardless.

## Dev

There is no control-plane log API, so **logs require a debug instance**:

```bash
just tinfoil-ssh-key koen-debug                    # once per machine
just tinfoil-debug v0.1.8 enclave@openmined.org    # redeploy with SSH
just tinfoil-logs 100                              # container logs
just tinfoil-shell                                 # shell in the enclave
just tinfoil-why                                   # status, error_message, shim boot stages
```

A debug instance is a **separate deployment** at `<name>.debug.<org>.containers.tinfoil.dev` and
deliberately **does not pass attestation** — `attest_peer` and Tinfoil's own `SecureClient` will
refuse it, because a debug enclave is not confidential. Never point real data at one.

Going back to production is not `--debug false`: while an SSH key is attached the container stays
in debug mode. Delete and recreate:

```bash
tinfoil container delete syft-enclave
just tinfoil-deploy v0.1.8 enclave@openmined.org
```

`just tinfoil-why` is the first thing to run on any failure — `error_message` from the control
plane named the cause immediately in every failure we hit, and the shim's `/health` reports
per-stage boot status (config, network, identity, cpu-attestation, certificate, firewall,
containers) until the workload takes over the port.

Other knobs:

- `--staging` / `--promote-release=false` to deploy a release without marking it latest.
- `dummy-attestation: true` in the config for a non-confidential host. Never in production: it defeats the entire point, and the choice is visible in the published config.
- Locally, without Tinfoil at all: bind-mount a fake mount and let auto-detection find it.
  ```bash
  mkdir -p /tmp/fake-tinfoil
  curl -s https://inference.tinfoil.sh/.well-known/tinfoil-attestation > /tmp/fake-tinfoil/attestation.json
  docker run --rm -v /tmp/fake-tinfoil:/tinfoil:ro ... docker.io/openminedreleasebot/syft-enclave:dev
  ```
  The enclave will publish that document as its own evidence; verification against our config repo will of course fail the measurement check. Useful for exercising the publish path, not the verify path.
- `SYFT_ENCLAVE_ATTESTATION_PROVIDER=none` disables attestation entirely.

## Formatting and validation

```bash
just tinfoil-config-get --raw > /tmp/published.yml   # what the repo currently holds
diff /tmp/published.yml tinfoil/tinfoil-config.yml
```

The CLI validates the config against the canonical schema when opening the PR and again in the
release workflow. To check it yourself before either (needs Go):

```bash
go run github.com/tinfoilsh/tinfoil-config/cmd/tinfoil-config@latest tinfoil/tinfoil-config.yml
```

It reports schema and policy violations and exits non-zero — worth running after any edit, since a
release is permanent.

To re-derive a measurement independently, take `tinfoil-deployment.json` from the release, confirm its sha256 equals the release's `tinfoil.hash` (and that the Sigstore DSSE signed that same digest), then run `sev-snp-measure` / `tdx-measure` over the firmware, kernel, initrd and `cmdline` it records. That file also embeds the full config, base64-encoded under `config` — which is how the verifier reads the image digest without trusting whoever served the file.
