# Tinfoil Deployment

An alternative to [Confidential Spaces](./terraform.md) with no GCP involved. The enclave runs in an AMD SEV-SNP or Intel TDX confidential VM managed by [Tinfoil](https://docs.tinfoil.sh), and a data owner verifies it against a measurement published in a Sigstore-signed GitHub release.

Two repositories are in play:

- **this one** builds and pushes the enclave image to Docker Hub;
- **[`OpenMined/syft-enclave-tinfoil`](https://github.com/OpenMined/syft-enclave-tinfoil)** holds the measured `tinfoil-config.yml`, and its GitHub releases publish the expected launch measurement.

The canonical copy of that config lives here at [`tinfoil/tinfoil-config.yml`](../tinfoil/tinfoil-config.yml), so the image and the config that pins it are reviewed together. `just tinfoil-release` syncs it over.

Run all commands from `packages/syft-enclave/`.

## What this proves, and what it does not

**Proves.** The enclave is genuine SEV-SNP/TDX hardware with debug disabled and its firmware TCB at or above minimum; it booted the exact CVM image and config published in a named release of the config repo; that release was signed by GitHub OIDC from that repo's tag via Sigstore; and the container image digest matches the one you pinned.

**Does not prove.** Which email or data owners the enclave was started with. Those are deploy-time `--variable`s, so they are outside the measurement and the attested code merely relays whatever its deployer handed it. Confidential Spaces is in the same position today (`tee-env-*` metadata is not checked either).

**Key binding, and how it is achieved.** A workload cannot inject a nonce into the report: its 64 bytes of user data are the sha256 of the shim's TLS public key followed by its HPKE public key. But that is exactly what makes binding possible — the report *commits to the key terminating a TLS connection to the enclave*. So the client:

1. verifies the report,
2. opens HTTPS to the enclave and checks the certificate it is served carries that same key,
3. checks the enclave signed the client's nonce with the key bundle it served,
4. and then trusts that bundle, which came down the same connection.

No certificate authority is involved anywhere: the enclave's certificate is self-signed, and the *report* is what decides whether to trust it. `attest_peer` then sets those keys for the peer, so the enclave's public keys are no longer an unsigned Drive file. This is the binding `docs/security.md` §5 describes.

It also gets freshness for free: a replayed report commits to a TLS key whose private half lives in an enclave the attacker does not control, so the pin fails.

**Freshness comes from a nonce, not from the report.** A workload cannot influence the report's user data, so the client sends a random nonce and the enclave signs it with the identity key from the bundle it just served. That proves two things the report cannot: the enclave *holds the private half* of the key we are about to encrypt to, and the answer was produced for *this* exchange rather than replayed. The bundle is adopted only when both `key_binding` and `nonce_freshness` pass.

**Drive is not a fallback.** Evidence is still published to `SYFT_version.json` — as provenance, and so the path exists if it is ever needed again — but the client always appraises a Tinfoil enclave from the live API. An unreachable enclave is an error, not a downgrade: accepting the Drive copy would silently mean unbound keys and a replayable report.

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

**Everything in the file is measured.** Its sha256 goes into the CVM's kernel command line, so any edit changes the measurement and needs a new config release. That is why the enclave email and data owners are *not* in it — see the table below.

**Egress defaults to `closed`.** Without the `allowlist` the enclave cannot reach Google Drive at all and will sit there doing nothing. Only exact hostnames work — wildcards and IP literals are both rejected by the schema. The two the code actually needs are `www.googleapis.com` (Drive v3) and `oauth2.googleapis.com` (refreshing the OAuth token). `accounts.google.com` is deliberately absent so an accidental interactive OAuth flow fails loudly.

Note that `networks` is a **map keyed by network name**, not a list of objects — the canonical Go
schema is `map[string]*NetworkSpec`. A container may attach to several networks but at most one of
them may have egress other than `closed`.

| Setting | Where it lives | Verifiable by a data owner? |
|---|---|---|
| Container image digest | measured config | yes |
| `SYFT_ENCLAVE_ATTESTATION_PROVIDER`, `SYFT_BOOTSTRAP` | measured config | yes |
| CPU / memory / GPU shape | measured config | yes |
| Egress allowlist, exposed paths | measured config | yes |
| `SYFT_ENCLAVE_EMAIL`, `SYFT_ENCLAVE_DATA_OWNERS` | `--variable` at deploy | **no** |
| `SYFT_ENCLAVE_REQUIRE_TEE`, `SYFT_ENCLAVE_USE_ENCRYPTION` | `--variable` at deploy | **no** |
| Drive OAuth token | `--secret` (name measured, value not) | n/a |

To make any of the deploy-time values verifiable, move them into the config's `env` block — at the cost of one config release per combination.

The Drive token reaches the enclave as `--secret SYFT_ENCLAVE_TOKEN_CONTENT`; register it with your Tinfoil org first (`tinfoil secret --help`). Tinfoil has no Secret Manager equivalent, so `SYFT_BOOTSTRAP=tinfoil` reads that injected env var and writes it to `SYFT_ENCLAVE_TOKEN_PATH` at 0600 before the runner starts.

## GPU deployments

Set `gpus:` in the config and publish a new release. Which shapes are available, and whether GPU changes the evidence format (NVIDIA confidential computing adds its own claims on Confidential Spaces), is not yet confirmed with Tinfoil — check before relying on it.

`cvm-version` pins the CVM base image the measurement is computed against; we track the version
[`tinfoilsh/tinfoil-containers-template`](https://github.com/tinfoilsh/tinfoil-containers-template)
uses, currently `0.14.7`. Its deprecation policy is not documented.

## Quickstart: production

```bash
# 1. Build + push the image, pin its digest in the config, open the config PR.
just tinfoil-release vX.Y.Z          # prints the digest — keep it

# 2. Merge that PR, then publish the measured, signed release.
just tinfoil-build-info              # latest tag + suggested next version
just tinfoil-publish vX.Y.Z          # ~1 min for the measurement to compute

# 3. Deploy.
just tinfoil-deploy vX.Y.Z enclave@openmined.org do1@openmined.org,do2@openmined.org

# 4. Check it is attesting at all.
just tinfoil-attest syft-enclave.openmined.containers.tinfoil.dev

# 5. Verify it the way a data owner would.
just tinfoil-verify syft-enclave.openmined.containers.tinfoil.dev \
  --expected-image-digest sha256:... --tag vX.Y.Z
```

Steps 1 and 2 need no Tinfoil API key. If the Tinfoil GitHub App is not installed on the config
repo, `tinfoil-publish` will not be able to dispatch — run the workflow directly instead, which
only needs your GitHub auth:

```bash
cd ~/workspace/syft-enclave-tinfoil
gh workflow run tinfoil-release.yml -f version=vX.Y.Z
gh run list --limit 2      # both "Release" and "Publish release" must succeed
```

Step 3 onwards is where the admin API key becomes mandatory.

Then from a data owner's client, against the evidence the enclave published to Drive:

```python
do.attest_peer(ENCLAVE_EMAIL, expected_image_digest="sha256:...")
```

Pass the digest `tinfoil-release` printed. Without it the image-digest check is **skipped**, not failed — the attestation then proves a genuine enclave booted a signed config, but not that the config pinned the image you reviewed.

The whole data-owner side (peer, attest over Drive, upload a dataset) is scripted:

```bash
# Reset the data owners FIRST, then redeploy — the enclave caches peer Drive
# folders at boot, so wiping state afterwards breaks peering.
uv run python ../enclave-model-api-example/scripts/reset_state.py \
    model_owner@openmined.org=../../credentials/token_model_owner.json
just tinfoil-deploy vX.Y.Z enclave@openmined.org model_owner@openmined.org

uv run --project ../.. python scripts/tinfoil_e2e_check.py \
    --enclave-email enclave@openmined.org \
    --do-email model_owner@openmined.org \
    --token ../../credentials/token_model_owner.json \
    --tag vX.Y.Z --expected-image-digest sha256:...
```

That is the check that matters: `just tinfoil-verify` fetches the report over HTTPS, while
`attest_peer` reads it from `SYFT_version.json` on Drive, which is the path the syft flow
actually uses.

Prefer `--tag` over the default "latest release" where you can. Unpinned, the release digest is fetched over the network, and a hostile source could substitute the digest of another *legitimately signed* release of the same repo — a rollback. A pinned tag makes the signature policy require that exact tag.

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
just tinfoil-deploy v0.1.8 enclave@openmined.org do1@x.org,do2@x.org
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

## Troubleshooting

Every row below was hit for real while bringing the first enclave up, in this order.

| Symptom | Cause and fix |
|---|---|
| `Firewall setup failed: creating docker network "default": operation is not permitted on predefined default network` | a network named `default` collides with Docker's predefined one. The canonical schema only reserves `shim-net`, so this **validates fine and fails on the host**. Name it anything else. |
| `FileNotFoundError: No usable temporary directory found in ['/tmp', ...]` and a crash-loop | containers run with a read-only rootfs, and `portalocker` (a syft dependency) calls `tempfile.gettempdir()` at import. Set `read_only: false`. |
| `2 validation errors for EnclaveSettings: email / data_owners Field required` — despite passing `--variable` | the shim only injects variables whose **key the measured config declares**. An undeclared `--variable` is silently dropped. Declare it as a bare name under `env:`. |
| The enclave hangs at "Building SyftEnclaveClient" | OAuth token refresh cannot reach `oauth2.googleapis.com`. See the egress note in `tinfoil-config.yml`: `allowlist` resolves hostnames to IPs once at boot and Google rotates them, so use `egress: open`. |
| `curl: (60) SSL certificate problem: self signed certificate` | expected. The enclave's TLS key is generated inside it and the report commits to that key, so there is no CA. Use `-k` (as `just tinfoil-attest` does) and get your trust from `just tinfoil-verify`. |
| `The server had an error while processing your request.` from the domain | the shim is up but your container is not serving on `upstream-port`. It has probably crashed — `just tinfoil-debug <tag> <email>` then `just tinfoil-logs`. |
| `ValueError: Serialization error: sender fingerprint mismatch` on a data owner's login | with `encryption=True` the keypair is persisted at `<syftbox_folder>/<email>/private/crypto_keys.json`. A fresh `SYFTBOX_FOLDER` mints a new identity that cannot verify that account's own history on Drive. Point at the account's existing folder, or wipe its Drive state first. |
| `upstream port is not set` | the config's `shim.upstream-port` is missing; it is required |
| Config rejected: image must be a digest | `image:` uses a tag; it must be `repo@sha256:...` |
| A path 404s | it is not in `shim.paths`. `/.well-known/tinfoil-attestation` is exempt |
| `Tag vX.Y.Z already exists` | releases are permanent; pick the next version |
| `image_digest` check fails with a 400 from `github-proxy.tinfoil.sh` | expected — the proxy only serves `tinfoil.hash`; the verifier falls through to github.com |
| `MissingOptionalDependency: ... tinfoil` | `uv pip install "syft-enclave[tinfoil]"` |
| `measurement_match` fails | the running enclave is not the release you are verifying against — check the deployed tag with `just tinfoil-status` |
| Peer skipped with "published no attestation evidence" | the enclave booted outside a TEE, or with `attestation_provider=none` |

Redeploying? Delete the accounts' syftboxes **first**, then redeploy — the enclave caches peer Drive folders at boot, so wiping state afterwards breaks the flow.

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
