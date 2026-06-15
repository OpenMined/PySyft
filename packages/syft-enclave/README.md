# syft-enclave

Enclave support for syft-client, enabling secure computation in Trusted Execution Environments (TEEs).

## About

- [Collaboration Flow](./docs/flow.md)
- [Security Overview](./docs/security.md)
- [Enclave Architecture](./docs/enclave_architecture.md)
- [API](./docs/api.md)

## Prerequisites

- Docker with buildx support (Docker Desktop includes this)
- `gcloud` [CLI installed](https://docs.cloud.google.com/sdk/docs/install-sdk)
- A GCP project with billing enabled
- [`just`](https://github.com/casey/just) and `jq`

All commands are defined in the [`Justfile`](./Justfile). Run them from this directory.

## One-time setup

```bash
just init YOUR_GCLOUD_PROJECT_ID TOKEN_PATH DATA_OWNERS
```

- `TOKEN_PATH` — credentials of the enclave email downloaded from the gcloud console.
- `DATA_OWNERS` — comma-separated emails of the data owners whose approval gates every job on this enclave, e.g. `do1@openmined.org,do2@openmined.org`.

This stores settings (including `data_owners`) in `~/.syft-enclaves/settings.json` and sets the active gcloud project. Every other recipe reads `project_id`, `zone`, and `data_owners` from this file — zone is **not** a per-call arg. To deploy in a different zone or change the data owners, re-run `just init YOUR_PROJECT_ID TOKEN_PATH DATA_OWNERS europe-west4-a`.

### Approval model

The data owners configured at `init` are fixed for the enclave: a job runs only after **all** of them approve it, regardless of which datasets the submission references. The emails are passed to the VM as `SYFT_ENCLAVE_DATA_OWNERS` at deploy time and held in memory by the running enclave. To change the approving data owners, re-run `just init` and redeploy.

## Production deployment

Hardened image — no SSH access, TEE enforcement enabled.

```bash
just start EMAIL                          # defaults: syft-enclave-vm, n2d-standard-2
just start EMAIL my-vm n2d-standard-4     # override name / machine type
just stop [name]                          # Teardown: Deletes VM and removes firewall rule (default: syft-enclave-vm)
```

The first run also provisions APIs, IAM roles, and firewall rules (idempotent).

## Debug deployment

Debug image — SSH enabled, container logs redirected to serial output.

```bash
just start-debug EMAIL                          # defaults: syft-enclave-vm, n2d-standard-2
just start-debug EMAIL my-vm n2d-standard-4     # override name / machine type
just stop [name]                                # Teardown: Deletes VM and removes firewall rule.
```

## Inspect a running VM

All inspect commands take an optional `name` (default: `syft-enclave-vm`). Zone is always read from `settings.json`.

```bash
# Works on both production and debug
just status [name]   # RUNNING / TERMINATED / etc.
just get-ip [name]   # external IP
just attest [name]   # fetch TEE attestation report

# Debug only
just ssh    [name]   # SSH into the VM (production image disables SSH)
just logs   [name]   # last 50 lines of serial output (production only shows boot logs;
                     # debug redirects container logs to serial output)
```
