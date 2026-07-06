# Terraform Deployment

A declarative alternative to the gcloud-based Justfile recipes (`init` / `provision-secret-sa` / `start` / `start-debug`). Terraform manages the full stack in one `apply`:

- Required GCP APIs (Compute, Confidential Computing, Secret Manager, IAM)
- The enclave service account and its IAM roles
- The Secret Manager secret **including the token version**
- The Confidential Space VM (AMD SEV)

State is **local** (`terraform/terraform.tfstate`, gitignored): one operator = one state file = one enclave deployment. The `.tf` configuration is shared via git; your per-deployment values live in a gitignored `terraform.tfvars`. The gcloud recipes remain available — but don't manage the _same_ resources with both flows.

## Prerequisites

- [Terraform](https://developer.hashicorp.com/terraform/install) ≥ 1.5
- `gcloud` CLI (used for authentication and by `just tf-logs` / `tf-ssh` / `tf-attest`)
- [`just`](https://github.com/casey/just)
- A GCP project with billing enabled

Run all commands from `packages/syft-enclave/`.

## Authentication (read this first)

The Terraform Google provider does **not** use your `gcloud auth login` credentials — it uses Application Default Credentials (ADC):

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

## Configure

```bash
cp terraform/terraform.tfvars.example terraform/terraform.tfvars
```

Then fill in:

| Variable        | Meaning                                                                 |
| --------------- | ----------------------------------------------------------------------- |
| `project_id`    | GCP project ID (the ID, not the display name — `gcloud projects list`). |
| `zone`          | Zone for the VM (default `us-central1-a`).                              |
| `enclave_email` | The enclave datasite's email (`SYFT_ENCLAVE_EMAIL`).                    |
| `data_owners`   | List of data-owner emails; **all** must approve every job.              |
| `token_file`    | Absolute path to the enclave's Google Drive token JSON.                 |

Optional overrides (commented in the example file): `vm_name`, `machine_type`, `boot_disk_size_gb`, `container_image`, `use_encryption`, `job_timeout_seconds`.

Do **not** set `dev_mode` in tfvars — `just tf-apply` / `just tf-apply-dev` force it on the command line, which takes precedence, so a stray tfvars value can never produce the wrong deployment type.

> ⚠️ **Token handling.** The token content is uploaded to Secret Manager _and_ stored in plaintext in the local `terraform.tfstate`. The state file is gitignored — never commit it, share it, or move it off your machine.

## Quickstart: production

Hardened image — no SSH, TEE enforcement, encryption on, container restart policy `Never`.

```bash
just tf-init      # once: download providers, set up local state
just tf-plan      # preview what will be created
just tf-apply     # provision APIs, SA, IAM, secret, and the VM
just tf-status    # RUNNING / TERMINATED / ...
just tf-output    # IP, SA email, secret resource
```

Attestation on production cannot be fetched over the network (no open ports, no SSH) — it is delivered via the peer flow and verified by peers with `client.attest_peer()`.

## Quickstart: dev mode

Debug image — SSH enabled, container logs redirected to serial output, encryption off (override with `use_encryption = true` in tfvars), restart policy `Always`.

```bash
just tf-apply-dev
just tf-logs        # full container logs via SSH + journalctl (falls back to serial)
just tf-ssh         # SSH into the VM
just tf-attest      # attestation report, fetched via SSH + localhost
```

> `tf-logs` prefers SSH + `journalctl` because the serial console caps redirected container output — chatty containers go quiet in serial logs after ~1MB. On production deployments (no SSH) it falls back to serial output, which mainly shows boot/launcher logs.

## Teardown

```bash
just tf-destroy
```

Deletes the VM, secret, service account, and IAM bindings. The enabled APIs are intentionally left on (`disable_on_destroy = false`) — disabling project-wide APIs could break other workloads in the project.


## Dev
### Iterating on the enclave code

- **Pure local (no GCP):** unchanged — `just local-build` and `just local-run` run the container on your machine with `SYFT_ENCLAVE_REQUIRE_TEE=false`.
- **On GCP:** build and push a tag, point Terraform at it, deploy in dev mode:

  ```bash
  just build-push "mytag"                 # or build-push-amd
  # in terraform/terraform.tfvars:
  #   container_image = "docker.io/openminedreleasebot/syft-client-enclave:mytag"
  just tf-apply-dev
  ```

  Re-pushed the _same_ tag? Nothing changes in config, so force a VM replacement:

  ```bash
  just tf-redeploy dev=true
  ```

## Troubleshooting

| Error                                | Fix                                                                                                                                                                                                                                                                                                                                                   |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `could not find default credentials` | Run `gcloud auth application-default login` — `gcloud auth login` alone is not enough.                                                                                                                                                                                                                                                                |
| `oauth2: "invalid_grant"`            | ADC token expired/revoked — re-run `gcloud auth application-default login`.                                                                                                                                                                                                                                                                           |
| Quota project warnings               | `gcloud auth application-default set-quota-project YOUR_PROJECT_ID`                                                                                                                                                                                                                                                                                   |
| Attestation fails on first boot      | Check the enclave SA has `roles/confidentialcomputing.workloadUser` (Terraform grants it to the dedicated SA only, not the default compute SA). A fresh apply waits 120s for IAM propagation (`time_sleep.iam_propagation`); if the launcher still 403s and exits (`exit_code=4`, dead VM), reset the VM: `gcloud compute instances reset <vm_name>`. |


## Formatting and validation

Before committing changes to the `.tf` files:

```bash
terraform -chdir=terraform fmt -recursive
terraform -chdir=terraform validate
```
