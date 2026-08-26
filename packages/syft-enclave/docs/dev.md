# Dev

## Debugging

Production image features:

- No SSH access
- No container log redirection (serial logs only show launcher output)
- `tee-restart-policy=Never` shuts down the VM if the container exits
- `Hardened:true` in attestation claims

Debug image features:

- SSH access via `just ssh`
- Container stdout/stderr visible in serial logs (`just logs`)
- `tee-restart-policy=Always` keeps the VM running if the container crashes
- `tee-container-log-redirect=true` redirects container logs to serial output

### Debug deployment (recommended for initial setup)

The debug image allows SSH access and container log redirection to serial output, making it easier to troubleshoot issues.

```bash
just start-debug                 # cpu (default)
just hardware=gpu start-debug    # gpu (a3-highgpu-1g, 1x H100) — see docs/terraform.md "GPU deployments"
```

Debug enclaves run with encryption off — data owner clients must match: `login_do(encryption=False)`.

## Building & pushing a new Docker Image

The image must be built for `linux/amd64` — GCP Confidential VMs are amd64 (AMD EPYC for cpu deployments, Intel Sapphire Rapids for gpu). On Apple Silicon, use the multi-arch recipe so the same tag works locally (arm64) and on GCP (amd64):

```bash
just build-push        # multi-arch (linux/amd64 + linux/arm64)
just build-push-amd    # amd64 only
```

Both push to `docker.io/openminedreleasebot/syft-enclave:latest`. You need push access to that Docker Hub namespace.

## Troubleshooting

| Symptom                                                 | Cause                                                         | Fix                                                                                 |
| ------------------------------------------------------- | ------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `exec format error` in serial logs                      | Docker image built for wrong architecture (arm64 on amd64 VM) | Rebuild with `just build-push` (multi-arch) or `just build-push-amd`                |
| `unexpected_snp_attestation`                            | Used `SEV_SNP` instead of `SEV`                               | Confidential Spaces only supports `SEV` and `TDX`, not `SEV_SNP`                    |
| `logging redirection only allowed on debug environment` | Used `tee-container-log-redirect=true` with production image  | Use `just start-debug` instead of `just start`                                      |
| `403 Forbidden` pulling image                           | VM service account lacks Artifact Registry access             | Grant `roles/artifactregistry.reader` or use a public Docker Hub image              |
| VM terminates immediately                               | Container crashed with `tee-restart-policy=Never`             | Switch to `just start-debug` to investigate                                         |
| `OnHostMaintenance` error                               | Missing `--maintenance-policy` flag                           | The recipes already set `--maintenance-policy=TERMINATE`; check you didn't override |
| GPU VM suddenly gone                                    | Flex-start run duration elapsed (VM + disk auto-delete)       | By design — redeploy with `just hardware=gpu start-debug`                           |
