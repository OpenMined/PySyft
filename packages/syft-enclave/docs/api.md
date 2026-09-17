## API Endpoints

Once the container is running, the following endpoints are available at `http://EXTERNAL_IP:8080`:

| Endpoint           | Description                                                       |
| ------------------ | ----------------------------------------------------------------- |
| `GET /`            | Landing page with syft version and available endpoints            |
| `GET /attestation` | TEE attestation report (signed JWT with hardware/software claims) |
| `GET /health`      | Health check                                                      |
| `GET /docs`        | FastAPI auto-generated Swagger UI                                 |

## Running the enclave runner

The runner is configured entirely through `SYFT_ENCLAVE_*` environment
variables — there is no command-line interface, because the runner is always
started programmatically (by the Confidential Spaces launcher in production,
or from a `.env` file during local development). Start it with:

```
python -m syft_enclaves
```

| Variable                            | Required | Default           | Description                                         |
| ----------------------------------- | -------- | ----------------- | --------------------------------------------------- |
| `SYFT_ENCLAVE_EMAIL`                | yes      | —                 | Enclave datasite email                              |
| `SYFT_ENCLAVE_SYFTBOX_FOLDER`       | no       | `~/SyftBox_email` | Root SyftBox folder                                 |
| `SYFT_ENCLAVE_TOKEN_PATH`           | yes      | —                 | Pre-authorized Google Drive OAuth token             |
| `SYFT_ENCLAVE_POLL_INTERVAL`        | no       | `10`              | Seconds between poll cycles                         |
| `SYFT_ENCLAVE_REQUIRE_TEE`          | no       | `false`           | Refuse to start outside a TEE                       |
| `SYFT_ENCLAVE_LOG_LEVEL`            | no       | `INFO`            | Logging level                                       |
| `SYFT_ENCLAVE_ATTESTATION_PROVIDER` | no       | `auto`            | `auto` / `confidential_space` / `tinfoil` / `none`  |
| `SYFT_ENCLAVE_TINFOIL_REPO`         | no       | —                 | Tinfoil config repo, recorded in published evidence |
| `SYFT_ENCLAVE_TINFOIL_RELEASE_TAG`  | no       | —                 | Tinfoil config release tag, as above                |

For local development, place these in a `.env` file in the working directory.
The same `python -m syft_enclaves` entry point runs unchanged locally, inside
Docker, in Confidential Spaces, and in a Tinfoil CVM — only the environment
differs. Which attestation provider is used is detected from the environment
unless `SYFT_ENCLAVE_ATTESTATION_PROVIDER` says otherwise; see
[Tinfoil Deployment](./tinfoil_deployment.md).

## Example: Fetching the attestation report

```bash
curl http://EXTERNAL_IP:8080/attestation | python3 -m json.tool
```

The response always includes:

- `provider` - which deployment target produced the evidence (`confidential_space` or `tinfoil`)
- `evidence` - the provider-agnostic envelope, exactly as published to peers in
  `SYFT_version.json`. `evidence.body` is the raw evidence for independent
  verification: the full JWT on Confidential Spaces, the base64 hardware report
  on Tinfoil.
- `attestation` - an unverified, display-only summary, whose shape depends on
  the provider.

On Confidential Spaces, `attestation` holds:

- `hardware.hwmodel` - TEE hardware type (`GCP_AMD_SEV`; `GCP_INTEL_TDX` on gpu deployments)
- `hardware.secboot` - Secure boot status
- `hardware.dbgstat` - Debug status (`enabled` for debug image, `disabled-since-boot` for production)
- `container.image_digest` - SHA256 of the running container image
- `gpu` - gpu deployments only: `cc_mode` (`"ON"` = confidential computing active), `gpus[].hwmodel` (`GCP_NVIDIA_H100`), driver version
- `gce.*` - GCP project, zone, instance info

On Tinfoil it holds the `document` (`format` + `body`), the verified `config`
the enclave booted with, and `container_status`. The hardware measurements are
not decoded here — a relying party gets them by verifying the report, which is
what `attest_peer` does.
