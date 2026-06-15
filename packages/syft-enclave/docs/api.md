## API Endpoints

Once the container is running, the following endpoints are available at `http://EXTERNAL_IP:8080`:

| Endpoint           | Description                                                       |
| ------------------ | ----------------------------------------------------------------- |
| `GET /`            | Landing page with syft-client version and available endpoints     |
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

| Variable                      | Required | Default           | Description                             |
| ----------------------------- | -------- | ----------------- | --------------------------------------- |
| `SYFT_ENCLAVE_EMAIL`          | yes      | —                 | Enclave datasite email                  |
| `SYFT_ENCLAVE_SYFTBOX_FOLDER` | no       | `~/SyftBox_email` | Root SyftBox folder                     |
| `SYFT_ENCLAVE_TOKEN_PATH`     | yes      | —                 | Pre-authorized Google Drive OAuth token |
| `SYFT_ENCLAVE_POLL_INTERVAL`  | no       | `10`              | Seconds between poll cycles             |
| `SYFT_ENCLAVE_REQUIRE_TEE`    | no       | `false`           | Refuse to start outside a TEE           |
| `SYFT_ENCLAVE_LOG_LEVEL`      | no       | `INFO`            | Logging level                           |

For local development, place these in a `.env` file in the working directory.
The same `python -m syft_enclaves` entry point runs unchanged locally, inside
Docker, and in Confidential Spaces — only the environment differs.

## Example: Fetching the attestation report

```bash
curl http://EXTERNAL_IP:8080/attestation | python3 -m json.tool
```

The response includes:

- `attestation.hardware.hwmodel` - TEE hardware type (`GCP_AMD_SEV`)
- `attestation.hardware.secboot` - Secure boot status
- `attestation.hardware.dbgstat` - Debug status (`enabled` for debug image, `disabled-since-boot` for production)
- `attestation.container.image_digest` - SHA256 of the running container image
- `attestation.gce.*` - GCP project, zone, instance info
- `raw_token` - Full JWT for independent verification against Google's JWKS
