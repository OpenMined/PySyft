## API Endpoints

Once the container is running, the following endpoints are available at `http://EXTERNAL_IP:8080`:

| Endpoint           | Description                                                                    |
| ------------------ | ------------------------------------------------------------------------------ |
| `GET /`            | Landing page with syft version and available endpoints                         |
| `GET /attestation` | TEE attestation report, on Tinfoil only. On Confidential Spaces it returns 404 |
| `GET /health`      | Health check                                                                   |
| `GET /docs`        | FastAPI auto-generated Swagger UI                                              |

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

The enclave serves the attestation report on Tinfoil only:

```bash
curl https://ENCLAVE_HOST/attestation | python3 -m json.tool
```

The response includes:

- `provider` - the deployment target that produced the evidence, always `tinfoil`.
- `evidence` - the evidence envelope, exactly as the enclave publishes it to peers in
  `SYFT_version.json`. `evidence.body` is the base64 hardware report, so you can verify the report
  yourself.
- `attestation` - a summary for display. Nothing in the summary is verified. It holds the
  `document` (`format` + `body`), the verified `config` the enclave booted with, and
  `container_status`. The summary does not decode the hardware measurements. A relying party gets
  those by verifying the report, which is what `attest_peer` does.

On Confidential Spaces the endpoint returns 404. Instead, the enclave's runner writes the signed
token to `SYFT_version.json`, and `attest_peer` reads the token from there.
