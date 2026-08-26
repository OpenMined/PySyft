# Changelog

## v0.10.0

- **Package renamed `syft-client` → `syft`.** Install with `pip install syft` and
  use `import syft as sy` (`import syft_client as sc` no longer works). Version
  numbering continues the PySyft lineage on PyPI: 0.10.0 follows syft-client
  0.1.117 and legacy PySyft 0.9.x. If you depend on the legacy PySyft ≤0.9 API,
  pin `syft<0.10`.
- `syft` is the sync engine only. `syft.login_do()` / `syft.login_ds()` return a
  `SyftboxManager` (peers + file sync); datasets and jobs (`create_dataset`,
  `submit_python_job`, `jobs`, `datasets`, `process_approved_jobs`, ...) are on
  `syft_rds.SyftRDSClient`: `pip install syft-rds` and
  `from syft_rds import login_do, login_ds`. Data owners running background
  services also install `syft-bg`. `syft` no longer depends on `syft-bg` or
  `syft-job`.
- Environment variables renamed, with no fallback to the old names:
  `SYFTCLIENT_TOKEN_PATH` → `SYFT_TOKEN_PATH`, `SYFTCLIENT_DEV_MODE` →
  `SYFT_DEV_MODE`, `SYFT_CLIENT_INSTALL_SOURCE` → `SYFT_INSTALL_SOURCE`.
- `MIN_SUPPORTED_SYFT_VERSION` is `0.10.0`: peers on syft-client 0.1.x are
  reported as incompatible. Every user sees the version-mismatch prompt on
  first login and must upgrade; 0.10.0 clients create fresh versioned folders
  on Google Drive.
- Enclave rebranded with the package: Docker Hub images are now
  `openminedreleasebot/syft-enclave` and `syft-enclave-inference` (old
  `syft-client-enclave*` images are not rebuilt), the attestation audience is
  `syft-attestation`, the version nonce is `syft-<version>`, and the enclave's
  HTTP landing/attestation JSON reports `service: syft-enclave` and `syft_version`.
  0.10.0 clients can only attest 0.10.0 images.

## v0.1.94

- #98 Fix truncated job logs when process exits quickly
- #97 Load datasets as DO on connect, make "any" datasets work on peer approval
- #96 Init version negotiation
- #95 Fix folder lookups to search within SyftBox folder
- #94 Add parallel dataset collection downloads for DS
- #93 Add parallel file downloads in DatasiteOutboxPuller
- #90 Cache datasets
- #88 Fix reversed path resolution in resolve_dataset_files_path
- #82 Fix loading of datasets
- #80 Init auto approve jobs
- #79 Improve pagination
- #78 Add timeout parameter to job execution methods
- #77 Fix SyftBoxManager peers attribute
- #74 Real peer requests
- #73 Datasets channel
- #72 Parquet computed field fix
- #71 Thread executor
- #69 Simpler folder job submission
- #64 Parquet fix
- #63 Add notification to syft-client
