# Syft Client Enclave - Confidential Spaces Deployment

This directory contains a Docker image that packages `syft` with a small HTTP server. On Google Confidential Spaces, the enclave writes a cryptographically signed TEE attestation token to `SYFT_version.json`, where peers read it. The HTTP server does not serve that token: on Confidential Spaces it serves only `/`, `/health` and `/docs`, and the enclave opens no inbound port to reach it.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  GCP Confidential VM (SEV/TDX - encrypted memory)   │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │  Confidential Space OS (hardened, read-only)  │  │
│  │                                               │  │
│  │  ┌─────────────────────────────────────────┐  │  │
│  │  │  TEE Container Launcher                 │  │  │
│  │  │  - Pulls & verifies container image     │  │  │
│  │  │  - Exposes attestation Unix socket      │  │  │
│  │  │  - Manages container lifecycle          │  │  │
│  │  └──────────────┬──────────────────────────┘  │  │
│  │                 │                             │  │
│  │  ┌──────────────▼──────────────────────────┐  │  │
│  │  │  syft-enclave container                 │  │  │
│  │  │  - Attestation published via gdrive     │  │  │
│  │  │  - signed JWT with TEE claims           │  │  │
│  │  └─────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## Deploying to Google Confidential Spaces

### Confidential Computing support

Confidential Spaces supports the following confidential compute types:

| Type  | Description                         | Machine Types                         |
| ----- | ----------------------------------- | ------------------------------------- |
| `SEV` | AMD Secure Encrypted Virtualization | `n2d-*` (AMD Milan)                   |
| `TDX` | Intel Trust Domain Extensions       | `c3-*`; `a3-highgpu-1g` (1× H100 GPU) |

> **Note:** AMD SEV-SNP is NOT supported by Confidential Spaces (only by raw Confidential VMs). The recipes use `SEV` for cpu deployments and `TDX` for gpu (`hardware=gpu`).
