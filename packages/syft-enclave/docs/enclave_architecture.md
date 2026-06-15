# Syft Client Enclave - Confidential Spaces Deployment

This directory contains a Docker image that packages `syft-client` with an HTTP attestation server. When deployed on Google Confidential Spaces, the `/attestation` endpoint returns a cryptographically signed TEE attestation report.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  GCP Confidential VM (AMD SEV - encrypted memory)   │
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
│  │  │  syft-client-enclave container          │  │  │
│  │  │  - Attestation published via gdrive     │  │  │
│  │  │  - signed JWT with TEE claims           │  │  │
│  │  └─────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## Deploying to Google Confidential Spaces

### Confidential Computing support

Confidential Spaces supports the following confidential compute types:

| Type  | Description                         | Machine Types       |
| ----- | ----------------------------------- | ------------------- |
| `SEV` | AMD Secure Encrypted Virtualization | `n2d-*` (AMD Milan) |
| `TDX` | Intel Trust Domain Extensions       | `c3-*`              |

> **Note:** AMD SEV-SNP is NOT supported by Confidential Spaces (only by raw Confidential VMs). The recipes use `SEV`.
