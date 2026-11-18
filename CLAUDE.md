# CLAUDE.md

Guidance for Claude Code in this repository. This file is checked in and shared with the team. Put
personal instructions in a `CLAUDE.local.md` next to this file; git ignores it.

## Guidelines

Follow these guidelines when you write code, PR descriptions or reviews. The `syft-pr-review` skill
rates PRs against the same guidelines.

@dev-guidelines/guidelines.md
@dev-guidelines/language.md

If keeping an import at the top of a file causes a hard circular import, ask the user whether to
resolve it now or later.

## Pull requests

Open PRs with the `open-pr` skill. It runs pre-commit, writes a conventional-commit title, and
fills in the PR template.

## Commands

```bash
uv pip install -e .        # install (uses uv)

just test-unit-fast        # fast unit tests; always run these first
just test-unit             # all unit tests
just test-integration      # integration tests (slow, real API); only run in the nightly build

uv run pytest ./tests/unit/test_sync_manager.py                      # one file
uv run pytest ./tests/unit/test_sync_manager.py::test_function_name  # one test
```

When you run unit tests with `pytest` directly, pass `-n auto`.

## Architecture

Syft-client is a privacy-preserving data collaboration framework enabling secure file syncing and job execution across peer-to-peer networks. It uses a **file-first** communication paradigm with transport-agnostic channels (currently Google Drive Files API).

### Core Design Principles

- **File-first**: All state communicated via synced files
- **Transport-agnostic**: Works with any message-delivery mechanism
- **Offline-first**: Full functionality when peers are offline
- **Peer-first**: Explicit peer authorization required

### Two modes

PySyft runs in one of two modes, and a setup uses only one of them. Check which mode a task is about
before you change jobs, roles or approvals, because the flows differ. In both modes, the DOs and the
DS use `SyftboxManager` as the main orchestrator.

#### Mode 1: DO and DS

There are two roles:

- **Data Scientist (DS)**: sees only the mock data. The DS submits a job to run on a DO's private
  data, and gets the results back.
- **Data Owner (DO)**: holds the private data. The DO reviews each job, approves it, runs it on the
  private data on its own machine, and returns the results to the DS.

#### Mode 2: Enclave

An enclave is a sealed confidential-compute environment (a TEE) that runs an open-source docker
container with syft, and syncs through Google Drive like any other peer. This mode usually has three
roles: two DOs, for example one with a private benchmark and one with a private model, and a DS who
writes the analysis that brings their data together. The enclave runs the job, not a DO, and only
once every DO has approved it, so no party sees another party's data. The code is in
`packages/syft-enclave`; start with its `docs/flow.md`.

### Workspace packages

The packages in `packages/` are uv workspace members:

- `syft-job`: job submission and execution (bash/Python jobs with venv isolation)
- `syft-datasets`: dataset management and sharing
- `syft-rds`: the remote data science client, datasets and jobs on top of the sync engine
- `syft-permissions`: the permission system for datasites
- `syft-perms`: the user-facing permission API
- `syft-bg`: background services, with a TUI dashboard
- `syft-enclave`: runs PySyft on an enclave
- `enclave-model-api-example`: an example LLM inference API on top of `syft-enclave`
- `syft-restrict`: verifies that JAX/Flax inference code only does math, and hides the model
  architecture
- `syft-migration`: versioning and on-the-fly migration of syft objects
- `syft-notebook-ui`: Jupyter notebook display utilities

### More detail

Read the file that matches your task before you start. They are in `dev-guidelines/architecture/`.

- `sync-flow.md`: how a sync pass moves changes between the DS and the DO, and how events and
  messages are stored. Read it when you work on sync, events or messages.
- `peers.md`: how a DS requests access and the DO approves it. Read it when you work on peers.
- `datasets.md`: how the DO creates a dataset and the DS reads it. Read it when you work on
  datasets.
- `jobs.md`: how a job is submitted, approved, run, and its results returned. Read it when you
  work on jobs.
