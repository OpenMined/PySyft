<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/OpenMined/PySyft/dev/docs/img/Syft-Logo-Light.svg">
  <img alt="Syft Logo" src="https://raw.githubusercontent.com/OpenMined/PySyft/dev/docs/img/Syft-Logo.svg" width="200px" />
</picture>

# PySyft v2

> **`syft` 0.10+ is the successor of `syft-client`.** `syft` is the sync engine (`import syft as sy`); datasets and jobs live in `syft-rds` (`from syft_rds import login_do, login_ds`). If you depend on the legacy PySyft ≤0.9 API, pin `syft<0.10`.

[![Unit Tests](https://github.com/OpenMined/pysyft/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/OpenMined/PySyft/actions/workflows/unit-tests.yml)
[![Integration Tests](https://github.com/OpenMined/pysyft/actions/workflows/integration-tests.yml/badge.svg)](https://github.com/OpenMined/pysyft/actions/workflows/integration-tests.yml)
[![PyPI](https://img.shields.io/pypi/v/syft)](https://pypi.org/project/syft/)
[![Python 3.10+](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FOpenMined%2Fpysyft%2Fdev%2Fpyproject.toml)](https://github.com/OpenMined/pysyft)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/OpenMined/pysyft/blob/main/pyproject.toml)

PySyft lets data scientists submit computations which are run by data owners on private data — all through cloud storage their organizations already use (Google Drive, Microsoft 365, etc.). No new infrastructure required.

## Docs

- [Workflow](https://github.com/OpenMined/PySyft/blob/dev/docs/workflow.md) — End-to-end privacy-preserving data analysis workflow
- [API Reference](https://github.com/OpenMined/PySyft/blob/dev/docs/API.md) — All public client methods and properties
- [Authentication & Setup](https://github.com/OpenMined/PySyft/blob/dev/docs/auth.md) — Google Cloud OAuth setup for local/Jupyter usage
- [Background Services](https://github.com/OpenMined/PySyft/blob/dev/packages/syft-bg/README.md) — Email notifications, auto-approval, and TUI dashboard
- [Connections](https://github.com/OpenMined/PySyft/blob/dev/docs/connections.md) — How the Google Drive transport layer works
- [Permissions](https://github.com/OpenMined/PySyft/blob/dev/packages/syft-permissions/docs/permission-user-docs.md) - Permissions for syft
- [Enclaves](https://github.com/OpenMined/PySyft/tree/dev/packages/syft-enclave) - Enclaves with syft

## Tutorials

- [Double blind LLM evals in enclaves](https://github.com/OpenMined/PySyft/blob/dev/notebooks/enclave/gemma/colab) — a researcher evaluates a private model on a private benchmark inside an enclave, with neither side revealing its assets
- [LLM user logs analysis](https://github.com/OpenMined/PySyft/blob/dev/notebooks/ai_audit/external/DS_Tutorial_v2.ipynb) — submit jobs against private LLM user logs and get back only the approved results ([data owner notebook](https://github.com/OpenMined/PySyft/blob/dev/notebooks/ai_audit/internal/DO_Tutorial_V3.ipynb))

## Features

- **Privacy-preserving** — Private data never leaves the data owner's machine; only approved results are shared
- **Transport-agnostic** — Works over Google Drive today, extensible to any file-based transport
- **Offline-first** — Full functionality even when peers are offline; changes sync when connectivity resumes
- **Peer-to-peer with explicit auth** — Data owners must approve each collaborator before any data flows
- **Isolated job execution** — Jobs run in sandboxed Python virtual environments with controlled access to private data
- **Dataset sharing with mock/private separation** — Data scientists explore mock data, then submit jobs that run on the real thing

## Quick Start

We assume two parties here, a Data Owner (DO) and a Data Scientist (DS), the DS wants to do an analysis on private data of the DO. For brevity we use code blocks, but in practice these would be distributed: each party executes their code on their own machine.

```bash
# Data scientist
uv pip install "syft>=0.10.0" "syft-rds>=0.6.0"
# Data owner (adds the background services)
uv pip install "syft>=0.10.0" "syft-rds>=0.6.0" "syft-bg>=0.3.12"
```

```python
import syft as sy                          # sync engine + helpers (resolve_dataset_file_path, bug_report, ...)
from syft_rds import login_do, login_ds    # datasets + jobs (the Remote Data Science client)
```

```python
# Login (colab auth, for non-colab pass token_path)
do = login_do(email="do@org.com") # use your own email
ds = login_ds(email="ds@org.com") # use another email here

# Peer request & approve
ds.add_peer("do@org.com")
do.approve_peer_request("ds@org.com")

# Create & sync dataset
do.create_dataset(
    name="census",
    mock_path="mock.txt", # create this and add some text
    private_path="private.txt", # create this and add some text
    users=["ds@org.com"],
)
do.sync(); ds.sync()
datasets = ds.datasets.get_all()
```

Write an `analysis.py` that reads the dataset and produces a result in our case this is just the length of the data. Inside a job, `resolve_dataset_file_path` automatically resolves to the **private** data:

```python
# analysis.py
import json
import syft as sy

data_path = sy.resolve_dataset_file_path("census")
with open(data_path, "r") as f:
    data = f.read()

with open("outputs/result.json", "w") as f:
    json.dump({"length": len(data)}, f)
```

Submit the job and retrieve results:

```python
# Submit job
ds.submit_python_job(
    user="do@org.com",
    code_path="analysis.py",
    job_name="analysis",
)
ds.sync(); do.sync()

# Data owner Approves & runs job
do.jobs["do@org.com"]["analysis"].approve()
do.process_approved_jobs(share_outputs_with_submitter=True)
do.sync(); ds.sync()
result = open(ds.jobs["do@org.com"]["analysis"].output_paths[0]).read()
```

## Packages

| Package                                                                                      | Description                                                       |
| -------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| [`syft-rds`](https://github.com/OpenMined/PySyft/tree/dev/packages/syft-rds)                 | Remote Data Science client: `login_do`/`login_ds`, datasets, jobs |
| [`syft-datasets`](https://github.com/OpenMined/PySyft/tree/dev/packages/syft-datasets)       | Dataset management and sharing                                    |
| [`syft-job`](https://github.com/OpenMined/PySyft/tree/dev/packages/syft-job)                 | Job submission and execution                                      |
| [`syft-permissions`](https://github.com/OpenMined/PySyft/tree/dev/packages/syft-permissions) | Permission system for Syft datasites                              |
| [`syft-perms`](https://github.com/OpenMined/PySyft/tree/dev/packages/syft-perms)             | User-facing permission API for Syft datasites                     |
| [`syft-bg`](https://github.com/OpenMined/PySyft/tree/dev/packages/syft-bg)                   | Background services TUI dashboard for SyftBox                     |
| [`syft-notebook-ui`](https://github.com/OpenMined/PySyft/tree/dev/packages/syft-notebook-ui) | Jupyter notebook display utilities                                |

## Development

```bash
# Install in development mode
uv pip install -e .

# Run tests
just test-unit          # Unit tests (fast, mocked)
just test-integration   # Integration tests (slow, real API)
```

---

Built by [OpenMined](https://openmined.org) — building open-source technology for privacy-preserving data science and AI.

## Support

For questions about PySyft, reach out via `#support` on <a href="https://slack.openmined.org/">Slack</a>.

# Community

Supported by the OpenMined Foundation, the OpenMined Community is an online network of over 17,000 technologists, researchers, and industry professionals keen to _unlock 1000x more data in every scientific field and industry_.

<a href="https://join.slack.com/t/openmined/shared_invite/zt-2hxwk07i9-HO7u5C7XOgou4Z62VU78zA"><img width=150px src="https://img.shields.io/badge/Join_us-%20slack-purple?logo=slack" /></a>

# Contributors

OpenMined and Syft appreciates all contributors, if you would like to fix a bug or suggest a new feature, please reach out via <a href="https://github.com/OpenMined/PySyft/issues">Github</a> or <a href="https://join.slack.com/t/openmined/shared_invite/zt-2hxwk07i9-HO7u5C7XOgou4Z62VU78zA/">Slack</a>!

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/OpenMined/PySyft/dev/docs/img/contributors_dark.jpg">
  <img src="https://raw.githubusercontent.com/OpenMined/PySyft/dev/docs/img/contributors_light.jpg" alt="Contributors" width="100%" />
</picture>

# About OpenMined

OpenMined is a non-profit foundation creating technology infrastructure that helps researchers get answers from data without needing a copy or direct access. Our community of technologists is building Syft.

<a href="https://donate.stripe.com/fZe03H0aLdAO59e9AA
"><img width=200px src="https://img.shields.io/badge/Donate_to-OpenMined-yellow?logo=stripe" /></a>

# Supporters

<table border="0">
<tr>
<th align="center">
<a href="https://sloan.org/"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/dev/docs/img/logo_sloan.png" /></a>
</th>
<th align="center">
<a href="https://opensource.fb.com/"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/dev/docs/img/logo_meta.png" /></a>
</th>
<th align="center">
<a href="https://pytorch.org/"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/dev/docs/img/logo_torch.png" /></a>
</th>
<th align="center">
<a href="https://www.dpmc.govt.nz/">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/OpenMined/PySyft/dev/docs/img/logo_nz_dark.png">
  <img src="https://raw.githubusercontent.com/OpenMined/PySyft/dev/docs/img/logo_nz_light.png" />
</picture>
</a>
</th>
<th align="center">
<a href="https://twitter.com/"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/dev/docs/img/logo_twitter.png" /></a>
</th>
<th align="center">
<a href="https://google.com/"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/dev/docs/img/logo_google.png" /></a>
</th>
<th align="center">
<a href="https://microsoft.com/"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/dev/docs/img/logo_microsoft.png" /></a>
</th>
<th align="center">
<a href="https://omidyar.com/"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/dev/docs/img/logo_on.png" /></a>
</th>
<th align="center">
<a href="https://www.udacity.com/"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/dev/docs/img/logo_udacity.png" /></a>
</th>
<th align="center">
<a href="https://www.centerfordigitalhealthinnovation.org/">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/OpenMined/PySyft/dev/docs/img/logo_cdhi_dark.png">
  <img src="https://raw.githubusercontent.com/OpenMined/PySyft/dev/docs/img/logo_cdhi_light.png" />
</picture>

</a>
</th>
<th align="center">
<a href="https://arkhn.org/">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/OpenMined/PySyft/dev/docs/img/logo_arkhn.png">
  <img src="https://raw.githubusercontent.com/OpenMined/PySyft/dev/docs/img/logo_arkhn_light.png" />
</picture>
</a>
</th>
</tr>
</table>

# License

[Apache License 2.0](https://github.com/OpenMined/PySyft/tree/dev/LICENSE)<br />
<a href="https://www.flaticon.com/free-icons/person" title="person icons">Person icons created by Freepik - Flaticon</a>
