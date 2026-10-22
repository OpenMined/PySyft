# syft-bg Python API

```python
import syft_bg
```

## Init

```python
syft_bg.init(do_email="you@example.com")
```

Creates the config file. Optionally pass `syftbox_root` and `token_path`.

## Ensure running

```python
syft_bg.ensure_running(["notify", "approve"])
```

Starts the listed services if they aren't already running. Pass `restart=True` to force a restart.

- `services` — list of service names, or a dict mapping service names to config overrides
- `restart` — if `True`, restart services even if already running (default `False`)

## Status

```python
syft_bg.status
```

Shows email, services, apis, and environment info. No parentheses needed.

## Create an api from a job

```python
from syft_bg import create_api_from_job

job = do_manager.jobs[0]
create_api_from_job(job)                                  # callable by the job's submitter
create_api_from_job(job, peers=["charlie@org.com"], input_policy="policies/rate_limit.py")
```

Creates an api from a job you have reviewed. The job's `run.sh` and Python file become
the api's code; `config.yaml` and `code/params.json` are what a call sends. The
arguments and their defaults are read from `code/params.json`. See
[docs/jobs.md](../../../docs/jobs.md) for calls and policies.

- `job` — `JobInfo` object to use as template
- `peers` — emails that may call the api. Defaults to the job's submitter
- `name` — name for the api (defaults to the job name)
- `args` — the api's arguments, instead of reading them from `code/params.json`
- `input_policy`, `output_policy` — scripts that run before and after each call
- `disclosures` — what a call's approval releases, e.g. `["logs"]`. Defaults to
  `default_disclosures` in config.yaml

## Create an api from files

```python
syft_bg.create_api(
    contents=["code/main.py", "run.sh"],
    file_paths=["code/params.json", "config.yaml"],
    peers=["charlie@org.com"],
    base_dir="path/to/a/job/submission",
)
```

`contents` must hold `run.sh`; `file_paths` must hold `config.yaml` and lists every file a
call sends. No file is in both. `params_file` names the file a call's arguments go in; it
defaults to the one JSON file under `code/` in `file_paths`. Paths are relative to `base_dir`.

## Manage apis

```python
syft_bg.list_apis()                                  # {name: ApiDefinition}
syft_bg.remove_api("my_analysis")
syft_bg.set_policies("my_analysis", input_policy=None)  # remove a policy
syft_bg.run_policy("policies/rate_limit.py", submitter="ds@org.com")
```

## Service control

```python
syft_bg.start()                              # start all services
syft_bg.stop()                               # stop all services
syft_bg.restart()                            # restart all services
syft_bg.ensure_running(["notify", "approve"])  # start listed services
syft_bg.logs("approve")                      # last 50 lines of approve service log
syft_bg.logs("notify")                       # last 50 lines of notify service log
```

## Typical notebook flow

```python
import syft_bg

# Create config
syft_bg.init(do_email="you@example.com")

# Start services
syft_bg.ensure_running(["notify", "approve"])

# Check everything
syft_bg.status

# Let the submitter call this job again as an api
job = do_manager.jobs[0]
syft_bg.create_api_from_job(job)
```
