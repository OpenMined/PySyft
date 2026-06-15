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

Shows email, services, auto-approval objects, and environment info. No parentheses needed.

## Auto-approve

```python
syft_bg.auto_approve(
    contents=["main.py"],
    file_paths=["params.json"],
    peers=["charlie@org.com"],
)
```

Registers files for auto-approval. Jobs matching these files from listed peers get approved automatically.

- `contents` — files (or directories) to approve by content
- `file_paths` — files to allow by name only (e.g. data files)
- `peers` — restrict to these emails. Omit to allow any peer
- `name` — optional name for the approval object

## Auto-approve from job

```python
from syft_bg import auto_approve_job

job = do_manager.jobs[0]

# Default: all files matched by name + content
auto_approve_job(job)

# Only match data.json by name, everything else by content
auto_approve_job(job, file_paths=["data.json"])

# Only content-match main.py, ignore other files
auto_approve_job(job, contents=["main.py"])

# Explicit: main.py by content, data.json by name only
auto_approve_job(job, contents=["main.py"], file_paths=["data.json"])
```

Creates an auto-approval config from an existing job's files. Calls `auto_approve()` internally.

- `job` — `JobInfo` object to use as template
- `contents` — filenames from the job to match by name AND content. Default (None): all files are content-matched
- `file_paths` — filenames from the job to match by name only. When set, all other files are content-matched
- `peers` — restrict to these emails. Defaults to the job's submitter
- `name` — optional name for the approval object (defaults to job name)

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

# Auto-approve future runs of this job
job = do_manager.jobs[0]
syft_bg.auto_approve_job(job)
```
