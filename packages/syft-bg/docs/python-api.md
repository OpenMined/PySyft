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
    contents=["code/main.py", "run.sh"],
    file_paths=["code/params.json", "config.yaml"],
    peers=["charlie@org.com"],
    base_dir="~/SyftBox/datasites/me@org.com/apis/jobs/inbox/ds@org.com/my-job",
)
```

Registers files for auto-approval. Jobs matching these files from listed peers get approved automatically.

A path is relative to the job submission root, so code sits under `code/` and
the script the runner executes is `run.sh`. The two lists together must name
every file of the submission, and `run.sh` must be in `contents`, or the rule
matches nothing. `config.yaml` belongs in `file_paths`: its bytes carry the job
name and the time it was submitted, so hashing them matches one job.

- `contents` — files (or directories) to approve by content
- `file_paths` — files to allow by name only (e.g. data files)
- `peers` — restrict to these emails. Omit to allow any peer
- `name` — optional name for the approval object
- `base_dir` — the job submission directory, which `contents` is resolved
  against. `file_paths` is stored as written, so write those relative to the
  same root

## Auto-approve from job

```python
from syft_bg import auto_approve_job

job = do_manager.jobs[0]

# Default: every file matched by content, except config.yaml and
# code/params.json, which are matched by name
auto_approve_job(job)

# Match data.json by name too; config.yaml and code/params.json stay by name
auto_approve_job(job, file_paths=["data.json"])

# Content-match main.py and the script that runs, allow the rest by name
auto_approve_job(
    job,
    contents=["main.py", "run.sh"],
    file_paths=["data.json", "config.yaml"],
)
```

`contents` must name `run.sh`, and the two lists together must name every file
of the job, or `auto_approve_job` refuses to write an object that could never
approve anything.

Creates an auto-approval config from an existing job's files. Calls `auto_approve()` internally.

- `job` — `JobInfo` object to use as template
- `contents` — filenames from the job to match by name AND content. Default (None): every file is content-matched except `config.yaml` and `code/params.json`, which are matched by name
- `file_paths` — filenames from the job to match by name only. With `contents` left unset, the other files are content-matched, except `config.yaml` and `code/params.json`, which stay matched by name. With both set, the two lists must name every file of the job between them
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
