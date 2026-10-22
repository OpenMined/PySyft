# Jobs, APIs and policies

A job is code that a data scientist (DS) sends to a data owner (DO). The DO reviews the code and runs it on their own machine, next to the private data. Only the results go back to the DS.

An API is a job that the DO approved in advance. Peers call an API with their own arguments, and each call runs as a job without a review by hand. Policies are scripts that the DO attaches to an API. They check each call before it runs and its results before they are shared.

## Jobs

### Submitting a job

The DS submits a Python file or a folder:

```python
ds_client.submit_python_job(
    user="do@example.com",
    code_path="analysis/",   # a .py file, or a folder
    job_name="my.job",
    entrypoint="main.py",    # optional for a folder that has main.py
)
```

For a folder, the entrypoint is the file that runs. When the folder holds `main.py`, or a single `.py` file, the client finds the entrypoint itself.

### What a job folder holds

```
my.job/
    run.sh      # the only file the runner executes
    code/       # the submitted code; the job runs with code/ as its working folder
    config.yaml # the job's name, type and submitter
```

The job writes its results to `outputs/`, inside `code/`. Only `outputs/` is shared with the DS. Anything the job writes elsewhere stays with the DO.

### Job lifecycle

A job moves through these statuses:

`pending` → `approved` → `running` → `done` or `failed`

The DO can also set a pending job to `rejected`. The steps happen in this order:

```python
do_client.sync()                            # the job arrives as pending
job = do_client.jobs[0]
job.approve()                               # approved
do_client.process_approved_jobs(
    share_outputs_with_submitter=True,      # running, then done or failed
)
```

`process_approved_jobs` syncs at the end, so the DS gets the results on their next sync:

```python
ds_client.sync()
ds_client.jobs[-1].output_paths
```

## APIs

### What an API is

An API is a job that the DO approved in advance. The DO chooses which peers may call it and which arguments they may pass. Each call is a new job with the same code and new arguments. The approve service, `syft_bg`, approves a call because the call matches the API, so no person reviews it.

### Creating an API (DO)

The DO creates an API from a job they reviewed:

```python
import syft_bg as sbg

sbg.create_api_from_job(job, name="word_count", peers=["ds@example.com"])
```

By convention, the arguments come from the job's `code/params.json`. `create_api_from_job` matches that file by name only, so each call can send its own values. It reads the argument names, types and defaults from the example values in the file.

Both parts can be configured. `file_paths` chooses which file is matched by name only. Any single JSON file in `code/` that is matched by name only holds the arguments. `args` sets the arguments directly, and then nothing is read from the file:

```python
sbg.create_api_from_job(
    job,
    file_paths=["code/settings.json"],
    args=[{"name": "word", "type": "str", "required": True}],
)
```

`create_api` creates an API from files on disk instead of from a job.

### What a call sends

An API has two lists of files. The files it pins by content, such as `run.sh` and `code/main.py`, are the API's code: a call can't change them. The files it takes by name (`file_paths`), such as `code/params.json`, are what a call sends. A call must send every file the API takes by name, and nothing else. No file is in both lists, and neither holds `config.yaml`: the DO writes it for each call.

An API can take any number of files by name, anywhere in the job. An API has arguments only when it takes exactly one file by name and that file is a `params.json`: `client.api` writes a call's arguments to it. An API that takes several files by name, for example a data file next to a JSON file, has no arguments; a call passes each file in `files=`.

### Calling an API (DS)

```python
ds_client.api                                 # the APIs shared with you
ds_client.api.word_count(word="privacy")      # waits, then returns the finished job
request = ds_client.api.word_count("privacy", block=False)  # returns the request
request.status                                # received, rejected or accepted
request.job                                   # the job, once the DO accepted it
ds_client.requests                            # your calls that aren't jobs yet

# An API without arguments that takes code/words.txt and code/stopwords.txt by name:
ds_client.api.word_list(files={"code/words.txt": "words.txt", "code/stopwords.txt": b"the\n"})
```

A call is not a job until the DO's client accepts it, so `client.jobs` lists it only from then on. If the DO rejects the call, a blocking call returns the request, with the reason in `request.review_reason`.

`files=` maps each file the API takes by name, other than its `params.json`, to its content (bytes) or a path to read it from. The call checks that it sends exactly the files the API takes.

Each call sends a request, not a whole job. A request is a folder in the DO's datasite, under `app_data/job/requests/<ds_email>/`, next to the `inbox/` that holds whole jobs. It holds `request.yaml`, which names the API, and the files the call sends. When the request becomes a job, the DO writes the job's `config.yaml` from `request.yaml`. `client.api` is a convenience: a DS can write any request with `JobClient.submit_api_request()`, and the DO's checks decide.

When the request arrives, the DO's client checks it:

- the API the request names is one of the DO's APIs;
- the DS is one of the API's peers;
- the request holds exactly the files the API takes by name: one it doesn't take, or one missing, rejects it;
- the arguments in the `params.json` fit the API's arguments.

If a check fails, the request is rejected, and the DS sees the reason. If every check passes, the DO's client turns the request into an ordinary job in `inbox/`, under the same name: the API's code, plus the call's arguments. The job is approved straight away, because creating the API was the DO's approval in advance. The DO's client then removes the request folder.

From then on the call is a job like any other. The DS follows it in `ds_client.jobs` under the name the call returned. The runner runs the API's policies around it, and the job keeps the code it ran, even when the DO changes the API later.

A whole job that a DS submits to `inbox/` always waits for the DO, even when its code is an API's code. Only a call of an API runs without the DO.

## Input policies

### What an input policy is for

An input policy runs before each call's job. It decides whether the call may run. Rate limiting is an example: allow 10 calls per DS per day.

### Adding an input policy

```python
sbg.create_api_from_job(job, name="word_count", input_policy="policies/rate_limit.py")

# Add, replace or remove a policy of an existing API:
sbg.set_policies("word_count", input_policy="policies/rate_limit_v2.py")
sbg.set_policies("word_count", input_policy=None)   # remove it
```

Changing the policies clears the API's state, because it was kept for the old policies. `reset_state=False` keeps it.

### The entrypoint of a policy is a bash script

The entrypoint of every policy is a bash script, `run.sh`, and the runner runs `bash run.sh`, as it does for jobs. The bash script can forward to Python. When you pass a `.py` file, the forwarding is done for you: the file is stored as `policy.py`, next to a generated `run.sh` that calls it:

```bash
#!/bin/bash
exec python3 "$(dirname "$0")/policy.py"
```

When you pass a `.sh` file, it is stored as `run.sh` as it is. Only that one file is stored, so a Python policy should be passed as the `.py` file.

The policy files are stored in the API folder and hashed. When a stored file no longer matches its hash, the job fails and does not run.

### Environment variables

The runner gives each policy these environment variables:

| Variable                         | Contents                                                   |
| -------------------------------- | ---------------------------------------------------------- |
| `SYFT_POLICY_PARAMS_PATH`        | The call's params JSON, read-only                          |
| `SYFT_POLICY_CALL_DIR`           | A copy of every file the call sent, laid out as in the job |
| `SYFT_POLICY_SUBMITTER`          | The DS's email                                             |
| `SYFT_POLICY_API_NAME`           | The API's name                                             |
| `SYFT_POLICY_STATE_DIR`          | The API's state folder, kept between runs                  |
| `SYFT_POLICY_DEFAULT_STATE_PATH` | `state.json` in the state folder, `{}` on the first run    |

The module docstring of `syft_job/policies/env.py` documents every variable and convention in one place.

### Denying a call

A policy denies a call by exiting with a non-zero code. In Python, any uncaught exception does that. The job is then marked `failed`. The last line of the policy's error output is written to `policy_error.txt` and shared with the DS. The full error output stays with the DO, because a traceback can show private paths or data.

### State and locking

The state folder belongs to the API, and only the DO can read it. The policy writes to `state.json`, or to any other file in the folder. Nothing changes unless the policy writes. The runner holds a lock on the state for the whole policy run, so two calls of one API never run their policies at the same time.

### Example: rate limit

```python
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

state_path = Path(os.environ["SYFT_POLICY_DEFAULT_STATE_PATH"])
submitter = os.environ["SYFT_POLICY_SUBMITTER"]
state = json.loads(state_path.read_text())

now = datetime.now(timezone.utc)
day_ago = now - timedelta(days=1)
calls = [t for t in state.get(submitter, []) if datetime.fromisoformat(t) > day_ago]
if len(calls) >= 10:
    raise ValueError("Rate limit reached: 10 calls per 24 hours")

state[submitter] = calls + [now.isoformat()]
state_path.write_text(json.dumps(state))
```

### Testing a policy

`sbg.run_policy` runs a policy the way the job runner runs it. It needs no job, peer or sync:

```python
result = sbg.run_policy("policies/rate_limit.py", params={"word": "privacy"},
                        submitter="ds@example.com")
result.passed   # True or False
result.error    # what the DS would see
result.state    # state.json after the run

# Run it several times against one state folder, to reach the limit:
for _ in range(11):
    result = sbg.run_policy("policies/rate_limit.py", state_dir="/tmp/rl_state")

# An existing API's policy, against a copy of its real state:
result = sbg.run_policy(api="word_count", kind="input")
```

## Output policies

### What an output policy is for

An output policy runs after each call's job. It checks the outputs before they are shared, and it decides who receives them. It gets the same environment variables as an input policy, and two more:

| Variable                      | Contents                                                              |
| ----------------------------- | --------------------------------------------------------------------- |
| `SYFT_POLICY_OUTPUTS_DIR`     | The job's `outputs/`. Whatever is here when the policy ends is shared |
| `SYFT_POLICY_RECIPIENTS_PATH` | `recipients.json`, a JSON list of the emails that receive the outputs |

When the job itself fails, the output policy does not run and the outputs are not shared.

### Recipients

The runner fills `recipients.json` with the submitter's email. The policy can read the list, change it and write it back. After the policy passes, the runner shares `outputs/` with exactly the emails in the list. Only approved peers receive outputs: the runner skips any other email and logs it for the DO, so a typo can't send results to a stranger.

An empty list holds the outputs back, but the job does not fail. The DO can share the outputs by hand later with `share_job_results`.

### What the DS sees

The DS always sees the job's status. When the output policy denies, the job is `failed` and the DS reads the reason in `job.policy_error`. When the policy removed the DS from the recipients, the job shows "Outputs not shared with you". Blocked outputs stay in the job's review folder, where only the DO reads them.

### Example: block a word

```python
import os
from pathlib import Path

outputs = Path(os.environ["SYFT_POLICY_OUTPUTS_DIR"])

for f in outputs.rglob("*"):
    if f.is_file() and "SECRET" in f.read_text(errors="ignore"):
        raise ValueError(f"Output {f.name} contains a blocked word")
```

### Example: redact CSV rows

Only `outputs/` is shared, so a job and its output policy can agree on another folder for staging. In this example the job writes `results.csv` to `output_candidates/`, next to `outputs/`. The policy replaces every row that mentions the secret, writes the cleaned file to `outputs/`, and logs the original rows in the state folder. `output_candidates/` is a convention of this example; the runner does not know about it.

```python
import csv
import os
from pathlib import Path

outputs = Path(os.environ["SYFT_POLICY_OUTPUTS_DIR"])
candidates = outputs.parent / "output_candidates"
redaction_log = Path(os.environ["SYFT_POLICY_STATE_DIR"]) / "redactions.csv"

with open(candidates / "results.csv", newline="") as f:
    reader = csv.DictReader(f)
    fields, rows = reader.fieldnames, list(reader)

redacted = [r for r in rows if "SECRET" in ",".join(r.values())]
cleaned = [{k: "[REDACTED]" for k in fields} if r in redacted else r for r in rows]

with open(outputs / "results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(cleaned)

with open(redaction_log, "a", newline="") as f:
    csv.DictWriter(f, fieldnames=fields).writerows(redacted)
```

Test it with sample files:

```python
result = sbg.run_policy("policies/redact_csv.py", candidates_dir="sample_candidates/")
result.outputs_dir   # the outputs after the run
```

### Example: add an auditor

```python
import json
import os
from pathlib import Path

recipients_path = Path(os.environ["SYFT_POLICY_RECIPIENTS_PATH"])
recipients = json.loads(recipients_path.read_text())
recipients.append("auditor@example.com")
recipients_path.write_text(json.dumps(recipients))
```
