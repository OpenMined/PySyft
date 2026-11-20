# Job flow

**1. Submit job (DS)**:

```python
ds_manager.submit_python_job(
    user=do_manager.email,
    code_path="/path/to/script.py",  # or folder path
    job_name="my.job",
    entrypoint="main.py",  # optional, auto-detected if main.py exists
)
```

**2. Receive and approve job (DO)**:

```python
do_manager.sync()  # Receives job in inbox
job = do_manager.job_client.jobs[0]
job.approve()  # Mark for execution
```

**3. Run job (DO)**:

```python
do_manager.job_runner.process_approved_jobs()  # Executes in isolated venv
```

Jobs run in a Python 3.12 venv. For folders with `pyproject.toml`, uses `uv sync`. Otherwise uses `uv venv`.

**4. Download results (DS)**:

```python
do_manager.sync()  # Push results to outbox
ds_manager.sync()  # Pull results
output_path = ds_manager.job_client.jobs[-1].output_paths[0]
```

Jobs can access private data via `syft://` protocol:

```python
import syft_client as sc
path = sc.resolve_path("syft://private/syft_datasets/my dataset/private.txt")
```
