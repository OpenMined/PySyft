"""Benchmark job submission and sync performance."""

import os
import time
import tempfile
from pathlib import Path
from syft_client.sync.syftbox_manager import SyftboxManager

REPO_ROOT = Path(__file__).parent.parent
CREDENTIALS_DIR = REPO_ROOT / "credentials"
EMAIL_DO = os.environ["AI_AUDIT_EMAIL_DO"]
EMAIL_DS = os.environ["AI_AUDIT_EMAIL_DS"]
token_path_do = CREDENTIALS_DIR / os.environ.get(
    "ai_audit_credentials_fname_do", "token_do.json"
)
token_path_ds = CREDENTIALS_DIR / os.environ.get(
    "ai_audit_credentials_fname_ds", "token_ds.json"
)


def create_test_job(index: int) -> Path:
    """Create a simple test job script."""
    job_dir = Path(tempfile.mkdtemp())
    script_path = job_dir / f"job_{index}.py"
    script_path.write_text(
        f"""
import os
import json
os.makedirs("outputs", exist_ok=True)
with open("outputs/result.json", "w") as f:
    json.dump({{"job_index": {index}, "status": "completed"}}, f)
"""
    )
    return script_path


def benchmark_job_sync():
    os.environ["PRE_SYNC"] = "false"

    # Clean start
    ds, do = SyftboxManager._pair_with_google_drive_testing_connection(
        do_email=EMAIL_DO,
        ds_email=EMAIL_DS,
        do_token_path=token_path_do,
        ds_token_path=token_path_ds,
        add_peers=False,
        use_in_memory_cache=False,
    )
    ds.delete_syftbox()
    do.delete_syftbox()

    # Fresh managers
    ds, do = SyftboxManager._pair_with_google_drive_testing_connection(
        do_email=EMAIL_DO,
        ds_email=EMAIL_DS,
        do_token_path=token_path_do,
        ds_token_path=token_path_ds,
        use_in_memory_cache=False,
    )

    # DS submits 10 jobs
    for i in range(10):
        script_path = create_test_job(i)
        ds.submit_python_job(
            user=do.email,
            code_path=str(script_path),
            job_name=f"benchmark.job.{i}",
        )

    time.sleep(2)

    # DO syncs and receives jobs
    do.sync()

    # DO approves and executes first 5 jobs
    for job in do.job_client.jobs[:5]:
        job.approve()
    do.job_runner.process_approved_jobs(stream_output=False)

    # DO syncs to push results
    do.sync()

    time.sleep(2)

    # DS first sync (downloads results)
    ds_sync1_start = time.time()
    ds.sync()
    ds_sync1_time = time.time() - ds_sync1_start

    # DS second sync (should be fast - caching)
    ds_sync2_start = time.time()
    ds.sync()
    ds_sync2_time = time.time() - ds_sync2_start

    print(f"DS first sync:  {ds_sync1_time:.2f}s")
    print(f"DS second sync: {ds_sync2_time:.2f}s")


if __name__ == "__main__":
    benchmark_job_sync()
