"""Benchmark initial DO sync time after 100 jobs have been submitted."""

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

NUM_JOBS = 100


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


def benchmark_initial_do_sync():
    os.environ["PRE_SYNC"] = "false"

    print(f"Setting up benchmark with {NUM_JOBS} jobs...")

    # Clean start - delete any existing syftboxes
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

    # Fresh pair for job submission
    ds, do = SyftboxManager._pair_with_google_drive_testing_connection(
        do_email=EMAIL_DO,
        ds_email=EMAIL_DS,
        do_token_path=token_path_do,
        ds_token_path=token_path_ds,
        use_in_memory_cache=False,
    )

    # DS submits 100 jobs
    print(f"Submitting {NUM_JOBS} jobs from DS...")
    submit_start = time.time()
    for i in range(NUM_JOBS):
        script_path = create_test_job(i)
        ds.submit_python_job(
            user=do.email,
            code_path=str(script_path),
            job_name=f"benchmark.job.{i}",
        )
        if (i + 1) % 10 == 0:
            print(f"  Submitted {i + 1}/{NUM_JOBS} jobs")
    submit_time = time.time() - submit_start
    print(f"Job submission complete: {submit_time:.2f}s")

    # Wait for Google Drive to propagate
    print("Waiting for Google Drive propagation...")
    time.sleep(5)

    # Create a fresh DO manager to simulate initial sync from a new pair
    print("Creating fresh DO manager for initial sync benchmark...")
    _, fresh_do = SyftboxManager._pair_with_google_drive_testing_connection(
        do_email=EMAIL_DO,
        ds_email=EMAIL_DS,
        do_token_path=token_path_do,
        ds_token_path=token_path_ds,
        use_in_memory_cache=False,
        clear_caches=True,
    )

    # Benchmark initial sync
    print("Starting initial DO sync...")
    sync_start = time.time()
    fresh_do.sync()
    initial_sync_time = time.time() - sync_start

    # Verify jobs were received
    jobs_received = len(fresh_do.job_client.jobs)

    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Jobs submitted:      {NUM_JOBS}")
    print(f"Jobs received:       {jobs_received}")
    print(f"Job submission time: {submit_time:.2f}s")
    print(f"Initial DO sync:     {initial_sync_time:.2f}s")
    print("=" * 50)


if __name__ == "__main__":
    benchmark_initial_do_sync()
