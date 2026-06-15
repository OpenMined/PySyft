"""Benchmark dataset loading performance."""

import os
import sys
import time
import tempfile
from pathlib import Path
from syft_client.sync.syftbox_manager import SyftboxManager

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


CREDENTIALS_DIR = REPO_ROOT / "credentials"
EMAIL_DO = os.environ["AI_AUDIT_EMAIL_DO"]
EMAIL_DS = os.environ["AI_AUDIT_EMAIL_DS"]
token_path_do = CREDENTIALS_DIR / os.environ.get(
    "ai_audit_credentials_fname_do", "token_do.json"
)
token_path_ds = CREDENTIALS_DIR / os.environ.get(
    "ai_audit_credentials_fname_ds", "token_ds.json"
)


def create_test_files(num_files=3, size_kb=10):
    mock_dir = Path(tempfile.mkdtemp())
    private_dir = Path(tempfile.mkdtemp())
    for i in range(num_files):
        (mock_dir / f"mock_{i}.txt").write_text("x" * (size_kb * 1024))
        (private_dir / f"private_{i}.txt").write_text("x" * (size_kb * 1024))
    return mock_dir, private_dir


def benchmark():
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

    # DO creates dataset
    mock_dir, private_dir = create_test_files(num_files=3)
    do.create_dataset(
        name="benchmark_dataset",
        mock_path=mock_dir,
        private_path=private_dir,
        summary="Test dataset",
        users=[ds.email],
    )

    # DS syncs (first time - downloads files)
    start = time.time()
    ds.sync()
    sync_time_1 = time.time() - start

    datasets = ds.datasets.get_all()
    print(f"First sync: {sync_time_1:.2f}s")
    print(f"Datasets loaded: {len(datasets)}")

    # DS syncs again (should skip download due to hash cache)
    start = time.time()
    ds.sync()
    sync_time_2 = time.time() - start
    print(f"Second sync: {sync_time_2:.2f}s (should be faster - no download)")


if __name__ == "__main__":
    benchmark()
