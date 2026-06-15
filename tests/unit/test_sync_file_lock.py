"""Feature test for the sync file lock.

Verifies that two SyftboxManagers in different OS processes cannot run
sync() concurrently. If the file lock is disabled or broken, this test fails.

Failure modes caught:
1. sync() stops calling _sync_file_lock() → no log entries → fails on count
2. _sync_file_lock() is broken (no flock call) → intervals overlap → fails
3. Cross-process serialization is broken → intervals overlap → fails
"""

import multiprocessing
import tempfile
import time
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch

from syft_client.sync.syftbox_manager import SyftboxManager


def _subprocess_sync_with_instrumented_lock(
    syftbox_folder_str: str,
    label: str,
    log_path_str: str,
    iterations: int,
    hold_ms: int,
):
    """Run inside a child process.

    Creates its own SyftboxManager pointing at the shared syftbox_folder,
    patches _sync_file_lock to record interval timings, then calls sync()
    multiple times. The intervals are written to the shared log file.
    """
    log_path = Path(log_path_str)

    _, manager = SyftboxManager.pair_with_mock_drive_service_connection(
        base_path1=Path(syftbox_folder_str),
        use_in_memory_cache=True,
    )

    original_lock = SyftboxManager._sync_file_lock

    @contextmanager
    def instrumented_lock(self):
        with original_lock(self):
            start = time.time()
            time.sleep(hold_ms / 1000.0)
            try:
                yield
            finally:
                end = time.time()
                with open(log_path, "a") as f:
                    f.write(f"{label},{start},{end}\n")

    with patch.object(SyftboxManager, "_sync_file_lock", instrumented_lock):
        for _ in range(iterations):
            try:
                manager.sync(auto_checkpoint=False)
            except Exception as e:
                # sync may fail due to mock drive state between subprocesses,
                # but the lock will still have been acquired and released.
                print(f"[{label}] sync raised (expected): {e}")


def test_sync_prevents_concurrent_sync_across_processes():
    """Two SyftboxManagers in different OS processes must not sync concurrently.

    This is the core feature: if the file lock is removed from sync() or broken,
    the two processes' sync intervals will overlap and this test will fail.
    """
    with tempfile.TemporaryDirectory() as tmp_root:
        shared_base = Path(tmp_root) / "shared"
        shared_base.mkdir()
        log_path = Path(tmp_root) / "intervals.log"
        log_path.touch()

        HOLD_MS = 150
        ITERATIONS = 3

        ctx = multiprocessing.get_context("spawn")
        p1 = ctx.Process(
            target=_subprocess_sync_with_instrumented_lock,
            args=(str(shared_base), "A", str(log_path), ITERATIONS, HOLD_MS),
        )
        p2 = ctx.Process(
            target=_subprocess_sync_with_instrumented_lock,
            args=(str(shared_base), "B", str(log_path), ITERATIONS, HOLD_MS),
        )

        p1.start()
        time.sleep(0.05)  # slight stagger so p1 likely acquires first
        p2.start()
        p1.join(timeout=60)
        p2.join(timeout=60)

        assert p1.exitcode == 0, f"subprocess A exited with {p1.exitcode}"
        assert p2.exitcode == 0, f"subprocess B exited with {p2.exitcode}"

        # Parse intervals: each line is "label,start,end"
        raw_lines = [ln for ln in log_path.read_text().strip().split("\n") if ln]
        intervals = []
        for line in raw_lines:
            label, start_s, end_s = line.split(",")
            intervals.append((label, float(start_s), float(end_s)))

        # Assertion 1: the lock was actually entered.
        # If sync() doesn't call _sync_file_lock(), the instrumentation never
        # fires and we get 0 intervals.
        expected_total = ITERATIONS * 2
        assert len(intervals) == expected_total, (
            f"Expected {expected_total} lock acquisitions "
            f"(ITERATIONS={ITERATIONS} × 2 processes), got {len(intervals)}. "
            f"This means sync() is not calling _sync_file_lock()."
        )

        # Assertion 2: no two intervals overlap.
        # If they overlap, the lock is not serializing across processes.
        intervals.sort(key=lambda x: x[1])
        for i in range(len(intervals) - 1):
            a_label, a_start, a_end = intervals[i]
            b_label, b_start, b_end = intervals[i + 1]
            assert b_start >= a_end, (
                f"Concurrent sync detected across processes!\n"
                f"  {a_label}: [{a_start:.3f}, {a_end:.3f}]\n"
                f"  {b_label}: [{b_start:.3f}, {b_end:.3f}]\n"
                f"B started before A ended — the file lock is not preventing "
                f"concurrent sync."
            )

        # Assertion 3: both processes actually ran.
        labels_seen = {label for label, _, _ in intervals}
        assert labels_seen == {"A", "B"}, (
            f"Expected both processes to run, saw labels: {labels_seen}"
        )
