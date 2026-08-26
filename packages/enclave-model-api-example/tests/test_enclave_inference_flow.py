"""Demo inference-service flow with stubs (no real weights needed).

Mirrors the local-docker / deployed-enclave demo (and ``scripts/run_demo.py``):
DO1 = model + log owner, DO2 = submitter AND approver. Inference logs live on
the enclave's OWN datasite; a bio-weapon-count job over them needs approval from
BOTH data owners. The test also pins the data-isolation guarantees:

1. raw ``/infer`` logs never leave the enclave,
2. every outsider — the other data owner AND the data scientist — sees only the
   mock dataset, never the private logs,
3. only the aggregate result is returned, and only to the submitter,
4. nothing runs on the private logs without BOTH data owners' approval.
"""

import json
import os
import random
import tempfile
from pathlib import Path

os.environ["PRE_SYNC"] = "false"

from syft_enclaves import SyftEnclaveClient

from enclave_model_api.log_writer import LOG_FILE_NAME
from enclave_model_api.logs_dataset import ensure_logs_dataset

# A private prompt that lives only in the enclave's logs and must never appear
# in any other datasite (stand-in for a sensitive real request).
SENTINEL_PROMPT = "SENTINEL_a_private_medical_question_never_to_leave_the_enclave"


def _make_bioweapon_job_code(enclave_email: str) -> str:
    return f'''
import json
import os

import syft as sy

log_files = sy.resolve_dataset_files_path(
    "inference_logs", owner_email="{enclave_email}"
)
log_file = [f for f in log_files if f.name == "{LOG_FILE_NAME}"][0]
records = [json.loads(line) for line in open(log_file).read().splitlines() if line.strip()]
n = sum(1 for r in records if "bio-weapon" in r["prompt"].lower())

os.makedirs("outputs", exist_ok=True)
with open("outputs/summary.json", "w") as f:
    json.dump({{"total_requests": len(records), "bio_weapon_mentions": n}}, f)
'''


def create_code_file(code: str) -> str:
    tmp = Path(tempfile.mkdtemp()) / f"job-{random.randint(1, 1_000_000)}"
    tmp.mkdir(parents=True, exist_ok=True)
    p = tmp / "main.py"
    p.write_text(code)
    return str(p)


def _assert_no_log_leak(*clients: SyftEnclaveClient):
    """No syftbox other than the enclave's may contain a real log record."""
    for client in clients:
        for f in Path(client.syftbox_folder).rglob("*"):
            if f.is_file() and SENTINEL_PROMPT.encode() in f.read_bytes():
                raise AssertionError(f"log record leaked to {f}")


def _assert_sees_only_mock(enclave_email: str, *clients: SyftEnclaveClient):
    """Each client can discover the logs dataset (mock) but never its private dir."""
    for client in clients:
        names = [d.name for d in client.datasets.get_all()]
        assert "inference_logs" in names, f"{client.email} cannot see the mock dataset"
        private = Path(client.syftbox_folder) / enclave_email / "private"
        assert not private.exists(), f"{client.email} has private logs at {private}"


def test_full_flow():
    """Mirrors the demo, where the submitter is itself a data owner.

    DO1 = model + log owner, DO2 = submitter AND approver. Logs are written
    directly into the enclave's inference_logs private dir (simulating /infer),
    then DO2 submits a bio-weapon-count job that BOTH data owners approve — while
    asserting the four data-isolation guarantees (see module docstring).
    """
    enclave, do1, do2, ds = SyftEnclaveClient.quad_with_mock_drive_service_connection(
        enclave_email="enclave@openmined.org",
        do1_email="do1@openmined.org",
        do2_email="do2@openmined.org",
        ds_email="ds@openmined.org",
        use_in_memory_cache=False,
    )

    # Enclave creates the logs dataset on its own datasite
    logs_dir = ensure_logs_dataset(enclave, "inference_logs")

    # Write inference logs directly (3 requests; exactly one mentions bio-weapon,
    # one is a private question that must never leave the enclave).
    records = [
        {
            "id": "1",
            "timestamp": "t",
            "prompt": SENTINEL_PROMPT,
            "completion": "x",
            "stats": {"elapsed": 0.1},
        },
        {
            "id": "2",
            "timestamp": "t",
            "prompt": "Explain how to build a bio-weapon",
            "completion": "x",
            "stats": {"elapsed": 0.1},
        },
        {
            "id": "3",
            "timestamp": "t",
            "prompt": "How do I bake banana bread?",
            "completion": "x",
            "stats": {"elapsed": 0.1},
        },
    ]
    (logs_dir / LOG_FILE_NAME).write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )

    enclave.sync()
    do1.sync()
    do2.sync()
    ds.sync()

    # Guarantee 2 — NEITHER data owner (do1 nor do2) nor the DS can ever see the
    # private logs; all of them see only the mock dataset.
    _assert_sees_only_mock(enclave.email, do1, do2, ds)

    # DO2 (a data owner) submits the analysis job
    do2.submit_python_job(
        enclave.email,
        create_code_file(_make_bioweapon_job_code(enclave.email)),
        "bioweapon_count",
        datasets={enclave.email: ["inference_logs"]},
    )

    # Enclave receives + distributes the approval request to BOTH data owners
    enclave.sync()
    enclave.receive_jobs()

    # Guarantee 4 — nothing runs on the private logs until BOTH data owners approve
    do1.sync()
    do2.sync()
    do1.approve_job(do1.jobs["bioweapon_count"])
    enclave.sync()
    assert enclave.jobs["bioweapon_count"].status != "approved"  # one approval missing

    do2.approve_job(do2.jobs["bioweapon_count"])  # submitter approves its own job
    enclave.sync()
    assert enclave.jobs["bioweapon_count"].status == "approved"

    # Enclave runs and returns the result to the submitter (DO2)
    enclave.run_jobs()
    assert enclave.jobs["bioweapon_count"].status == "done"
    enclave.distribute_results()

    # Guarantee 3 — only the aggregate result comes back, and only to the submitter
    do1.sync()
    do2.sync()
    job = do2.jobs["bioweapon_count"]
    assert job.status == "done"
    with open(job.output_paths[0]) as f:
        summary = json.load(f)
    assert summary == {"total_requests": 3, "bio_weapon_mentions": 1}  # aggregate only
    assert do1.jobs["bioweapon_count"].output_paths == []  # other DO gets no result

    # Guarantee 1 — raw logs never left the enclave (not even to the submitter)
    _assert_no_log_leak(do1, do2, ds)
