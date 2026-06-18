"""Full inference-service flow with stubs (no real weights needed).

Model owner (DO1) uploads weights through syftbox → inference server loads
them from the synced location → /infer requests are logged into a dataset on
the enclave's OWN datasite (private data never leaves the enclave) → a
researcher's analysis job on those logs needs approval from BOTH data owners.
"""

import json
import os
import random
import tempfile
from pathlib import Path

os.environ["PRE_SYNC"] = "false"

from fastapi.testclient import TestClient

from syft_enclaves import SyftEnclaveClient

from enclave_model_api.log_writer import LOG_FILE_NAME
from enclave_model_api.logs_dataset import ensure_logs_dataset
from enclave_model_api.paths import private_dataset_dir
from enclave_model_api.server import create_app
from enclave_model_api.service import InferenceService

from inference_stub import StubBackend, make_stub_weights

SENTINEL_PROMPT = "SENTINEL_a_private_medical_question_never_to_leave_the_enclave"


def create_model_mock_file() -> Path:
    tmp = Path(tempfile.mkdtemp()) / f"model-mock-{random.randint(1, 1_000_000)}"
    tmp.mkdir(parents=True, exist_ok=True)
    p = tmp / "model_card.txt"
    p.write_text("Gemma 3 270m-IT — served behind POST /infer on the enclave.")
    return p


def create_model_private_dir() -> Path:
    tmp = Path(tempfile.mkdtemp()) / f"weights-{random.randint(1, 1_000_000)}"
    return make_stub_weights(tmp)


def _make_job_code(enclave_email: str) -> str:
    return f'''
import json
import os

import syft_client as sc

log_files = sc.resolve_dataset_files_path(
    "inference_logs", owner_email="{enclave_email}"
)
log_file = [f for f in log_files if f.name == "{LOG_FILE_NAME}"][0]
records = [json.loads(line) for line in open(log_file).read().splitlines()]

os.makedirs("outputs", exist_ok=True)
with open("outputs/log_summary.json", "w") as f:
    json.dump({{
        "total_requests": len(records),
        "avg_elapsed": sum(r["stats"]["elapsed"] for r in records) / len(records),
    }}, f, indent=2)
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


def _make_bioweapon_job_code(enclave_email: str) -> str:
    return f'''
import json
import os

import syft_client as sc

log_files = sc.resolve_dataset_files_path(
    "inference_logs", owner_email="{enclave_email}"
)
log_file = [f for f in log_files if f.name == "{LOG_FILE_NAME}"][0]
records = [json.loads(line) for line in open(log_file).read().splitlines() if line.strip()]
n = sum(1 for r in records if "bio-weapon" in r["prompt"].lower())

os.makedirs("outputs", exist_ok=True)
with open("outputs/summary.json", "w") as f:
    json.dump({{"total_requests": len(records), "bio_weapon_mentions": n}}, f)
'''


def test_demo_flow_do2_submits_and_approves():
    """Mirrors the local docker demo, where the submitter is itself a data owner.

    DO1 = model + log owner, DO2 = submitter AND approver. Logs are written
    directly into the enclave's inference_logs private dir (simulating /infer),
    then DO2 submits a bio-weapon-count job that BOTH data owners approve.
    """
    enclave, do1, do2, _ds = SyftEnclaveClient.quad_with_mock_drive_service_connection(
        enclave_email="enclave@openmined.org",
        do1_email="do1@openmined.org",
        do2_email="do2@openmined.org",
        ds_email="ds@openmined.org",
        use_in_memory_cache=False,
    )

    # Enclave creates the logs dataset on its own datasite
    logs_dir = ensure_logs_dataset(enclave, "inference_logs")

    # Write inference logs directly (3 requests; exactly one mentions bio-weapon)
    records = [
        {
            "id": "1",
            "timestamp": "t",
            "prompt": "How do I bake banana bread?",
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
            "prompt": "What is the capital of France?",
            "completion": "x",
            "stats": {"elapsed": 0.1},
        },
    ]
    (logs_dir / LOG_FILE_NAME).write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )

    # DO2 discovers the logs dataset (mock only)
    enclave.sync()
    do2.sync()
    assert "inference_logs" in [d.name for d in do2.datasets.get_all()]

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

    do2.sync()
    job = do2.jobs["bioweapon_count"]
    assert job.status == "done"
    with open(job.output_paths[0]) as f:
        summary = json.load(f)
    assert summary["total_requests"] == 3
    assert summary["bio_weapon_mentions"] == 1


def test_inference_service_full_flow():
    """Weights in → /infer logged on enclave → jointly-approved analysis job."""
    enclave, model_owner, log_owner, researcher = (
        SyftEnclaveClient.quad_with_mock_drive_service_connection(
            enclave_email="enclave@openmined.org",
            do1_email="model_owner@openmined.org",
            do2_email="log_owner@openmined.org",
            ds_email="researcher@openmined.org",
            use_in_memory_cache=False,
        )
    )

    # Step 1 — Model owner uploads weights through syftbox
    model_owner.create_dataset(
        name="gemma3_model",
        mock_path=create_model_mock_file(),
        private_path=create_model_private_dir(),
        summary="Gemma 3 270m-IT weights (stub)",
        users=[researcher.email, enclave.email],
        upload_private=True,
        sync=False,
    )
    model_owner.share_private_dataset("gemma3_model", enclave.email)
    model_owner.sync()
    enclave.sync()

    # Step 2 — Enclave creates the logs dataset on its OWN datasite
    logs_dir = ensure_logs_dataset(enclave, "inference_logs")
    # .resolve(): on macOS /tmp is a symlink to /private/tmp
    assert (
        logs_dir.resolve()
        == private_dataset_dir(
            enclave.syftbox_folder, enclave.email, "inference_logs"
        ).resolve()
    )

    # Step 3 — Inference server loads the synced weights and serves /infer
    service = InferenceService(
        backend=StubBackend(),
        model_size="270m",
        weights_dir=private_dataset_dir(
            enclave.syftbox_folder, model_owner.email, "gemma3_model"
        ),
        logs_dir=logs_dir,
    )
    assert service.try_load()
    http = TestClient(create_app(service))

    for prompt in [SENTINEL_PROMPT, "What is the capital of NL?"]:
        response = http.post("/infer", json={"query": prompt})
        assert response.status_code == 200

    records = [
        json.loads(line) for line in (logs_dir / LOG_FILE_NAME).read_text().splitlines()
    ]
    assert len(records) == 2
    assert records[0]["prompt"] == SENTINEL_PROMPT

    # Step 4 — Researcher discovers the logs dataset (mock only, no private data)
    enclave.sync()
    researcher.sync()
    researcher_datasets = [d.name for d in researcher.datasets.get_all()]
    assert "inference_logs" in researcher_datasets
    assert not (Path(researcher.syftbox_folder) / enclave.email / "private").exists()
    _assert_no_log_leak(model_owner, log_owner, researcher)

    # Step 5 — Researcher submits an analysis job on the enclave-owned logs
    researcher.submit_python_job(
        enclave.email,
        create_code_file(_make_job_code(enclave.email)),
        "log_analysis",
        datasets={enclave.email: ["inference_logs"]},
    )

    # Step 6 — Enclave receives + distributes; BOTH DOs must approve
    enclave.sync()
    enclave.receive_jobs()

    model_owner.sync()
    log_owner.sync()
    model_owner.approve_job(model_owner.jobs["log_analysis"])
    enclave.sync()
    assert enclave.jobs["log_analysis"].status != "approved"  # one approval missing

    log_owner.approve_job(log_owner.jobs["log_analysis"])
    enclave.sync()
    assert enclave.jobs["log_analysis"].status == "approved"

    # Step 7 — Run and distribute results (submitter only, never the DOs)
    enclave.run_jobs()
    assert enclave.jobs["log_analysis"].status == "done"
    enclave.distribute_results()

    # Step 8 — Researcher receives aggregate results only
    researcher.sync()
    researcher_job = researcher.jobs["log_analysis"]
    assert researcher_job.status == "done"
    with open(researcher_job.output_paths[0]) as f:
        summary = json.load(f)
    assert summary["total_requests"] == 2
    assert summary["avg_elapsed"] > 0

    # Step 9 — Still no raw log record anywhere outside the enclave
    model_owner.sync()
    log_owner.sync()
    _assert_no_log_leak(model_owner, log_owner, researcher)
