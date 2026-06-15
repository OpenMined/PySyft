import json
import os
import random
import tempfile
from pathlib import Path

import pytest

os.environ["PRE_SYNC"] = "false"

from syft_enclaves import SyftEnclaveClient


def create_tmp_dataset_files(prefix=""):
    tmp_dir = (
        Path(tempfile.mkdtemp())
        / f"syft-job-test-{prefix}-{random.randint(1, 1000000)}"
    )
    tmp_dir.mkdir(parents=True, exist_ok=True)
    mock_path = tmp_dir / "mock.txt"
    private_path = tmp_dir / "private.txt"
    mock_path.write_text(f"mock data {prefix}")
    private_path.write_text(f"private data {prefix}")
    return mock_path, private_path


def make_job_code(do1_email: str, do2_email: str) -> str:
    return f"""\
import json
import syft_client as sc

data_path_1 = sc.resolve_dataset_file_path("dataset1", owner_email="{do1_email}")
data_path_2 = sc.resolve_dataset_file_path("dataset2", owner_email="{do2_email}")

with open(data_path_1, "r") as f:
    data1 = f.read()

with open(data_path_2, "r") as f:
    data2 = f.read()

result = {{"total_length": len(data1) + len(data2)}}

with open("outputs/result.json", "w") as f:
    f.write(json.dumps(result))
"""


SIMPLE_JOB_CODE = """\
import json
import os

result = {"status": "ok", "cwd": os.getcwd()}

os.makedirs("outputs", exist_ok=True)
with open("outputs/result.json", "w") as f:
    f.write(json.dumps(result))
"""


def create_tmp_code_file(code: str):
    tmp_dir = Path(tempfile.mkdtemp()) / f"syft-job-code-{random.randint(1, 1000000)}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    code_path = tmp_dir / "main.py"
    code_path.write_text(code)
    return str(code_path)


@pytest.mark.parametrize("encryption", [False, True])
def test_enclave_job_distribution(encryption):
    """Test full flow: DS submits job to enclave, enclave distributes to DOs."""
    enclave, do1, do2, ds = SyftEnclaveClient.quad_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        encryption=encryption,
    )

    # DO1 creates dataset1
    mock1, private1 = create_tmp_dataset_files("ds1")
    do1.create_dataset(
        name="dataset1",
        mock_path=mock1,
        private_path=private1,
        summary="Dataset 1",
        users=[ds.email],
        upload_private=True,
        sync=False,
    )

    # DO2 creates dataset2
    mock2, private2 = create_tmp_dataset_files("ds2")
    do2.create_dataset(
        name="dataset2",
        mock_path=mock2,
        private_path=private2,
        summary="Dataset 2",
        users=[ds.email],
        upload_private=True,
        sync=False,
    )

    # DOs share private datasets with enclave
    do1.share_private_dataset("dataset1", enclave.email)
    do2.share_private_dataset("dataset2", enclave.email)

    # Sync all — DS sees mock datasets
    do1.sync()
    do2.sync()
    ds.sync()
    ds_datasets = ds.datasets.get_all()
    assert len(ds_datasets) == 2

    # DS submits job to enclave
    code_path = create_tmp_code_file(make_job_code(do1.email, do2.email))
    ds.submit_python_job(
        enclave.email,
        code_path,
        "test_job",
        datasets={do1.email: ["dataset1"], do2.email: ["dataset2"]},
    )

    # Enclave syncs to receive job files from DS
    enclave.sync()

    # Enclave distributes job to DOs
    enclave.receive_jobs()

    # DOs sync to receive forwarded job files
    do1.sync()
    do2.sync()

    # Assert DOs received the job
    do1_jobs = do1.jobs
    assert len(do1_jobs) >= 1
    do1_job_names = [j.name for j in do1_jobs]
    assert "test_job" in do1_job_names

    do2_jobs = do2.jobs
    assert len(do2_jobs) >= 1
    do2_job_names = [j.name for j in do2_jobs]
    assert "test_job" in do2_job_names


@pytest.mark.parametrize("encryption", [False, True])
def test_enclave_job_approval_flow(encryption):
    """Test: enclave receives job, distributes to DOs, both approve, enclave sees approved."""
    enclave, do1, do2, ds = SyftEnclaveClient.quad_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        encryption=encryption,
    )

    # DOs create datasets
    mock1, private1 = create_tmp_dataset_files("do1")
    do1.create_dataset(
        name="dataset1",
        mock_path=mock1,
        private_path=private1,
        summary="Dataset 1",
        users=[ds.email],
        upload_private=True,
        sync=False,
    )
    mock2, private2 = create_tmp_dataset_files("do2")
    do2.create_dataset(
        name="dataset2",
        mock_path=mock2,
        private_path=private2,
        summary="Dataset 2",
        users=[ds.email],
        upload_private=True,
        sync=False,
    )
    do1.share_private_dataset("dataset1", enclave.email)
    do2.share_private_dataset("dataset2", enclave.email)
    do1.sync()
    do2.sync()
    ds.sync()

    # DS submits job to enclave
    code_path = create_tmp_code_file(make_job_code(do1.email, do2.email))
    ds.submit_python_job(
        enclave.email,
        code_path,
        "test_job",
        datasets={do1.email: ["dataset1"], do2.email: ["dataset2"]},
    )

    # Enclave receives and distributes
    enclave.sync()
    enclave.receive_jobs()

    # Verify approval files were created on enclave
    enclave_job = enclave.jobs["test_job"]
    review_dir = enclave_job.job_review_path
    assert (review_dir / f"{do1.email}_approval_state.json").exists()
    assert (review_dir / f"{do2.email}_approval_state.json").exists()
    assert enclave_job.status == "pending"
    assert enclave_job.job_headers["job_type"] == "enclave"

    # DOs sync to see the job
    do1.sync()
    do2.sync()

    # DO1 approves
    do1_job = do1.jobs["test_job"]
    assert do1_job.job_headers["job_type"] == "enclave"
    assert do1_job.status == "pending"
    do1.approve_job(do1_job)

    # After DO1 approves but before DO2, enclave still sees pending
    enclave.sync()
    enclave_job = enclave.jobs["test_job"]
    assert enclave_job.status == "pending"

    # DO2 approves
    do2_job = do2.jobs["test_job"]
    do2.approve_job(do2_job)

    # Enclave syncs and sees both approved
    enclave.sync()
    enclave_job = enclave.jobs["test_job"]
    assert enclave_job.status == "approved"


@pytest.mark.parametrize("encryption", [False, True])
def test_enclave_full_job_flow(encryption):
    """Test full flow: submit, distribute, approve, run, share results with DS and DOs."""
    enclave, do1, do2, ds = SyftEnclaveClient.quad_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        encryption=encryption,
    )

    mock1, private1 = create_tmp_dataset_files("do1")
    do1.create_dataset(
        name="dataset1",
        mock_path=mock1,
        private_path=private1,
        summary="Dataset 1",
        users=[ds.email, enclave.email],
        upload_private=True,
        sync=False,
    )
    mock2, private2 = create_tmp_dataset_files("do2")
    do2.create_dataset(
        name="dataset2",
        mock_path=mock2,
        private_path=private2,
        summary="Dataset 2",
        users=[ds.email, enclave.email],
        upload_private=True,
        sync=False,
    )
    do1.share_private_dataset("dataset1", enclave.email)
    do2.share_private_dataset("dataset2", enclave.email)
    do1.sync()
    do2.sync()
    ds.sync()

    code_path = create_tmp_code_file(make_job_code(do1.email, do2.email))
    ds.submit_python_job(
        enclave.email,
        code_path,
        "test_job",
        datasets={do1.email: ["dataset1"], do2.email: ["dataset2"]},
        share_results_with_do=True,
    )

    # Enclave receives and distributes
    enclave.sync()
    enclave.receive_jobs()

    # DOs sync and approve
    do1.sync()
    do2.sync()
    do1.approve_job(do1.jobs["test_job"])
    do2.approve_job(do2.jobs["test_job"])

    # Enclave syncs → approved
    enclave.sync()
    assert enclave.jobs["test_job"].status == "approved"

    # Enclave runs job and distributes results
    enclave.run_jobs()
    enclave.distribute_results()

    # Verify enclave job is done
    enclave_job = enclave.jobs["test_job"]
    assert enclave_job.status == "done"

    # DS syncs and checks result
    ds.sync()
    ds_job = ds.jobs["test_job"]
    assert ds_job.status == "done"
    assert len(ds_job.output_paths) > 0
    with open(ds_job.output_paths[0], "r") as f:
        result = json.loads(f.read())
    assert "total_length" in result
    assert result["total_length"] > 0

    # DOs sync and check they received results
    do1.sync()
    do2.sync()
    do1_job = do1.jobs["test_job"]
    do2_job = do2.jobs["test_job"]
    assert len(do1_job.output_paths) > 0
    assert len(do2_job.output_paths) > 0


def test_approval_gated_on_configured_data_owners():
    """A configured data owner must approve even when the submission doesn't
    reference its dataset — the gate is the enclave's configured data_owners,
    not the submission's datasets."""
    enclave, do1, do2, ds = SyftEnclaveClient.quad_with_mock_drive_service_connection(
        use_in_memory_cache=False,
    )
    # Enclave is configured to require both do1 and do2.
    assert set(enclave.data_owners) == {do1.email, do2.email}

    # DO1 creates a dataset; DO2 has none in this submission.
    mock1, private1 = create_tmp_dataset_files("do1")
    do1.create_dataset(
        name="dataset1",
        mock_path=mock1,
        private_path=private1,
        summary="Dataset 1",
        users=[ds.email],
        upload_private=True,
        sync=False,
    )
    do1.sync()
    ds.sync()

    # DS submits a job referencing ONLY do1's dataset.
    code_path = create_tmp_code_file(SIMPLE_JOB_CODE)
    ds.submit_python_job(
        enclave.email,
        code_path,
        "test_job",
        datasets={do1.email: ["dataset1"]},
    )

    enclave.sync()
    enclave.receive_jobs()

    # Approval files exist for BOTH configured data owners, despite do2 not
    # being referenced in the submission.
    review_dir = enclave.jobs["test_job"].job_review_path
    assert (review_dir / f"{do1.email}_approval_state.json").exists()
    assert (review_dir / f"{do2.email}_approval_state.json").exists()

    do1.sync()
    do2.sync()

    # Only do1 approves — job stays pending (do2 still required).
    do1.approve_job(do1.jobs["test_job"])
    enclave.sync()
    assert enclave.jobs["test_job"].status == "pending"

    # do2 approves — now the gate is satisfied.
    do2.approve_job(do2.jobs["test_job"])
    enclave.sync()
    assert enclave.jobs["test_job"].status == "approved"
