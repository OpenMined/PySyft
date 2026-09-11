"""How a job is addressed: the subscript chain, and the hint that teaches it.

These drive JobClient over a tmp_path SyftBox folder, like the lifecycle tests
next door, but assert on what jobs[...] returns and what the table's hint says
rather than on a job running.
"""

import re
from pathlib import Path

import pytest
from syft_job.client import JobClient
from syft_job.config import SyftJobConfig

DO_EMAIL = "do@test.org"
DS_EMAIL = "ds@test.org"
PEER_EMAIL = "peer@test.org"
DS2_EMAIL = "ds2@test.org"

MAIN_PY = """\
import os

print("hello from job")
os.makedirs("outputs", exist_ok=True)
with open("outputs/result.txt", "w") as f:
    f.write("done")
"""


def test_hint_names_submitter_and_job(tmp_path: Path):
    """The DO hint must point at jobs["submitter"]["name"], not jobs[0].

    Positional indexing is not safe to recommend: positions shift as jobs are
    added. A bare name is not safe either — it searches every datasite, so a
    peer's job of the same name can answer instead. The email in a chain is the
    other party, which for a data owner is the submitter.
    """
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    code_file = tmp_path / "main.py"
    code_file.write_text(MAIN_PY)

    # has_do_role gates the hint — it is only shown to a data owner.
    do_config = SyftJobConfig(
        syftbox_folder=syftbox, current_user_email=DO_EMAIL, has_do_role=True
    )
    ds_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS_EMAIL)
    ds_client = JobClient(config=ds_config)
    do_client = JobClient(config=do_config)

    ds_client.submit_python_job(
        user=DO_EMAIL, code_path=str(code_file), job_name="analysis.job"
    )

    jobs = do_client.jobs
    text = str(jobs)
    html = jobs._repr_html_()

    for rendering in (text, html):
        assert f'jobs["{DS_EMAIL}"]["analysis.job"].approve()' in rendering
        assert "jobs[0]" not in rendering
        assert DO_EMAIL not in rendering.split("💡")[1], "the DO's own email is noise"

    # The hint names a job the DO can actually approve, as written.
    assert do_client.jobs[DS_EMAIL]["analysis.job"].status == "pending"


def _index_name_pairs_from_text(text: str) -> list[tuple[int, str]]:
    return [(int(i), name) for i, name in re.findall(r"\[(\d+)\s*\]\s+(\S+)", text)]


def _index_name_pairs_from_html(html: str) -> list[tuple[int, str]]:
    return [
        (int(i), name.strip())
        for i, name in re.findall(
            r'class="syftjob-index">\[(\d+)\]</span>.*?'
            r'class="syftjob-td syftjob-job-name">\s*([^<]+)',
            html,
            flags=re.DOTALL,
        )
    ]


def test_jobs_table_index_matches_getitem(tmp_path: Path):
    """The [N] printed in the table must be the N that jobs[N] returns.

    The table groups by datasite owner (root first, then peers). __getitem__
    used to subscript a newest-first list, so with two owners the row labelled
    [0] was not jobs[0].
    """
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    code_file = tmp_path / "main.py"
    code_file.write_text(MAIN_PY)

    do_config = SyftJobConfig(
        syftbox_folder=syftbox, current_user_email=DO_EMAIL, has_do_role=True
    )
    ds_config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS_EMAIL)
    ds_client = JobClient(config=ds_config)
    do_client = JobClient(config=do_config)

    # Older job on the root datasite, then a newer one on a peer datasite.
    ds_client.submit_python_job(
        user=DO_EMAIL, code_path=str(code_file), job_name="older-root-job"
    )
    ds_client.submit_python_job(
        user=PEER_EMAIL, code_path=str(code_file), job_name="newest-peer-job"
    )

    jobs = do_client.jobs
    assert [job.name for job in jobs] == ["older-root-job", "newest-peer-job"]

    text_pairs = _index_name_pairs_from_text(str(jobs))
    html_pairs = _index_name_pairs_from_html(jobs._repr_html_())
    assert text_pairs == [(0, "older-root-job"), (1, "newest-peer-job")]
    assert html_pairs == [(0, "older-root-job"), (1, "newest-peer-job")]

    for index, name in text_pairs + html_pairs:
        assert jobs[index].name == name


def test_hint_names_submitter_for_shared_name(tmp_path: Path):
    """A name two submitters share needs the submitter to reach one job.

    Job names are unique per datasite and submitter, so two data scientists can
    submit "analysis" to the same data owner. The hint adds the submitter rather
    than giving up on the name.
    """
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    code_file = tmp_path / "main.py"
    code_file.write_text(MAIN_PY)

    do_config = SyftJobConfig(
        syftbox_folder=syftbox, current_user_email=DO_EMAIL, has_do_role=True
    )
    do_client = JobClient(config=do_config)
    ds1_client = JobClient(
        config=SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS_EMAIL)
    )
    ds2_client = JobClient(
        config=SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS2_EMAIL)
    )

    # "solo" first, so the ambiguous name is the newest and would otherwise win.
    ds1_client.submit_python_job(
        user=DO_EMAIL, code_path=str(code_file), job_name="solo"
    )
    ds1_client.submit_python_job(
        user=DO_EMAIL, code_path=str(code_file), job_name="analysis"
    )
    ds2_client.submit_python_job(
        user=DO_EMAIL, code_path=str(code_file), job_name="analysis"
    )

    jobs = do_client.jobs
    for rendering in (str(jobs), jobs._repr_html_()):
        assert f'jobs["{DS2_EMAIL}"]["analysis"].approve()' in rendering

    # The chain the hint gives reaches exactly one job, and the datasite alone
    # does not, which is why it names the submitter.
    job = jobs[DS2_EMAIL]["analysis"]
    assert (job.submitted_by, job.status) == (DS2_EMAIL, "pending")
    with pytest.raises(ValueError, match="Multiple jobs are named 'analysis'"):
        jobs[DO_EMAIL]["analysis"]


def test_hint_names_every_submitter_of_shared_name(tmp_path: Path):
    """Every job in the table shares its name, so every chain needs a submitter.

    The datasite alone reaches neither, and the hint still has to name something
    the data owner can run.
    """
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    code_file = tmp_path / "main.py"
    code_file.write_text(MAIN_PY)

    do_config = SyftJobConfig(
        syftbox_folder=syftbox, current_user_email=DO_EMAIL, has_do_role=True
    )
    do_client = JobClient(config=do_config)
    for ds_email in (DS_EMAIL, DS2_EMAIL):
        JobClient(
            config=SyftJobConfig(syftbox_folder=syftbox, current_user_email=ds_email)
        ).submit_python_job(
            user=DO_EMAIL, code_path=str(code_file), job_name="analysis"
        )

    jobs = do_client.jobs
    for rendering in (str(jobs), jobs._repr_html_()):
        assert f'jobs["{DS2_EMAIL}"]["analysis"].approve()' in rendering
        assert "jobs[0]" not in rendering

    for ds_email in (DS_EMAIL, DS2_EMAIL):
        assert jobs[ds_email]["analysis"].submitted_by == ds_email


def test_no_hint_when_do_owns_no_jobs(tmp_path: Path):
    """Every subscript would name a job on another datasite, which approve() refuses.

    A hint is worse than no hint when the only command it can give raises
    PermissionError.
    """
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    code_file = tmp_path / "main.py"
    code_file.write_text(MAIN_PY)

    do_config = SyftJobConfig(
        syftbox_folder=syftbox, current_user_email=DO_EMAIL, has_do_role=True
    )
    do_client = JobClient(config=do_config)
    JobClient(
        config=SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS_EMAIL)
    ).submit_python_job(
        user=PEER_EMAIL, code_path=str(code_file), job_name="peer-owned.job"
    )

    jobs = do_client.jobs
    assert [job.datasite_owner_email for job in jobs] == [PEER_EMAIL]
    for rendering in (str(jobs), jobs._repr_html_()):
        assert "peer-owned.job" in rendering
        assert "💡" not in rendering


def test_email_key_resolves_name_shared_by_two_datasites(tmp_path: Path):
    """One submitter can send the same job name to two data owners.

    The bare name has no way to choose between them and raises. Selecting the
    datasite first leaves one job, which is the whole point of the two-step
    subscript.
    """
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    code_file = tmp_path / "main.py"
    code_file.write_text(MAIN_PY)

    ds_client = JobClient(
        config=SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS_EMAIL)
    )
    for owner in (DO_EMAIL, PEER_EMAIL):
        ds_client.submit_python_job(
            user=owner, code_path=str(code_file), job_name="analysis.job"
        )
        JobClient(
            config=SyftJobConfig(syftbox_folder=syftbox, current_user_email=owner)
        ).scan_inbox()

    jobs = ds_client.jobs
    with pytest.raises(ValueError, match="Multiple jobs are named 'analysis.job'"):
        jobs["analysis.job"]

    for owner in (DO_EMAIL, PEER_EMAIL):
        job = jobs[owner]["analysis.job"]
        assert job.datasite_owner_email == owner


def test_email_key_names_emails_it_has(tmp_path: Path):
    """An email with no jobs on it is a typo the message has to help with."""
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    code_file = tmp_path / "main.py"
    code_file.write_text(MAIN_PY)

    ds_client = JobClient(
        config=SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS_EMAIL)
    )
    ds_client.submit_python_job(
        user=DO_EMAIL, code_path=str(code_file), job_name="analysis.job"
    )

    with pytest.raises(ValueError) as exc:
        ds_client.jobs["nobody@test.org"]

    message = str(exc.value)
    assert "nobody@test.org" in message
    assert DO_EMAIL in message, "must name the datasites that do have jobs"
    assert DS_EMAIL in message, "must name the submitters too"


def test_hint_adds_datasite_when_submitter_used_name_twice(
    tmp_path: Path,
):
    """One submitter can send the same name to the DO and to a peer.

    The submitter alone then reaches both, so this is the one case where the
    DO's own datasite has to join the chain, and the only thing that earns the
    third key.
    """
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    code_file = tmp_path / "main.py"
    code_file.write_text(MAIN_PY)

    do_client = JobClient(
        config=SyftJobConfig(
            syftbox_folder=syftbox, current_user_email=DO_EMAIL, has_do_role=True
        )
    )
    ds_client = JobClient(
        config=SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS_EMAIL)
    )
    for owner in (DO_EMAIL, PEER_EMAIL):
        ds_client.submit_python_job(
            user=owner, code_path=str(code_file), job_name="analysis.job"
        )
        JobClient(
            config=SyftJobConfig(syftbox_folder=syftbox, current_user_email=owner)
        ).scan_inbox()

    jobs = do_client.jobs
    assert {job.datasite_owner_email for job in jobs} == {DO_EMAIL, PEER_EMAIL}
    chain = f'jobs["{DO_EMAIL}"]["{DS_EMAIL}"]["analysis.job"]'
    for rendering in (str(jobs), jobs._repr_html_()):
        assert f"{chain}.approve()" in rendering

    # The submitter alone reaches both datasites, so the chain needs both keys.
    with pytest.raises(ValueError, match="Multiple jobs are named 'analysis.job'"):
        jobs[DS_EMAIL]["analysis.job"]
    assert jobs[DO_EMAIL][DS_EMAIL]["analysis.job"].datasite_owner_email == DO_EMAIL


def test_job_name_cannot_contain_at_sign(tmp_path: Path):
    """An '@' is how a subscript tells a datasite email from a job name."""
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    code_file = tmp_path / "main.py"
    code_file.write_text(MAIN_PY)

    ds_client = JobClient(
        config=SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS_EMAIL)
    )
    with pytest.raises(ValueError, match="cannot contain '@'"):
        ds_client.submit_python_job(
            user=DO_EMAIL, code_path=str(code_file), job_name="ds@test.org"
        )
