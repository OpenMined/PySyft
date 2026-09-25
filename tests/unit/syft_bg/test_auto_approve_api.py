"""Tests for the auto-approve Python API and live-reload behavior."""

import fcntl
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from syft_bg.api.api import (
    auto_approve,
    auto_approve_job,
    list_auto_approvals,
    remove_auto_approve,
)
from syft_bg.api.utils import (
    get_job_user_files,
    resolve_content_files,
    resolve_job_file_args,
)
from syft_bg.approve import api_store as api_store_module
from syft_bg.approve.api_store import ApiStore
from syft_bg.approve.handlers.job import JobApprovalHandler
from syft_bg.common.config import get_default_paths
from syft_bg.common.syft_bg_config import SyftBgConfig

DO_EMAIL = "do@test.com"


@contextmanager
def _patched_paths(tmp: Path):
    """Point config.yaml and the api lock at tmp, with a DO datasite in tmp."""
    patched = replace(get_default_paths(), config=tmp / "config.yaml")
    with (
        patch("syft_bg.api.api.get_default_paths", return_value=patched),
        patch("syft_bg.api.utils.get_default_paths", return_value=patched),
        patch("syft_bg.approve.config.get_default_paths", return_value=patched),
        patch("syft_bg.common.syft_bg_config.get_default_paths", return_value=patched),
        patch("syft_bg.approve.api_store.get_syftbg_dir", return_value=tmp),
    ):
        SyftBgConfig(do_email=DO_EMAIL, syftbox_root=tmp / "syftbox").save()
        yield ApiStore(tmp / "syftbox", DO_EMAIL)


def _seed(store: ApiStore, tmp: Path, name: str, peer: str = "alice@test.com"):
    script = tmp / f"{name}_main.py"
    script.write_text("print('hi')\n")
    store.create(name, [("main.py", script)], [], [peer])


def _write_script(tmp: Path) -> Path:
    content_dir = tmp / "project"
    content_dir.mkdir()
    script = content_dir / "main.py"
    script.write_text("print('hi')\n")
    return script


class TestListAutoApprovals:
    def test_returns_objects(self, temp_dir):
        with _patched_paths(temp_dir) as store:
            _seed(store, temp_dir, "r1")
            _seed(store, temp_dir, "r2", "bob@test.com")
            result = list_auto_approvals()
            assert set(result.keys()) == {"r1", "r2"}
            assert result["r1"].peers == ["alice@test.com"]
            assert result["r2"].peers == ["bob@test.com"]

    def test_empty(self, temp_dir):
        with _patched_paths(temp_dir):
            assert list_auto_approvals() == {}


class TestRemoveAutoApprove:
    def test_deletes_api_folder(self, temp_dir):
        with _patched_paths(temp_dir) as store:
            _seed(store, temp_dir, "r1")
            _seed(store, temp_dir, "r2")

            result = remove_auto_approve("r1")

            assert result.success is True
            assert result.name == "r1"
            assert not store.api_dir("r1").exists()
            assert set(list_auto_approvals().keys()) == {"r2"}

    def test_unknown_returns_error(self, temp_dir):
        with _patched_paths(temp_dir) as store:
            _seed(store, temp_dir, "r1")
            result = remove_auto_approve("does_not_exist")
            assert result.success is False
            assert "not found" in (result.error or "")
            assert set(list_auto_approvals().keys()) == {"r1"}


class TestAutoApprove:
    def test_stores_api_in_syftbox(self, temp_dir):
        with _patched_paths(temp_dir) as store:
            script = _write_script(temp_dir)
            result = auto_approve(contents=[str(script)], peers=["alice@test.com"])

            assert result.success is True
            api_dir = temp_dir / "syftbox" / DO_EMAIL / "app_data" / "apis" / "main"
            assert (api_dir / "api.yaml").exists()
            assert (api_dir / "files" / "main.py").read_text() == "print('hi')\n"
            assert "objects" not in (temp_dir / "config.yaml").read_text()
            assert store.get("main").peers == ["alice@test.com"]

    @pytest.mark.parametrize("answer,created", [("n", False), ("y", True)])
    def test_no_peers_asks_for_confirmation(self, temp_dir, answer, created):
        with _patched_paths(temp_dir) as store:
            script = _write_script(temp_dir)
            with patch("builtins.input", return_value=answer) as mock_input:
                result = auto_approve(contents=[str(script)])

            mock_input.assert_called_once()
            assert result.success is created
            assert store.exists("main") is created

    def test_no_peers_without_terminal_aborts(self, temp_dir):
        with _patched_paths(temp_dir) as store:
            script = _write_script(temp_dir)
            with patch("builtins.input", side_effect=EOFError):
                result = auto_approve(contents=[str(script)])
            assert result.success is False
            assert store.names() == []

    def test_allow_any_peer_skips_prompt(self, temp_dir):
        with _patched_paths(temp_dir) as store:
            script = _write_script(temp_dir)
            with patch("builtins.input") as mock_input:
                result = auto_approve(contents=[str(script)], allow_any_peer=True)
            mock_input.assert_not_called()
            assert result.success is True
            assert store.get("main").peers == []

    def test_lock_not_held_during_file_io(self, temp_dir):
        with _patched_paths(temp_dir):
            script = _write_script(temp_dir)
            lock_path = temp_dir / "apis.lock"
            observed = {}
            original_copy = api_store_module._copy_and_hash_files

            def _check_lock_then_copy(content_files, files_dir):
                lock_path.touch(exist_ok=True)
                with open(lock_path) as lock_handle:
                    try:
                        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        observed["lock_was_free"] = True
                        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                    except BlockingIOError:
                        observed["lock_was_free"] = False
                return original_copy(content_files, files_dir)

            with patch.object(
                api_store_module,
                "_copy_and_hash_files",
                side_effect=_check_lock_then_copy,
            ):
                result = auto_approve(contents=[str(script)], allow_any_peer=True)

            assert result.success is True
            assert observed["lock_was_free"] is True

    def test_name_collision_picks_next_available_name(self, temp_dir):
        with _patched_paths(temp_dir) as store:
            script = _write_script(temp_dir)
            _seed(store, temp_dir, "main", "racer@test.com")

            result = auto_approve(contents=[str(script)], allow_any_peer=True)

            assert result.success is True
            assert result.name == "main_1"
            objects = list_auto_approvals()
            assert set(objects.keys()) == {"main", "main_1"}
            assert objects["main"].peers == ["racer@test.com"]
            entry = objects["main_1"].file_contents[0]
            assert "main_1" in entry.path
            assert Path(entry.path).read_text() == "print('hi')\n"

    def test_orphaned_directory_at_target_name_returns_failure(self, temp_dir):
        """A stale, non-empty folder without api.yaml at the resolved name must
        give a clean failure, and the staging dir must not leak."""
        with _patched_paths(temp_dir) as store:
            script = _write_script(temp_dir)
            orphan_dir = store.api_dir("main")
            orphan_dir.mkdir(parents=True)
            (orphan_dir / "leftover.txt").write_text("stale")

            result = auto_approve(contents=[str(script)], allow_any_peer=True)

            assert result.success is False
            assert "main" in (result.error or "")
            assert list_auto_approvals() == {}
            assert [p.name for p in store.apis_dir.iterdir()] == ["main"]

    def test_staging_dir_cleaned_up_when_copy_and_hash_fails(self, temp_dir):
        with _patched_paths(temp_dir) as store:
            script = _write_script(temp_dir)
            with patch.object(
                api_store_module,
                "_copy_and_hash_files",
                side_effect=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad byte"),
            ):
                result = auto_approve(contents=[str(script)], allow_any_peer=True)

            assert result.success is False
            assert list(store.apis_dir.iterdir()) == []


class TestHandlerReloadsApis:
    """The approve service must pick up api changes without a restart."""

    def _write_submission(self, base_dir: Path) -> Path:
        """A submission tree: code/main.py plus the root files the runner reads."""
        (base_dir / "code").mkdir(parents=True)
        (base_dir / "code" / "main.py").write_text("print('hello')\n")
        (base_dir / "run.sh").write_text("#!/bin/bash\npython code/main.py\n")
        (base_dir / "config.yaml").write_text("name: test-job\n")
        return base_dir

    def _make_test_job(
        self, submission_dir: Path, submitted_by: str = "alice@test.com"
    ):
        job = MagicMock()
        job.name = "test-job"
        job.status = "pending"
        job.submitted_by = submitted_by
        job.job_submission_path = submission_dir
        job.code_dir = submission_dir / "code"
        job.files = []
        return job

    def _create_matching_api(self, store: ApiStore, submission_dir: Path) -> None:
        content_files = [
            (f.relative_to(submission_dir).as_posix(), f)
            for f in sorted(submission_dir.rglob("*"))
            if f.is_file()
        ]
        store.create("r1", content_files, [], ["alice@test.com"])

    def _make_handler(self, temp_dir: Path) -> tuple[JobApprovalHandler, ApiStore]:
        syftbox_root = temp_dir / "syftbox"
        client = MagicMock(syftbox_folder=syftbox_root, email=DO_EMAIL)
        handler = JobApprovalHandler(client=client, config_path=temp_dir / "c.yaml")
        return handler, ApiStore(syftbox_root, DO_EMAIL)

    def test_picks_up_added_and_removed_api(self, temp_dir):
        submission = self._write_submission(temp_dir / "job")
        handler, store = self._make_handler(temp_dir)
        job = self._make_test_job(submission)

        assert handler.evaluate_auto_approval(job).match is False

        self._create_matching_api(store, submission)
        assert handler.evaluate_auto_approval(job).match is True

        store.delete("r1")
        assert handler.evaluate_auto_approval(job).match is False


class TestJobUserFiles:
    """What an approval object is built from."""

    FILES = (
        "code/main.py",
        "run.sh",
        "config.yaml",
        "code/outputs/result.json",
        "code/.venv/bin/python",
        "code/__pycache__/main.pyc",
    )

    def _submission(self, temp_dir: Path, status: str) -> MagicMock:
        submission = temp_dir / status
        for rel in self.FILES:
            path = submission / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x")
        job = MagicMock()
        job.job_submission_path = submission
        job.status = status
        return job

    def test_pending_job_keeps_every_file(self, temp_dir):
        """Nothing has run, so these are files the submitter shipped.

        Every later job of that shape carries them too, and the matcher counts
        them, so a rule that left them out could never match.
        """
        assert set(get_job_user_files(self._submission(temp_dir, "pending"))) == {
            "code/main.py",
            "run.sh",
            "config.yaml",
            "code/outputs/result.json",
            "code/.venv/bin/python",
            "code/__pycache__/main.pyc",
        }

    def test_ran_job_drops_run_created_dirs(self, temp_dir):
        """`_prepare_outputs_dir` makes code/outputs/ and run.sh makes .venv/.

        A pending job carries neither, so pinning them would match nothing.
        """
        assert set(get_job_user_files(self._submission(temp_dir, "done"))) == {
            "code/main.py",
            "run.sh",
            "config.yaml",
        }


class TestAutoApproveJobUnreadableFiles:
    """A pending job can ship files that are not UTF-8 text."""

    def _pending_job(self, temp_dir: Path) -> MagicMock:
        submission = temp_dir / "job"
        (submission / "code" / "__pycache__").mkdir(parents=True)
        (submission / "code" / "main.py").write_text("print(1)")
        (submission / "code" / "__pycache__" / "main.pyc").write_bytes(
            b"\xff\xfe\x00binary"
        )
        (submission / "run.sh").write_text("python code/main.py")
        (submission / "config.yaml").write_text("name: job")
        job = MagicMock()
        job.job_submission_path = submission
        job.status = "pending"
        job.name = "job"
        job.submitted_by = "alice@test.com"
        job.datasite_owner_email = DO_EMAIL
        job._client.config.syftbox_folder = temp_dir / "syftbox"
        return job

    def test_returns_error_and_writes_nothing(self, temp_dir):
        """The matcher cannot read the file either, so no rule could match."""
        job = self._pending_job(temp_dir)
        with _patched_paths(temp_dir) as store:
            result = auto_approve_job(job)

            assert result.success is False
            assert "code/__pycache__/main.pyc" in result.error
            assert store.names() == []


_USER_FILES = {
    "run.sh": Path("/j/run.sh"),
    "config.yaml": Path("/j/config.yaml"),
    "code/main.py": Path("/j/code/main.py"),
    "code/utils/helpers.py": Path("/j/code/utils/helpers.py"),
    "code/a/helpers.py": Path("/j/code/a/helpers.py"),
}


class TestResolveJobFileArgs:
    """Owner-typed names resolve against paths from the submission root."""

    def test_exact_path_is_kept(self):
        assert resolve_job_file_args(_USER_FILES, ["code/main.py"]) == (
            ["code/main.py"],
            None,
        )

    def test_bare_name_resolves(self):
        assert resolve_job_file_args(_USER_FILES, ["main.py"]) == (
            ["code/main.py"],
            None,
        )

    def test_path_relative_to_code_resolves(self):
        """The form the API took when paths were relative to code/."""
        assert resolve_job_file_args(_USER_FILES, ["utils/helpers.py"]) == (
            ["code/utils/helpers.py"],
            None,
        )

    def test_dot_prefix_resolves(self):
        assert resolve_job_file_args(_USER_FILES, ["./code/main.py"]) == (
            ["code/main.py"],
            None,
        )

    def test_ambiguous_name_is_an_error(self):
        resolved, error = resolve_job_file_args(_USER_FILES, ["helpers.py"])
        assert resolved == []
        assert "several files" in error

    def test_partial_component_does_not_match(self):
        """A suffix matches whole path components only."""
        resolved, error = resolve_job_file_args(_USER_FILES, ["ils/helpers.py"])
        assert resolved == []
        assert "not found" in error


class TestResolveContentFilesBaseDir:
    """With a base directory, the stored path is the one a job is matched on."""

    def _submission(self, temp_dir: Path) -> Path:
        submission = temp_dir / "job"
        (submission / "code").mkdir(parents=True)
        (submission / "code" / "main.py").write_text("print(1)")
        (submission / "run.sh").write_text("python code/main.py")
        return submission

    def test_dot_prefix_is_normalized(self, temp_dir):
        base = self._submission(temp_dir)
        files, error = resolve_content_files(["./code/main.py", "./run.sh"], base)
        assert error is None
        assert [rel for rel, _ in files] == ["code/main.py", "run.sh"]

    def test_absolute_path_inside_base_is_made_relative(self, temp_dir):
        base = self._submission(temp_dir)
        files, error = resolve_content_files([str(base / "code" / "main.py")], base)
        assert error is None
        assert [rel for rel, _ in files] == ["code/main.py"]

    def test_path_outside_base_is_an_error(self, temp_dir):
        base = self._submission(temp_dir)
        (temp_dir / "outside.py").write_text("x")
        files, error = resolve_content_files(["../outside.py"], base)
        assert files == []
        assert "outside" in error
