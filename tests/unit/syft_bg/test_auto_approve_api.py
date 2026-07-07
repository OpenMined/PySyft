"""Tests for the list/remove auto-approve Python API and config-reload behavior."""

import fcntl
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch


from syft_bg.api.api import auto_approve, list_auto_approvals, remove_auto_approve
from syft_bg.api.utils import copy_and_hash_files
from syft_bg.approve.config import (
    AutoApprovalObj,
    AutoApprovalsConfig,
    AutoApproveConfig,
    FileEntry,
)
from syft_bg.approve.handlers.job import JobApprovalHandler
from syft_bg.common.config import get_default_paths
from syft_bg.common.syft_bg_config import SyftBgConfig


@contextmanager
def _patched_paths(tmp: Path):
    """Redirect default paths so list/remove API operates against a tmp config."""
    original = get_default_paths()
    patched = replace(
        original,
        config=tmp / "config.yaml",
        auto_approvals_dir=tmp / "auto_approvals",
    )
    with (
        patch("syft_bg.api.api.get_default_paths", return_value=patched),
        patch("syft_bg.api.utils.get_default_paths", return_value=patched),
        patch("syft_bg.approve.config.get_default_paths", return_value=patched),
        patch("syft_bg.common.syft_bg_config.get_default_paths", return_value=patched),
    ):
        yield patched


def _seed_config(tmp: Path, objects: dict[str, AutoApprovalObj]) -> Path:
    """Write a config YAML with the given auto-approval objects to tmp/config.yaml."""
    config_path = tmp / "config.yaml"
    approve_config = AutoApproveConfig(auto_approvals=AutoApprovalsConfig(objects=objects))
    SyftBgConfig(approve=approve_config).save(config_path)
    return config_path


def _make_obj(peer: str = "alice@test.com") -> AutoApprovalObj:
    return AutoApprovalObj(
        file_contents=[
            FileEntry(relative_path="main.py", path="/tmp/main.py", hash="sha256:abc")
        ],
        peers=[peer],
    )


class TestListAutoApprovals:
    def test_returns_objects(self, temp_dir):
        with _patched_paths(temp_dir):
            _seed_config(temp_dir, {"r1": _make_obj(), "r2": _make_obj("bob@test.com")})
            result = list_auto_approvals()
            assert set(result.keys()) == {"r1", "r2"}
            assert result["r1"].peers == ["alice@test.com"]
            assert result["r2"].peers == ["bob@test.com"]

    def test_empty(self, temp_dir):
        with _patched_paths(temp_dir):
            _seed_config(temp_dir, {})
            assert list_auto_approvals() == {}


class TestRemoveAutoApprove:
    def test_deletes_object_and_files(self, temp_dir):
        with _patched_paths(temp_dir):
            _seed_config(temp_dir, {"r1": _make_obj(), "r2": _make_obj()})
            obj_dir = temp_dir / "auto_approvals" / "r1"
            obj_dir.mkdir(parents=True)
            (obj_dir / "main.py").write_text("print('hi')\n")

            result = remove_auto_approve("r1")

            assert result.success is True
            assert result.name == "r1"
            assert not obj_dir.exists()
            remaining = list_auto_approvals()
            assert set(remaining.keys()) == {"r2"}

    def test_unknown_returns_error(self, temp_dir):
        with _patched_paths(temp_dir):
            _seed_config(temp_dir, {"r1": _make_obj()})
            result = remove_auto_approve("does_not_exist")
            assert result.success is False
            assert "not found" in (result.error or "")
            assert set(list_auto_approvals().keys()) == {"r1"}

    def test_no_files_dir_still_succeeds(self, temp_dir):
        """Removing an object whose files dir doesn't exist shouldn't error."""
        with _patched_paths(temp_dir):
            _seed_config(temp_dir, {"r1": _make_obj()})
            result = remove_auto_approve("r1")
            assert result.success is True
            assert list_auto_approvals() == {}

    def test_unknown_does_not_create_config_file(self, temp_dir):
        """A no-op failure (nothing to remove) must not write config.yaml
        into existence — the not-found path must not trigger edit()'s save."""
        with _patched_paths(temp_dir) as patched:
            assert not patched.config.exists()

            result = remove_auto_approve("does_not_exist")

            assert result.success is False
            assert not patched.config.exists()


class TestAutoApproveLockScope:
    """auto_approve() must not hold the config lock during file I/O."""

    def test_lock_not_held_during_file_io(self, temp_dir):
        with _patched_paths(temp_dir) as patched:
            content_dir = temp_dir / "project"
            content_dir.mkdir()
            (content_dir / "script.py").write_text("print('lock test')\n")

            lock_path = patched.config.with_suffix(".lock")
            observed = {}

            def _check_lock_then_copy(content_files, name):
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                lock_path.touch(exist_ok=True)
                with open(lock_path) as lock_handle:
                    try:
                        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        observed["lock_was_free"] = True
                        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                    except BlockingIOError:
                        observed["lock_was_free"] = False
                return copy_and_hash_files(content_files, name)

            with patch(
                "syft_bg.api.api.copy_and_hash_files",
                side_effect=_check_lock_then_copy,
            ):
                result = auto_approve(contents=[str(content_dir / "script.py")])

            assert result.success is True
            assert observed["lock_was_free"] is True

    def test_rare_name_collision_renames_and_does_not_clobber(self, temp_dir):
        """If another writer claims the candidate name while the (unlocked)
        file I/O is in flight, auto_approve() must detect the collision
        under the lock, rename to a free name, and not clobber the racer."""
        with _patched_paths(temp_dir):
            content_dir = temp_dir / "project"
            content_dir.mkdir()
            (content_dir / "main.py").write_text("print('hi')\n")

            def _race_then_copy(content_files, name):
                _seed_config(temp_dir, {name: _make_obj("racer@test.com")})
                return copy_and_hash_files(content_files, name)

            with patch(
                "syft_bg.api.api.copy_and_hash_files", side_effect=_race_then_copy
            ):
                result = auto_approve(contents=[str(content_dir / "main.py")])

            assert result.success is True
            assert result.name == "main_1"

            objects = list_auto_approvals()
            assert set(objects.keys()) == {"main", "main_1"}
            assert objects["main"].peers == ["racer@test.com"]

            entry = objects["main_1"].file_contents[0]
            assert "main_1" in entry.path
            assert Path(entry.path).read_text() == "print('hi')\n"


class TestHandlerReloadsConfig:
    """The approve service must pick up YAML changes without a restart."""

    def _make_test_job(self, code_dir: Path, submitted_by: str = "alice@test.com"):
        job = MagicMock()
        job.name = "test-job"
        job.status = "pending"
        job.submitted_by = submitted_by
        job.code_dir = code_dir
        job.files = []
        return job

    def _create_matching_autoapprove_obj_from_dir(
        self, code_dir: Path, peer: str
    ) -> AutoApprovalObj:
        entries = [
            FileEntry.from_file(str(f.relative_to(code_dir)), f)
            for f in sorted(code_dir.rglob("*"))
            if f.is_file()
        ]
        return AutoApprovalObj(file_contents=entries, peers=[peer])

    def test_picks_up_added_object(self, temp_dir):
        code_dir = temp_dir / "code"
        code_dir.mkdir()
        (code_dir / "main.py").write_text("print('hello')\n")
        config_path = _seed_config(temp_dir, {})

        handler = JobApprovalHandler(client=MagicMock(), config_path=config_path)
        job = self._make_test_job(code_dir)

        # No object yet — should not match.
        first = handler.evaluate_auto_approval(job)
        assert first.match is False

        # Add a matching object directly to the YAML on disk (no restart).
        SyftBgConfig(
            approve=AutoApproveConfig(
                auto_approvals=AutoApprovalsConfig(
                    objects={
                        "r1": self._create_matching_autoapprove_obj_from_dir(
                            code_dir, "alice@test.com"
                        )
                    }
                )
            )
        ).save(config_path)

        # Same handler instance, next evaluation re-reads the YAML.
        second = handler.evaluate_auto_approval(job)
        assert second.match is True

    def test_picks_up_removed_object(self, temp_dir):
        code_dir = temp_dir / "code"
        code_dir.mkdir()
        (code_dir / "main.py").write_text("print('hello')\n")

        config_path = _seed_config(
            temp_dir,
            {
                "r1": self._create_matching_autoapprove_obj_from_dir(
                    code_dir, "alice@test.com"
                )
            },
        )

        handler = JobApprovalHandler(client=MagicMock(), config_path=config_path)
        job = self._make_test_job(code_dir)

        first = handler.evaluate_auto_approval(job)
        assert first.match is True

        # Wipe the object from the YAML.
        SyftBgConfig(
            approve=AutoApproveConfig(auto_approvals=AutoApprovalsConfig(objects={}))
        ).save(config_path)

        second = handler.evaluate_auto_approval(job)
        assert second.match is False
