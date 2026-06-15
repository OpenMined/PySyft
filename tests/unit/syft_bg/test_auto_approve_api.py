"""Tests for the list/remove auto-approve Python API and config-reload behavior."""

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch


from syft_bg.api.api import list_auto_approvals, remove_auto_approve
from syft_bg.approve.config import (
    AutoApprovalObj,
    AutoApprovalsConfig,
    AutoApproveConfig,
    FileEntry,
)
from syft_bg.approve.handlers.job import JobApprovalHandler
from syft_bg.common.config import get_default_paths


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
        patch("syft_bg.approve.config.get_default_paths", return_value=patched),
    ):
        yield patched


def _seed_config(tmp: Path, objects: dict[str, AutoApprovalObj]) -> Path:
    """Write a config YAML with the given auto-approval objects to tmp/config.yaml."""
    config_path = tmp / "config.yaml"
    AutoApproveConfig(auto_approvals=AutoApprovalsConfig(objects=objects)).save(
        config_path
    )
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
        AutoApproveConfig(
            auto_approvals=AutoApprovalsConfig(
                objects={
                    "r1": self._create_matching_autoapprove_obj_from_dir(
                        code_dir, "alice@test.com"
                    )
                }
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
        AutoApproveConfig(auto_approvals=AutoApprovalsConfig(objects={})).save(
            config_path
        )

        second = handler.evaluate_auto_approval(job)
        assert second.match is False
