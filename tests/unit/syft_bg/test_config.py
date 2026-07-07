"""Tests for configuration loading."""

import fcntl
from pathlib import Path

import pytest
import yaml

from syft_bg.approve.config import (
    AutoApproveConfig,
    AutoApprovalObj,
    AutoApprovalsConfig,
    PeerApprovalConfig,
    FileEntry,
)
from syft_bg.common.syft_bg_config import SyftBgConfig
from syft_bg.notify.config import NotifyConfig


class TestFileEntry:
    """Tests for FileEntry (renamed from ScriptEntry)."""

    def test_from_dict(self):
        entry = FileEntry.model_validate(
            {"relative_path": "train.py", "path": "/tmp/train.py", "hash": "sha256:aaa"}
        )
        assert entry.relative_path == "train.py"
        assert entry.path == "/tmp/train.py"
        assert entry.hash == "sha256:aaa"

    def test_model_dump(self):
        entry = FileEntry(
            relative_path="main.py", path="/tmp/main.py", hash="sha256:bbb"
        )
        d = entry.model_dump()
        assert d == {
            "relative_path": "main.py",
            "path": "/tmp/main.py",
            "hash": "sha256:bbb",
        }


class TestAutoApprovalObj:
    """Tests for AutoApprovalObj."""

    def test_from_dict(self):
        obj = AutoApprovalObj.model_validate(
            {
                "file_contents": [
                    {
                        "relative_path": "train.py",
                        "path": "/tmp/train.py",
                        "hash": "sha256:aaa",
                    }
                ],
                "file_paths": ["params.json"],
                "peers": ["alice@test.com"],
            }
        )
        assert len(obj.file_contents) == 1
        assert obj.file_contents[0].relative_path == "train.py"
        assert obj.file_paths == ["params.json"]
        assert obj.peers == ["alice@test.com"]

    def test_defaults(self):
        obj = AutoApprovalObj()
        assert obj.file_contents == []
        assert obj.file_paths == []
        assert obj.peers == []

    def test_multiple_scripts(self):
        obj = AutoApprovalObj(
            file_contents=[
                FileEntry(
                    relative_path="main.py", path="/tmp/main.py", hash="sha256:aaa"
                ),
                FileEntry(
                    relative_path="utils.py", path="/tmp/utils.py", hash="sha256:bbb"
                ),
            ],
        )
        assert len(obj.file_contents) == 2
        assert obj.file_contents[0].relative_path == "main.py"
        assert obj.file_contents[1].relative_path == "utils.py"


class TestAutoApproveConfig:
    """Tests for AutoApproveConfig."""

    def test_default_config(self):
        config = AutoApproveConfig()
        assert config.do_email is None
        assert config.syftbox_root is None
        assert config.interval == 5
        assert config.auto_approvals.enabled is True
        assert config.auto_approvals.objects == {}
        assert config.peers.enabled is False

    def test_load_from_file(self, sample_config):
        config = SyftBgConfig.from_path(sample_config).approve
        assert config.do_email == "test@example.com"
        assert config.syftbox_root == Path("/tmp/syftbox")
        assert config.interval == 5
        assert config.auto_approvals.enabled is True
        assert "analysis" in config.auto_approvals.objects
        obj = config.auto_approvals.objects["analysis"]
        assert len(obj.file_contents) == 1
        assert obj.file_contents[0].relative_path == "main.py"
        assert obj.file_contents[0].hash == "sha256:abc123"
        assert "alice@uni.edu" in obj.peers
        assert "bob@co.com" in obj.peers

    def test_load_nonexistent_returns_defaults(self, temp_dir):
        config = SyftBgConfig.load(temp_dir / "nonexistent.yaml").approve
        assert config.do_email is None
        assert config.auto_approvals.enabled is True

    def test_save_config(self, temp_dir):
        config_path = temp_dir / "config.yaml"
        approve_config = AutoApproveConfig(
            do_email="save@example.com",
            syftbox_root=Path("/tmp/saved"),
            interval=10,
        )
        approve_config.auto_approvals.enabled = False
        approve_config.auto_approvals.objects["test_obj"] = AutoApprovalObj(
            file_contents=[
                FileEntry(
                    relative_path="main.py", path="/tmp/main.py", hash="sha256:xyz"
                )
            ],
            peers=["alice@test.com"],
        )
        SyftBgConfig(approve=approve_config).save(config_path)

        loaded = SyftBgConfig.from_path(config_path).approve
        assert loaded.do_email == "save@example.com"
        assert loaded.interval == 10
        assert loaded.auto_approvals.enabled is False
        assert "test_obj" in loaded.auto_approvals.objects
        obj = loaded.auto_approvals.objects["test_obj"]
        assert obj.file_contents[0].hash == "sha256:xyz"
        assert obj.peers == ["alice@test.com"]

    def test_save_reload_multi_script_roundtrip(self, temp_dir):
        config_path = temp_dir / "config.yaml"
        approve_config = AutoApproveConfig(do_email="rt@test.com")
        approve_config.auto_approvals.objects["multi"] = AutoApprovalObj(
            file_contents=[
                FileEntry(
                    relative_path="main.py", path="/tmp/main.py", hash="sha256:aaa"
                ),
                FileEntry(
                    relative_path="utils.py", path="/tmp/utils.py", hash="sha256:bbb"
                ),
            ],
            peers=["ds@test.com"],
        )
        SyftBgConfig(approve=approve_config).save(config_path)

        loaded = SyftBgConfig.from_path(config_path).approve
        obj = loaded.auto_approvals.objects["multi"]
        assert len(obj.file_contents) == 2
        assert obj.file_contents[0].relative_path == "main.py"
        assert obj.file_contents[1].relative_path == "utils.py"

    def test_load_empty_objects(self, temp_dir):
        config_path = temp_dir / "config.yaml"
        config_path.write_text("""
do_email: test@example.com
approve:
  auto_approvals:
    enabled: true
    objects: {}
""")
        config = SyftBgConfig.from_path(config_path).approve
        assert config.auto_approvals.objects == {}

    def test_gmail_token_path_propagates_from_top_level(self, temp_dir):
        """Regression test: AutoApproveConfig.load() used to never read
        gmail_token_path from disk at all, always falling back to the
        default-factory value. Going through SyftBgConfig fixes this."""
        config_path = temp_dir / "config.yaml"
        custom_token_path = Path("/tmp/custom_gmail_token.json")
        SyftBgConfig(gmail_token_path=custom_token_path).save(config_path)

        loaded = SyftBgConfig.from_path(config_path)
        assert loaded.approve.gmail_token_path == custom_token_path


class TestSyftBgConfigEdit:
    """Tests for SyftBgConfig.edit(), the locked read-modify-write helper."""

    def test_edit_saves_on_clean_exit(self, temp_dir):
        config_path = temp_dir / "config.yaml"
        with SyftBgConfig.edit(config_path) as config:
            config.approve.auto_approvals.enabled = False

        reloaded = SyftBgConfig.from_path(config_path)
        assert reloaded.approve.auto_approvals.enabled is False

    def test_edit_does_not_save_on_exception(self, temp_dir):
        config_path = temp_dir / "config.yaml"

        with pytest.raises(ValueError):
            with SyftBgConfig.edit(config_path) as config:
                config.approve.auto_approvals.enabled = False
                raise ValueError("boom")

        assert not config_path.exists()

    def test_edit_holds_exclusive_lock(self, temp_dir):
        config_path = temp_dir / "config.yaml"

        with SyftBgConfig.edit(config_path) as config:
            config.approve.auto_approvals.enabled = False

            lock_path = config_path.with_suffix(".lock")
            with open(lock_path) as lock_handle:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


class TestSyftBgConfigMergeAndSave:
    """Regression tests for the review-flagged merge/save correctness bugs."""

    def test_save_preserves_unknown_top_level_keys(self, temp_dir):
        """save() must not silently drop YAML keys it doesn't model."""
        config_path = temp_dir / "config.yaml"
        config_path.write_text("""
do_email: test@example.com
some_future_field: keep_me
approve:
  interval: 5
""")
        with SyftBgConfig.edit(config_path) as config:
            config.approve.interval = 10

        raw = yaml.safe_load(config_path.read_text())
        assert raw["some_future_field"] == "keep_me"
        assert raw["approve"]["interval"] == 10

    def test_merge_does_not_clobber_custom_service_token_path(self, temp_dir):
        """A per-service drive_token_path override must survive a reload."""
        config_path = temp_dir / "config.yaml"
        custom_path = Path("/custom/approve_token.json")

        syft_bg_config = SyftBgConfig()
        syft_bg_config.set_service_config(
            "approve", {"drive_token_path": custom_path}
        )
        syft_bg_config.save(config_path)

        reloaded = SyftBgConfig.from_path(config_path)
        assert reloaded.approve.drive_token_path == custom_path

    def test_edit_does_not_bake_common_fields_into_approve(self, temp_dir):
        """edit() must not permanently write do_email into approve's own
        fields, so a later top-level change is still inherited on reload."""
        config_path = temp_dir / "config.yaml"
        SyftBgConfig(do_email="alice@test.com").save(config_path)

        with SyftBgConfig.edit(config_path) as config:
            config.approve.auto_approvals.enabled = False

        raw = yaml.safe_load(config_path.read_text())
        assert raw["approve"]["do_email"] is None

        # Top-level do_email changes later; approve should still inherit it.
        raw["do_email"] = "bob@test.com"
        config_path.write_text(
            yaml.safe_dump(raw, sort_keys=False, default_flow_style=False)
        )

        reloaded = SyftBgConfig.from_path(config_path)
        assert reloaded.approve.do_email == "bob@test.com"


class TestAutoApprovalsConfig:
    """Tests for AutoApprovalsConfig."""

    def test_from_dict_with_objects(self):
        config = AutoApprovalsConfig.model_validate(
            {
                "enabled": True,
                "objects": {
                    "my_analysis": {
                        "file_contents": [
                            {
                                "relative_path": "main.py",
                                "path": "/tmp/main.py",
                                "hash": "sha256:abc",
                            }
                        ],
                        "file_paths": [],
                        "peers": ["alice@test.com"],
                    },
                },
            }
        )
        assert config.enabled is True
        assert "my_analysis" in config.objects
        assert config.objects["my_analysis"].file_contents[0].relative_path == "main.py"

    def test_defaults(self):
        config = AutoApprovalsConfig()
        assert config.enabled is True
        assert config.objects == {}


class TestPeerApprovalConfig:
    """Tests for PeerApprovalConfig."""

    def test_from_dict(self):
        config = PeerApprovalConfig.model_validate(
            {
                "enabled": True,
                "approved_domains": ["example.com", "test.org"],
                "auto_share_datasets": ["dataset1"],
            }
        )
        assert config.enabled is True
        assert config.approved_domains == ["example.com", "test.org"]
        assert config.auto_share_datasets == ["dataset1"]

    def test_defaults(self):
        config = PeerApprovalConfig()
        assert config.enabled is False
        assert config.approved_domains == []
        assert config.auto_share_datasets == []


class TestNotifyConfig:
    """Tests for NotifyConfig."""

    def test_default_config(self):
        config = NotifyConfig()
        assert config.do_email is None
        assert config.syftbox_root is None
