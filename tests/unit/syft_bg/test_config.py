"""Tests for configuration loading."""

from pathlib import Path

from syft_bg.approve.config import (
    AutoApproveConfig,
    AutoApprovalObj,
    AutoApprovalsConfig,
    PeerApprovalConfig,
    FileEntry,
)
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
        config = AutoApproveConfig.load(sample_config)
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
        config = AutoApproveConfig.load(temp_dir / "nonexistent.yaml")
        assert config.do_email is None
        assert config.auto_approvals.enabled is True

    def test_save_config(self, temp_dir):
        config_path = temp_dir / "config.yaml"
        config = AutoApproveConfig(
            do_email="save@example.com",
            syftbox_root=Path("/tmp/saved"),
            interval=10,
        )
        config.auto_approvals.enabled = False
        config.auto_approvals.objects["test_obj"] = AutoApprovalObj(
            file_contents=[
                FileEntry(
                    relative_path="main.py", path="/tmp/main.py", hash="sha256:xyz"
                )
            ],
            peers=["alice@test.com"],
        )
        config.save(config_path)

        loaded = AutoApproveConfig.load(config_path)
        assert loaded.do_email == "save@example.com"
        assert loaded.interval == 10
        assert loaded.auto_approvals.enabled is False
        assert "test_obj" in loaded.auto_approvals.objects
        obj = loaded.auto_approvals.objects["test_obj"]
        assert obj.file_contents[0].hash == "sha256:xyz"
        assert obj.peers == ["alice@test.com"]

    def test_save_reload_multi_script_roundtrip(self, temp_dir):
        config_path = temp_dir / "config.yaml"
        config = AutoApproveConfig(do_email="rt@test.com")
        config.auto_approvals.objects["multi"] = AutoApprovalObj(
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
        config.save(config_path)

        loaded = AutoApproveConfig.load(config_path)
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
        config = AutoApproveConfig.load(config_path)
        assert config.auto_approvals.objects == {}


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
