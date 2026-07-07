"""Regression tests for the review-flagged unlocked read-modify-write bugs
in ensure_running() and save_gcp_project_id() (both now route through the
locked SyftBgConfig.edit() instead of a bare load()/save() pair)."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from syft_bg.api.api import ensure_running
from syft_bg.api.utils import save_gcp_project_id
from syft_bg.common.config import get_default_paths
from syft_bg.common.syft_bg_config import SyftBgConfig


class TestEnsureRunning:
    def test_no_config_file_does_not_create_one(self, temp_dir):
        config_path = temp_dir / "config.yaml"
        patched = replace(get_default_paths(), config=config_path)

        with patch("syft_bg.api.api.get_default_paths", return_value=patched):
            ensure_running(["approve"])

        assert not config_path.exists()

    def test_persists_settings_through_locked_edit(self, temp_dir):
        config_path = temp_dir / "config.yaml"
        SyftBgConfig().save(config_path)
        patched = replace(get_default_paths(), config=config_path)

        mock_service = MagicMock()
        mock_service.is_running.return_value = True
        mock_manager = MagicMock()
        mock_manager.get_service.return_value = mock_service

        with (
            patch("syft_bg.api.api.get_default_paths", return_value=patched),
            patch("syft_bg.api.api.ServiceManager", return_value=mock_manager),
        ):
            ensure_running({"approve": {"interval": 42}})

        reloaded = SyftBgConfig.from_path(config_path)
        assert reloaded.approve.interval == 42

    def test_unknown_service_does_not_save(self, temp_dir):
        config_path = temp_dir / "config.yaml"
        SyftBgConfig(do_email="a@test.com").save(config_path)
        patched = replace(get_default_paths(), config=config_path)

        mock_manager = MagicMock()
        mock_manager.get_service.return_value = None

        with (
            patch("syft_bg.api.api.get_default_paths", return_value=patched),
            patch("syft_bg.api.api.ServiceManager", return_value=mock_manager),
        ):
            with pytest.raises(ValueError):
                ensure_running({"bogus": {}})

        reloaded = SyftBgConfig.from_path(config_path)
        assert reloaded.do_email == "a@test.com"


class TestSaveGcpProjectId:
    def test_noop_when_config_missing(self, temp_dir):
        config_path = temp_dir / "config.yaml"
        patched = replace(get_default_paths(), config=config_path)

        with (
            patch("syft_bg.api.utils.get_default_paths", return_value=patched),
            patch(
                "syft_bg.api.utils.get_project_id_from_credentials",
                return_value="proj-123",
            ),
        ):
            save_gcp_project_id(Path("/fake/credentials.json"))

        assert not config_path.exists()

    def test_sets_project_id_when_config_exists(self, temp_dir):
        config_path = temp_dir / "config.yaml"
        SyftBgConfig().save(config_path)
        patched = replace(get_default_paths(), config=config_path)

        with (
            patch("syft_bg.api.utils.get_default_paths", return_value=patched),
            patch(
                "syft_bg.common.syft_bg_config.get_default_paths",
                return_value=patched,
            ),
            patch(
                "syft_bg.api.utils.get_project_id_from_credentials",
                return_value="proj-123",
            ),
        ):
            save_gcp_project_id(Path("/fake/credentials.json"))

        reloaded = SyftBgConfig.from_path(config_path)
        assert reloaded.email_approve.gcp_project_id == "proj-123"
