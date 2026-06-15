"""
Tests for detecting syft-client installation source.

These tests verify that we can correctly determine how syft-client was installed:
- Editable install from local directory
- Non-editable install from local directory
- Install from PyPI
- Install from GitHub URL
- Override via environment variable
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def clear_install_source_cache():
    """Clear the lru_cache before each test."""
    from syft_job.install_source import get_syft_client_install_source

    get_syft_client_install_source.cache_clear()
    yield
    get_syft_client_install_source.cache_clear()


class MockDistribution:
    """Mock distribution for testing."""

    def __init__(
        self,
        name: str,
        path: str,
        direct_url: dict | None = None,
        version: str = "0.1.94",
    ):
        self.name = name
        self._path = path
        self._direct_url = direct_url
        self.version = version

    def read_text(self, filename: str) -> str:
        if filename == "direct_url.json" and self._direct_url:
            return json.dumps(self._direct_url)
        raise FileNotFoundError(f"{filename} not found")


def create_mock_distributions(*dists):
    """Create a mock distributions() function that returns the given distributions."""

    def mock_distributions():
        return iter(dists)

    return mock_distributions


class TestGetInstallSource:
    """Tests for get_syft_client_install_source function."""

    def test_env_var_override_takes_precedence(self):
        """Environment variable should override all other detection methods."""
        from syft_job.install_source import get_syft_client_install_source

        with patch.dict(os.environ, {"SYFT_CLIENT_INSTALL_SOURCE": "/custom/path"}):
            result = get_syft_client_install_source()
            assert result == "/custom/path"

    def test_editable_install_returns_local_path(self):
        """Editable install should return the local directory path."""
        from syft_job.install_source import get_syft_client_install_source

        mock_dist = MockDistribution(
            name="syft-client",
            path="/path/to/site-packages/syft_client-0.1.94.dist-info",
            direct_url={
                "url": "file:///Users/test/workspace/syft-client",
                "dir_info": {"editable": True},
            },
        )

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SYFT_CLIENT_INSTALL_SOURCE", None)
            with patch(
                "syft_job.install_source.distributions",
                create_mock_distributions(mock_dist),
            ):
                result = get_syft_client_install_source()
                assert result == "/Users/test/workspace/syft-client"

    def test_local_non_editable_install_returns_path(self):
        """Non-editable local install should return the local directory path."""
        from syft_job.install_source import get_syft_client_install_source

        mock_dist = MockDistribution(
            name="syft-client",
            path="/path/to/site-packages/syft_client-0.1.94.dist-info",
            direct_url={
                "url": "file:///Users/test/workspace/syft-client",
                "dir_info": {},  # No editable flag
            },
        )

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SYFT_CLIENT_INSTALL_SOURCE", None)
            with patch(
                "syft_job.install_source.distributions",
                create_mock_distributions(mock_dist),
            ):
                result = get_syft_client_install_source()
                assert result == "/Users/test/workspace/syft-client"

    def test_github_install_returns_git_url(self):
        """GitHub install should return the git+ URL with branch."""
        from syft_job.install_source import get_syft_client_install_source

        mock_dist = MockDistribution(
            name="syft-client",
            path="/path/to/site-packages/syft_client-0.1.94.dist-info",
            direct_url={
                "url": "https://github.com/OpenMined/syft-client",
                "vcs_info": {
                    "vcs": "git",
                    "commit_id": "abc123def456",
                    "requested_revision": "main",
                },
            },
        )

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SYFT_CLIENT_INSTALL_SOURCE", None)
            with patch(
                "syft_job.install_source.distributions",
                create_mock_distributions(mock_dist),
            ):
                result = get_syft_client_install_source()
                assert result == "git+https://github.com/OpenMined/syft-client@main"

    def test_github_install_with_branch(self):
        """GitHub install with specific branch should include branch in URL."""
        from syft_job.install_source import get_syft_client_install_source

        mock_dist = MockDistribution(
            name="syft-client",
            path="/path/to/site-packages/syft_client-0.1.94.dist-info",
            direct_url={
                "url": "https://github.com/OpenMined/syft-client.git",
                "vcs_info": {
                    "vcs": "git",
                    "commit_id": "abc123def456",
                    "requested_revision": "feature/new-stuff",
                },
            },
        )

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SYFT_CLIENT_INSTALL_SOURCE", None)
            with patch(
                "syft_job.install_source.distributions",
                create_mock_distributions(mock_dist),
            ):
                result = get_syft_client_install_source()
                assert (
                    result
                    == "git+https://github.com/OpenMined/syft-client.git@feature/new-stuff"
                )

    def test_github_install_without_branch_uses_commit(self):
        """GitHub install without requested_revision should use commit_id."""
        from syft_job.install_source import get_syft_client_install_source

        mock_dist = MockDistribution(
            name="syft-client",
            path="/path/to/site-packages/syft_client-0.1.94.dist-info",
            direct_url={
                "url": "https://github.com/OpenMined/syft-client",
                "vcs_info": {
                    "vcs": "git",
                    "commit_id": "abc123def456",
                },
            },
        )

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SYFT_CLIENT_INSTALL_SOURCE", None)
            with patch(
                "syft_job.install_source.distributions",
                create_mock_distributions(mock_dist),
            ):
                result = get_syft_client_install_source()
                assert (
                    result
                    == "git+https://github.com/OpenMined/syft-client@abc123def456"
                )

    def test_pypi_install_returns_package_name_with_version(self):
        """PyPI install (no direct_url.json) should return package name with version."""
        from syft_job.install_source import get_syft_client_install_source

        mock_dist = MockDistribution(
            name="syft-client",
            path="/path/to/site-packages/syft_client-0.1.94.dist-info",
            direct_url=None,  # No direct_url.json for PyPI installs
            version="0.1.94",
        )

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SYFT_CLIENT_INSTALL_SOURCE", None)
            with patch(
                "syft_job.install_source.distributions",
                create_mock_distributions(mock_dist),
            ):
                result = get_syft_client_install_source()
                assert result == "syft-client==0.1.94"

    def test_no_package_found_returns_pypi_name_without_version_and_warns(self, caplog):
        """When syft-client is not installed, fall back to PyPI name and log a warning."""
        import logging

        from syft_job.install_source import get_syft_client_install_source

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SYFT_CLIENT_INSTALL_SOURCE", None)
            with patch(
                "syft_job.install_source.distributions",
                create_mock_distributions(),  # No distributions
            ):
                with caplog.at_level(logging.WARNING):
                    result = get_syft_client_install_source()

                assert result == "syft-client"
                assert "Could not detect syft-client installation source" in caplog.text
                assert "Falling back to" in caplog.text

    def test_multiple_distributions_finds_dist_info(self):
        """When both egg-info and dist-info exist, use dist-info with direct_url."""
        from syft_job.install_source import get_syft_client_install_source

        # Egg-info without direct_url.json (older format)
        mock_egg_info = MockDistribution(
            name="syft-client",
            path="syft_client.egg-info",
            direct_url=None,
        )

        # Dist-info with direct_url.json (newer format)
        mock_dist_info = MockDistribution(
            name="syft-client",
            path="/path/to/site-packages/syft_client-0.1.94.dist-info",
            direct_url={
                "url": "file:///Users/test/workspace/syft-client",
                "dir_info": {"editable": True},
            },
        )

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SYFT_CLIENT_INSTALL_SOURCE", None)
            with patch(
                "syft_job.install_source.distributions",
                create_mock_distributions(mock_egg_info, mock_dist_info),
            ):
                result = get_syft_client_install_source()
                assert result == "/Users/test/workspace/syft-client"


class TestGetInstallSourceCurrentEnvironment:
    """Test that we can detect the current environment's install source."""

    def test_current_environment_detection(self):
        """
        Test that the function works in the current environment.

        This test verifies that our detection logic actually works with
        real installed packages, not just mocks.
        """
        from syft_job.install_source import get_syft_client_install_source

        # Clear the env var to test auto-detection
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SYFT_CLIENT_INSTALL_SOURCE", None)
            result = get_syft_client_install_source()

            # The result should be a non-empty string
            assert isinstance(result, str)
            assert len(result) > 0

            # In this test environment (editable install), it should be a path
            # If this is an editable install, it should be the repo root
            if Path(result).exists():
                # Check it looks like the syft-client repo
                assert Path(result).is_dir()
                # Should contain syft_client package
                assert (Path(result) / "syft_client").exists() or (
                    Path(result) / "pyproject.toml"
                ).exists()


class TestRuntimeEvaluation:
    """Test that install source is evaluated at runtime, not import time."""

    def test_install_source_respects_env_var_change_after_import(self):
        """
        Verify that changing the env var after import affects the result.

        This confirms that the install source is evaluated at call time,
        not at module import time. This is important so that tests and
        runtime code can override the source without needing to reload modules.

        Note: We use cache_clear() between calls because the function uses
        lru_cache for performance. In production, the cache means the env var
        must be set BEFORE the first call to get_syft_client_install_source().
        """
        from syft_job.install_source import get_syft_client_install_source

        # First, check with no env var
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SYFT_CLIENT_INSTALL_SOURCE", None)
            get_syft_client_install_source.cache_clear()
            result_without_env = get_syft_client_install_source()

        # Now set an env var and verify it's respected
        with patch.dict(os.environ, {"SYFT_CLIENT_INSTALL_SOURCE": "/new/test/path"}):
            get_syft_client_install_source.cache_clear()
            result_with_env = get_syft_client_install_source()
            assert result_with_env == "/new/test/path"

        # The results should be different (env var overrides auto-detection)
        assert result_without_env != result_with_env
