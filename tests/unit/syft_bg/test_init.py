"""Tests for syft_bg.api.api.init() and its OAuth token handling."""

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from syft_bg.api.api import init
from syft_bg.api.utils import move_token_to_syftbg_dir
from syft_bg.common.config import get_default_paths
from syft_bg.common.syft_bg_config import SyftBgConfig


class TestMoveTokenToSyftbgDir:
    """move_token_to_syftbg_dir must return the canonical (post-copy) path."""

    def test_copies_and_returns_canonical_path(self, temp_dir):
        source = temp_dir / "external_token.json"
        source.write_text("{}")
        syftbg_dir = temp_dir / "syftbg"

        with patch("syft_bg.api.utils.get_syftbg_dir", return_value=syftbg_dir):
            result = move_token_to_syftbg_dir(source)

        canonical = syftbg_dir / "token.json"
        assert result == canonical
        assert canonical.exists()

    def test_returns_same_path_when_already_canonical(self, temp_dir):
        syftbg_dir = temp_dir / "syftbg"
        syftbg_dir.mkdir()
        canonical = syftbg_dir / "token.json"
        canonical.write_text("{}")

        with patch("syft_bg.api.utils.get_syftbg_dir", return_value=syftbg_dir):
            result = move_token_to_syftbg_dir(canonical)

        assert result == canonical


@contextmanager
def _patched_paths(tmp: Path):
    """Redirect config, syft-bg dir, and gmail token default to a temp directory."""
    original = get_default_paths()
    syftbg_dir = tmp / "syftbg"
    patched = replace(original, config=syftbg_dir / "config.yaml")
    with (
        patch("syft_bg.api.api.get_default_paths", return_value=patched),
        patch("syft_bg.api.utils.get_syftbg_dir", return_value=syftbg_dir),
        patch("syft_bg.common.syft_bg_config.get_default_paths", return_value=patched),
    ):
        yield patched


class TestInitTokenPath:
    """api.init(token_path=...) must set both gmail and drive token paths."""

    def test_sets_both_gmail_and_drive_token_path(self, temp_dir):
        token_source = temp_dir / "external_token.json"
        token_source.write_text("{}")

        with _patched_paths(temp_dir) as patched:
            init(do_email="alice@test.com", token_path=token_source)

            config = SyftBgConfig.from_path(patched.config)

        canonical_token_path = temp_dir / "syftbg" / "token.json"
        assert config.drive_token_path == canonical_token_path
        assert config.gmail_token_path == canonical_token_path
