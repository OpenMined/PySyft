"""
Detect how syft-client was installed to determine the correct dependency string.

This module uses PEP 610 (direct_url.json) to determine if syft-client was installed
from a local directory, a git URL, or from PyPI.

See: https://packaging.python.org/en/latest/specifications/direct-url/
"""

import json
import logging
import os
from functools import lru_cache
from importlib.metadata import distributions

logger = logging.getLogger(__name__)

PACKAGE_NAME = "syft-client"
ENV_VAR_NAME = "SYFT_CLIENT_INSTALL_SOURCE"


def _parse_direct_url(direct_url: dict) -> str:
    """
    Parse a direct_url.json content and return the pip-installable string.

    Args:
        direct_url: Parsed direct_url.json content

    Returns:
        A string suitable for pip/uv install
    """
    url = direct_url.get("url", "")

    # VCS install (git, hg, svn, bzr)
    if "vcs_info" in direct_url:
        vcs_info = direct_url["vcs_info"]
        vcs = vcs_info.get("vcs", "git")

        # Get the revision to pin to
        revision = vcs_info.get("requested_revision") or vcs_info.get("commit_id")

        if revision:
            return f"{vcs}+{url}@{revision}"
        return f"{vcs}+{url}"

    # Local directory install (editable or not)
    if "dir_info" in direct_url:
        # Convert file:// URL to path
        if url.startswith("file://"):
            return url[7:]  # Remove "file://" prefix
        return url

    # Archive URL install
    if "archive_info" in direct_url:
        return url

    # Fallback: return the URL as-is
    return url


def _find_syft_client_info() -> tuple[dict | None, str | None]:
    """
    Find the direct_url.json content and version for syft-client.

    Returns:
        Tuple of (direct_url dict or None, version string or None)
    """
    version = None

    for dist in distributions():
        # try except because some distributions may not have a name and it raises
        try:
            if dist.name != PACKAGE_NAME:
                continue
        except Exception:
            continue

        # Always capture the version
        if version is None:
            version = dist.version

        try:
            content = dist.read_text("direct_url.json")
            if content:
                return json.loads(content), version
        except FileNotFoundError:
            # This distribution doesn't have direct_url.json, try next
            continue
        except json.JSONDecodeError:
            continue

    return None, version


@lru_cache(maxsize=1)
def get_syft_client_install_source() -> str:
    """
    Determine how syft-client was installed and return the appropriate dependency string.

    Priority:
    1. Environment variable override (SYFT_CLIENT_INSTALL_SOURCE)
    2. Auto-detection from package metadata (direct_url.json)
    3. Fallback to PyPI package name with version

    Returns:
        A string suitable for pip/uv install, one of:
        - A local path (e.g., "/Users/test/workspace/syft-client")
        - A git URL (e.g., "git+https://github.com/OpenMined/syft-client@main")
        - The PyPI package name with version (e.g., "syft-client==0.1.94")
    """
    # Check for environment variable override
    env_override = os.environ.get(ENV_VAR_NAME)
    if env_override:
        return env_override

    # Try to detect from package metadata
    direct_url, version = _find_syft_client_info()
    if direct_url:
        return _parse_direct_url(direct_url)

    # Fallback to PyPI package name with version
    if version:
        return f"{PACKAGE_NAME}=={version}"

    logger.warning(
        f"Could not detect syft-client installation source or version. "
        f"Falling back to '{PACKAGE_NAME}'. "
        f"Jobs may fail if syft-client is not available on PyPI."
    )
    return PACKAGE_NAME
