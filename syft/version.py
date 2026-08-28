"""
Version constants for syft.
Single source of truth for all version strings.
Bump these versions on each release (via `bump2version patch/minor/major`).
"""

# Current client version — the single source of truth.
# pyproject.toml and syft/__init__.py read from here.
# Must stay a plain X.Y.Z: peers compare it with _parse_semver and Drive folder
# names embed it (see sync/version/version_info.py and gdrive_transport.py).
SYFT_VERSION = "0.10.0"

# Minimum client version we support communicating with.
# 0.10.0 is the first release published as `syft` (formerly `syft-client`
# 0.1.x); compatibility is same major.minor, so nothing older can talk to us.
MIN_SUPPORTED_SYFT_VERSION = "0.10.0"

# Protocol version - bump when making breaking changes to the sync protocol
PROTOCOL_VERSION = "1.0.0"

# Minimum protocol version we support
MIN_SUPPORTED_PROTOCOL_VERSION = "1.0.0"

# Name of the version file stored in SyftBox folder
VERSION_FILE_NAME = "SYFT_version.json"

if __name__ == "__main__":
    print(SYFT_VERSION)
