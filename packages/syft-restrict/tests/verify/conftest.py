"""Shared fixtures/helpers for the static-checker tests.

The tests are split by intent:
- ``test_whitelist.py``       — green path: constructs the private region IS allowed to use.
- ``test_disallowed.py``      — default-deny catalog: straightforward rejections.
- ``test_bypasses.py``        — multi-step / subtle escape regressions.
- ``test_whitelisted_lib.py`` — library calls by name and operator bundles (manual allow).
- ``test_ranges.py``          — private-range argument handling, not code policy.
"""

import pytest
from syft_restrict import verify

from verify.helpers import (
    make_policy,
    normalize_source,
)


@pytest.fixture
def policy():
    return make_policy()


@pytest.fixture
def verify_all(policy):
    """Verify ``source`` with the whole file marked private, using the standard policy."""

    def _run(source: str | list[str], pol=None, private=None):
        source = normalize_source(source)
        if private is None:
            private = [[1, len(source.splitlines())]]
        return verify(source, private, pol or policy)

    return _run
