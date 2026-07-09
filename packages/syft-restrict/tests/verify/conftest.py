"""Shared fixtures/helpers for the static-checker tests (research approach B).

The tests are split by intent:
- ``test_whitelist.py``     — language constructs the private region IS allowed to use.
- ``test_blacklist.py``     — constructs/calls/attrs that are rejected (default-deny).
- ``test_whitelisted_lib.py`` — things we *manually* allow: library calls by name and operator bundles.
"""

from pathlib import Path

import pytest
from syft_restrict import Policy, verify
from syft_restrict.verifier import VerifyResult

FIXTURES = Path(__file__).parents[1] / "fixtures"
REPO_ROOT = Path(__file__).parents[4]

ALLOW_FUNCTIONS = ["jax.*", "flax.linen.*"]
ALLOW_METHODS = ["arithmetic", "indexing", "comparison"]


def make_policy(functions=ALLOW_FUNCTIONS, methods=ALLOW_METHODS, disallow=None):
    return Policy.parse(list(functions), list(methods), list(disallow or []))


@pytest.fixture
def policy():
    return make_policy()


@pytest.fixture
def verify_all(policy):
    """Verify ``source`` with the whole file marked private, using the standard policy."""

    def _run(source, pol=None, private=None):
        if private is None:
            private = [[1, len(source.splitlines())]]
        return verify(source, private, pol or policy)

    return _run


def get_error_codes(result: VerifyResult):
    """The set of violation codes in a VerifyResult (handy for asserts)."""
    return {v.code for v in result.violations}
