import inspect
from pathlib import Path

from syft_restrict import Policy, VerifyResult

FIXTURES = Path(__file__).parents[1] / "fixtures"
REPO_ROOT = Path(__file__).parents[4]

ALLOW_FUNCTIONS = ["jax.*", "flax.linen.*"]
ALLOW_OPERATORS = ["arithmetic", "indexing", "comparison"]


def normalize_source(source: str | list[str]) -> str:
    """Normalize test source: join lists, strip common indent, ensure trailing newline."""
    if isinstance(source, list):
        source = "\n".join(source)
    # Leading newline lets cleandoc strip the common indent of triple-quoted blocks;
    # re-add a trailing newline afterward (cleandoc removes it).
    source = inspect.cleandoc("\n" + source).strip()
    return source


def make_policy(functions=ALLOW_FUNCTIONS, operators=ALLOW_OPERATORS, disallow=None):
    return Policy.parse(list(functions), list(operators), list(disallow or []))


def get_error_codes(result: VerifyResult):
    """The set of violation codes in a VerifyResult (handy for asserts)."""
    return {v.code for v in result.violations}
