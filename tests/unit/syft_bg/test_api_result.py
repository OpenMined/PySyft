"""What ApiResult shows the DO: an api's arguments, and a cleared state."""

from syft_bg.api.api import set_policies
from syft_bg.api.results import ApiResult
from syft_rds.apis import ApiArg, ApiDefinition

from tests.unit.syft_bg.test_create_api import _create, _patched_paths, _write_project

ARGS = [
    {"name": "dataset", "type": "str", "default": "adult"},
    {"name": "n", "type": "int", "required": True},
]
STATE_CLEARED = "state: cleared (pass reset_state=False to keep it)"


def test_create_api_lists_args(temp_dir):
    with _patched_paths(temp_dir):
        result = _create(_write_project(temp_dir), allow_any_peer=True, args=ARGS)

    assert result.args == ["dataset: str = 'adult'", "n: int"]
    assert "  args:\n    - dataset: str = 'adult'\n    - n: int\n" in repr(result)
    assert not result.state_cleared and STATE_CLEARED not in repr(result)


def test_set_policies_says_when_state_cleared(temp_dir):
    policy_path = temp_dir / "policy.py"
    policy_path.write_text("pass\n")
    with _patched_paths(temp_dir):
        _create(_write_project(temp_dir), allow_any_peer=True, args=ARGS)
        cleared_result = set_policies("main", input_policy=policy_path)
        kept_result = set_policies("main", output_policy=policy_path, reset_state=False)

    assert cleared_result.state_cleared and STATE_CLEARED in repr(cleared_result)
    assert not kept_result.state_cleared
    assert STATE_CLEARED not in repr(kept_result)
    assert kept_result.args == ["dataset: str = 'adult'", "n: int"]


def test_api_result_str_matches_repr():
    ok_result = ApiResult.from_api(
        "main", ApiDefinition(args=[ApiArg(name="a", default=1)])
    )
    failed_result = ApiResult(success=False, error="No policy at x.py")

    assert str(ok_result) == repr(ok_result)
    assert str(ok_result).startswith("ApiResult: OK [main]")
    assert str(failed_result) == "ApiResult: FAILED\n  error: No policy at x.py"
