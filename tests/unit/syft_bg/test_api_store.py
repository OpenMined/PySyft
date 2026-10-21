"""Tests for storing auto-approval objects as apis in SyftBox."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from syft_bg.api.utils import read_job_args
from syft_bg.approve.api_store import ApiStore, DuplicateApiError
from syft_bg.approve.policy_resolver import api_policy_resolver
from syft_job.policies import PolicyResolutionError, python_run_script
from syft_perms import SyftPermContext
from syft_permissions import PERMISSION_FILE_NAME, RuleSet
from syft_rds.apis import ApiArg

DO_EMAIL = "do@test.com"


@pytest.fixture
def store(temp_dir):
    with patch("syft_bg.approve.api_store.get_syftbg_dir", return_value=temp_dir):
        yield ApiStore(temp_dir / "syftbox", DO_EMAIL)


@pytest.fixture
def script(temp_dir) -> Path:
    path = temp_dir / "main.py"
    path.write_text("print('hi')\n")
    return path


def _readers(store: ApiStore, name: str) -> list[str]:
    ruleset = RuleSet.load(store.api_dir(name) / PERMISSION_FILE_NAME)
    assert [r.pattern for r in ruleset.rules] == ["**"]
    return ruleset.rules[0].access.read


def test_create_writes_api_folder_with_peer_permissions(store, script, temp_dir):
    name, obj = store.create(
        "analysis", [("run.sh", script)], ["config.yaml"], ["bob@a.com", "al@b.com"]
    )

    api_dir = temp_dir / "syftbox" / DO_EMAIL / "app_data" / "apis" / "analysis"
    assert name == "analysis"
    assert (api_dir / "files" / "run.sh").read_text() == "print('hi')\n"
    stored = yaml.safe_load((api_dir / "api.yaml").read_text())
    assert set(stored["file_contents"][0]) == {"relative_path", "hash"}
    assert obj.file_contents[0].path == str(api_dir / "files" / "run.sh")
    assert obj.file_paths == ["config.yaml"]
    assert _readers(store, "analysis") == ["al@b.com", "bob@a.com"]


def test_no_peers_grants_read_to_everyone(store, script):
    store.create("open", [("run.sh", script)], [], [])
    assert _readers(store, "open") == ["*"]


def test_save_revokes_removed_peers(store, script):
    store.create("analysis", [("run.sh", script)], [], ["a@x.com", "b@x.com"])

    obj = store.get("analysis")
    obj.peers.remove("a@x.com")
    store.save("analysis", obj)

    assert store.get("analysis").peers == ["b@x.com"]
    assert _readers(store, "analysis") == ["b@x.com"]


def test_name_collision_gets_suffix(store, script, temp_dir):
    other = temp_dir / "other.py"
    other.write_text("print('other')\n")
    store.create("analysis", [("run.sh", script)], [], ["a@x.com"])
    name, _ = store.create("analysis", [("run.sh", other)], [], ["a@x.com"])
    assert name == "analysis_1"
    assert store.names() == ["analysis", "analysis_1"]


def test_delete_removes_folder(store, script):
    store.create("analysis", [("run.sh", script)], [], ["a@x.com"])
    assert store.delete("analysis") is True
    assert not store.api_dir("analysis").exists()
    assert store.delete("analysis") is False


def test_create_stores_args(store, script):
    args = [
        ApiArg(name="a", type="int", default=1),
        ApiArg(name="b", type="str", required=True),
    ]
    store.create("adder", [("run.sh", script)], [], ["a@x.com"], args=args)

    assert store.get("adder").args == args
    stored = yaml.safe_load((store.api_dir("adder") / "api.yaml").read_text())
    assert stored["args"][0] == {
        "name": "a",
        "type": "int",
        "default": 1,
        "required": False,
    }


@pytest.mark.parametrize(
    "params,expected",
    [
        (
            '{"a": 1, "b": "x"}',
            [
                ApiArg(name="a", type="int", default=1),
                ApiArg(name="b", type="str", default="x"),
            ],
        ),
        ("[1, 2]", []),
        ("not json", []),
    ],
)
def test_read_job_args(temp_dir, params, expected):
    (temp_dir / "code").mkdir()
    (temp_dir / "code" / "params.json").write_text(params)
    job = MagicMock(job_submission_path=temp_dir)
    assert read_job_args(job, ["config.yaml", "code/params.json"]) == expected


INPUT_POLICY = "raise SystemExit('denied')\n"


@pytest.fixture
def policy(temp_dir) -> Path:
    path = temp_dir / "rate_limit.py"
    path.write_text(INPUT_POLICY)
    return path


def test_python_policy_gets_generated_run_sh(store, script, policy):
    _, obj = store.create(
        "api", [("run.sh", script)], [], ["a@x.com"], input_policy=policy
    )

    folder = store.api_dir("api") / "policies" / "input"
    assert (folder / "policy.py").read_text() == INPUT_POLICY
    assert (folder / "run.sh").read_text() == python_run_script()
    assert [e.relative_path for e in obj.input_policy] == [
        "policies/input/policy.py",
        "policies/input/run.sh",
    ]
    assert obj.output_policy == []


def test_policy_state_readable_by_owner_only(store, script, temp_dir):
    store.create("api", [("run.sh", script)], [], ["a@x.com"])

    ctx = SyftPermContext(datasite=temp_dir / "syftbox" / DO_EMAIL)
    api_rel = "app_data/apis/api"
    assert ctx.open(f"{api_rel}/api.yaml").has_read_access("a@x.com")
    assert not ctx.open(f"{api_rel}/policy_state/state.json").has_read_access("a@x.com")
    assert (store.api_dir("api") / "policy_state" / "state.json").read_text() == "{}"


def test_duplicate_api_refused_for_overlapping_peers(store, script):
    store.create("first", [("run.sh", script)], [], ["a@x.com"])

    with pytest.raises(DuplicateApiError, match="'first'"):
        store.create("second", [("run.sh", script)], [], ["a@x.com", "b@x.com"])
    with pytest.raises(DuplicateApiError):
        store.create("second", [("run.sh", script)], [], [])
    name, _ = store.create("second", [("run.sh", script)], [], ["b@x.com"])
    assert name == "second"


def test_set_policies_replaces_removes_and_keeps_state(store, script, policy):
    store.create("api", [("run.sh", script)], [], ["a@x.com"], input_policy=policy)
    state_file = store.api_dir("api") / "policy_state" / "state.json"
    state_file.write_text('{"calls": 3}')
    sh_policy = policy.with_suffix(".sh")
    sh_policy.write_text("exit 0\n")

    obj = store.set_policies("api", output_policy=sh_policy)
    assert [e.relative_path for e in obj.output_policy] == ["policies/output/run.sh"]
    assert obj.input_policy  # left out, so kept
    assert state_file.read_text() == '{"calls": 3}'

    obj = store.set_policies("api", input_policy=None, reset_state=True)
    assert obj.input_policy == []
    assert not (store.api_dir("api") / "policies" / "input").exists()
    assert state_file.read_text() == "{}"


def _submission(temp_dir: Path, run_sh: str) -> Path:
    root = temp_dir / "submission"
    root.mkdir(exist_ok=True)
    (root / "run.sh").write_text(run_sh)
    return root


def test_resolver_refuses_changed_policy_and_deleted_api(
    store, script, policy, temp_dir
):
    store.create("api", [("run.sh", script)], [], ["a@x.com"], input_policy=policy)
    resolver = api_policy_resolver(store)
    job = _submission(temp_dir, script.read_text())
    policies = resolver("api", job)
    assert policies.input.folder == store.api_dir("api") / "policies" / "input"

    (store.api_dir("api") / "policies" / "input" / "policy.py").write_text("pass\n")
    with pytest.raises(PolicyResolutionError, match="changed"):
        resolver("api", job)
    store.delete("api")
    with pytest.raises(PolicyResolutionError, match="no longer exists"):
        resolver("api", job)


def test_resolver_refuses_job_that_no_longer_matches(store, script, policy, temp_dir):
    """An api deleted and created again under its name, with other files."""
    job = _submission(temp_dir, script.read_text())
    store.create("api", [("run.sh", script)], [], ["a@x.com"], input_policy=policy)
    store.delete("api")
    other = temp_dir / "other.sh"
    other.write_text("echo other\n")
    store.create("api", [("run.sh", other)], [], ["a@x.com"], input_policy=policy)

    with pytest.raises(PolicyResolutionError, match="no longer matches api 'api'"):
        api_policy_resolver(store)("api", job)
