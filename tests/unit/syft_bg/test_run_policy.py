"""sbg.run_policy: running a policy on its own, without a job."""

from pathlib import Path
from unittest.mock import patch

import pytest
from syft_bg import run_policy
from syft_bg.approve.api_store import ApiStore

DO_EMAIL = "do@test.com"
DS_EMAIL = "ds@test.com"

RATE_LIMIT = """import json, os
from pathlib import Path
path = Path(os.environ["SYFT_POLICY_DEFAULT_STATE_PATH"])
state = json.loads(path.read_text())
if state.get("calls", 0) >= 2:
    raise ValueError("Rate limit reached")
state["calls"] = state.get("calls", 0) + 1
path.write_text(json.dumps(state))
"""

REDACT = """import os
from pathlib import Path
outputs = Path(os.environ["SYFT_POLICY_OUTPUTS_DIR"])
candidates = outputs.parent / "output_candidates"
text = (candidates / "result.txt").read_text()
(outputs / "result.txt").write_text(text.replace("SECRET", "[REDACTED]"))
"""

ECHO_PARAMS = """import json, os, sys
from pathlib import Path
params = json.loads(Path(os.environ["SYFT_POLICY_PARAMS_PATH"]).read_text())
print(params["word"], os.environ["SYFT_POLICY_SUBMITTER"], file=sys.stderr)
sys.exit(1)
"""


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_rate_limit_against_one_state_dir(temp_dir):
    policy = _write(temp_dir / "rate_limit.py", RATE_LIMIT)
    state_dir = temp_dir / "state"

    results = [run_policy(policy, state_dir=state_dir) for _ in range(3)]

    assert [r.passed for r in results] == [True, True, False]
    assert results[1].state == {"calls": 2}
    assert results[2].error == "ValueError: Rate limit reached"


def test_policy_sees_params_and_submitter(temp_dir):
    policy = _write(temp_dir / "echo.py", ECHO_PARAMS)

    result = run_policy(policy, params={"word": "privacy"}, submitter=DS_EMAIL)

    assert result.passed is False
    assert result.error == f"privacy {DS_EMAIL}"


def test_output_policy_builds_outputs_from_candidates(temp_dir):
    policy = _write(temp_dir / "redact.py", REDACT)
    candidates = temp_dir / "candidates"
    _write(candidates / "result.txt", "the SECRET word")

    result = run_policy(policy, candidates_dir=candidates, submitter=DS_EMAIL)

    assert result.passed is True
    assert result.recipients == [DS_EMAIL]
    assert (result.outputs_dir / "result.txt").read_text() == "the [REDACTED] word"


def test_api_policy_runs_against_copy_of_state(temp_dir):
    store = ApiStore(temp_dir / "syftbox", DO_EMAIL)
    run_sh = _write(temp_dir / "run.sh", "echo hi\n")
    policy = _write(temp_dir / "rate_limit.py", RATE_LIMIT)
    with patch("syft_bg.approve.api_store.get_syftbg_dir", return_value=temp_dir):
        store.create("api", [("run.sh", run_sh)], [], [DS_EMAIL], input_policy=policy)
    real_state = store.api_dir("api") / "policy_state" / "state.json"
    real_state.write_text('{"calls": 1}')

    result = run_policy(api="api", kind="input", api_store=store)

    assert result.passed is True and result.state == {"calls": 2}
    assert real_state.read_text() == '{"calls": 1}'


@pytest.mark.parametrize("kind", ["input", "output"])
def test_api_without_that_policy_fails(temp_dir, kind):
    store = ApiStore(temp_dir / "syftbox", DO_EMAIL)
    run_sh = _write(temp_dir / "run.sh", "echo hi\n")
    with patch("syft_bg.approve.api_store.get_syftbg_dir", return_value=temp_dir):
        store.create("api", [("run.sh", run_sh)], [], [DS_EMAIL])

    result = run_policy(api="api", kind=kind, api_store=store)

    assert result.passed is False
    assert result.error == f"api 'api' has no {kind} policy"
