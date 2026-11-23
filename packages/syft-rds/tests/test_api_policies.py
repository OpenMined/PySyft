"""An api with input and output policies, created by a DO and called by a DS."""

import json
import tempfile
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, NamedTuple
from unittest.mock import patch

from syft_bg.api import create_api_from_job
from syft_bg.api.results import ApiResult
from syft_bg.approve.handlers.job import ApprovedJobHandler
from syft_bg.common.config import get_default_paths
from syft_rds import SyftRDSClient
from syft_rds.apis import Api, ApiDefinition, FileEntry
from syft_rds.apis.repr import api_collection_repr_html, api_collection_repr_str
from syft_rds.client import _print_call_result

ADDER_CODE = """import json
with open("params.json") as f:
    params = json.load(f)
with open("outputs/result.json", "w") as f:
    json.dump({"sum": params["a"] + params["b"]}, f)
"""

# One call in total.
RATE_LIMIT_POLICY = """import json, os
from pathlib import Path
path = Path(os.environ["SYFT_POLICY_DEFAULT_STATE_PATH"])
state = json.loads(path.read_text())
if state.get("calls", 0) >= 1:
    raise ValueError("Rate limit reached: 1 call")
state["calls"] = state.get("calls", 0) + 1
path.write_text(json.dumps(state))
"""

BLOCK_LARGE_SUM_POLICY = """import json, os
from pathlib import Path
outputs = Path(os.environ["SYFT_POLICY_OUTPUTS_DIR"])
if json.loads((outputs / "result.json").read_text())["sum"] > 100:
    raise ValueError("Sum too large to share")
"""

# Approves no reader for any file, so the outputs are held back.
HOLD_BACK_POLICY = """import os
from pathlib import Path
Path(os.environ["SYFT_POLICY_READ_PERMISSIONS_PATH"]).write_text("{}")
"""


@contextmanager
def _temp_config_paths():
    """Redirect syft-bg's config and lock files to a temp directory."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        patched = replace(get_default_paths(), config=tmp_path / "config.yaml")
        with (
            patch("syft_bg.common.config.get_default_paths", return_value=patched),
            patch("syft_bg.api.api.get_default_paths", return_value=patched),
            patch("syft_bg.api.utils.get_default_paths", return_value=patched),
            patch(
                "syft_bg.common.syft_bg_config.get_default_paths", return_value=patched
            ),
            patch("syft_bg.approve.api_store.get_syftbg_dir", return_value=tmp_path),
        ):
            yield patched, tmp_path


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def _submit_adder_job(ds_manager, do_manager, params: dict):
    project_dir = Path(tempfile.mkdtemp(prefix="test_api_policies_"))
    _write(project_dir / "main.py", ADDER_CODE)
    _write(project_dir / "params.json", json.dumps(params))
    ds_manager.submit_python_job(
        user=do_manager.email,
        code_path=str(project_dir),
        job_name="adder",
        entrypoint="main.py",
    )
    do_manager.sync()
    return do_manager.jobs[-1]


def _run_calls(do_manager, config_path: Path) -> set[str]:
    """Run the approve service once; return the names of the jobs it ran."""
    do_manager.sync()
    handler = ApprovedJobHandler(
        client=do_manager, config_path=config_path, verbose=False
    )
    ran = {j.name for j in handler.run_approved_jobs()}
    do_manager.sync()
    return ran


def _ds_job(ds_manager, name: str):
    ds_manager.sync()
    return next(j for j in ds_manager.jobs if j.name == name)


class _ApiSetup(NamedTuple):
    ds_manager: SyftRDSClient
    do_manager: SyftRDSClient
    job: Any  # the job the api was built from
    api_result: ApiResult
    config_path: Path  # the syft-bg config the approve service reads


@contextmanager
def _adder_api(input_policy: str | None = None, output_policy: str | None = None):
    """Pair a DS and a DO, and turn an adder job into an api with these policies."""
    ds_manager, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
    )
    job = _submit_adder_job(ds_manager, do_manager, {"a": 1, "b": 2})
    with _temp_config_paths() as (paths, tmp):
        input_path = _write(tmp / "input.py", input_policy) if input_policy else None
        output_path = (
            _write(tmp / "output.py", output_policy) if output_policy else None
        )
        api_result = create_api_from_job(
            job, input_policy=input_path, output_policy=output_path
        )
        assert api_result.success
        do_manager.sync()
        ds_manager.sync()
        yield _ApiSetup(ds_manager, do_manager, job, api_result, paths.config)


def test_api_view_shows_policy_source():
    with _adder_api(RATE_LIMIT_POLICY, BLOCK_LARGE_SUM_POLICY) as setup:
        assert setup.api_result.policies == ["input", "output"]
        api = setup.ds_manager.api.adder
        # The DS sees each policy's source code, which contains these error texts.
        assert "Rate limit reached" in repr(api)
        assert "Sum too large to share" in api._repr_html_()


def test_api_call_runs_as_request():
    with _adder_api(RATE_LIMIT_POLICY, BLOCK_LARGE_SUM_POLICY) as setup:
        # A call is a request: it carries the params, not the api's code.
        call = setup.ds_manager.api.adder(1, b=2, block=False)
        assert call.job is None
        code_files = sorted(p.name for p in (call.request_path / "code").iterdir())
        assert code_files == ["params.json"]
        # The DO's client turns the call into an approved job, which the approve
        # service runs. Nothing approves the job the api was built from.
        assert _run_calls(setup.do_manager, setup.config_path) == {call.name}

    do_jobs = {j.name: j for j in setup.do_manager.jobs}
    assert do_jobs[setup.job.name].status == "pending"
    # The DO's view of a call shows the api and its policies.
    do_job = do_jobs[call.name]
    assert do_job.api_name == "adder"
    html = do_job._repr_html_()
    assert "Input policy" in html and "Output policy" in html


def test_output_policy_refuses_large_sum(capsys):
    with _adder_api(output_policy=BLOCK_LARGE_SUM_POLICY) as setup:
        large_call = setup.ds_manager.api.adder(100, b=1, block=False)
        _run_calls(setup.do_manager, setup.config_path)
        blocked_job = _ds_job(setup.ds_manager, large_call.name)

    assert blocked_job.status == "failed"
    assert blocked_job.policy_error == "ValueError: Sum too large to share"
    # A refused call shares no log; the DS is sent to the policy's reason.
    assert str(blocked_job.stderr) == (
        "stderr.txt is not shared with you: the data owner's policy "
        "refused the call. The reason is in job.policy_error."
    )
    assert blocked_job.output_paths == []
    capsys.readouterr()
    _print_call_result(blocked_job)
    assert capsys.readouterr().out == (
        f"Job '{large_call.name}' finished with status: failed\n"
        "   The data owner's policy refused the call: "
        "ValueError: Sum too large to share\n"
    )


def test_input_policy_refuses_call_over_rate_limit(capsys):
    with _adder_api(input_policy=RATE_LIMIT_POLICY) as setup:
        first_call = setup.ds_manager.api.adder(1, b=2, block=False)
        _run_calls(setup.do_manager, setup.config_path)
        assert _ds_job(setup.ds_manager, first_call.name).status == "done"
        # The first call used up the rate limit of one call.
        second_call = setup.ds_manager.api.adder(1, b=2, block=False)
        _run_calls(setup.do_manager, setup.config_path)
        limited_job = _ds_job(setup.ds_manager, second_call.name)

    assert limited_job.status == "failed"
    assert limited_job.policy_error == "ValueError: Rate limit reached: 1 call"
    capsys.readouterr()
    _print_call_result(limited_job)
    assert (
        "refused the call: ValueError: Rate limit reached: 1 call"
        in capsys.readouterr().out
    )


def test_held_back_outputs_point_to_policy(capsys):
    with _adder_api(output_policy=HOLD_BACK_POLICY) as setup:
        call = setup.ds_manager.api.adder(1, b=2, block=False)
        _run_calls(setup.do_manager, setup.config_path)
        held_job = _ds_job(setup.ds_manager, call.name)

    assert held_job.status == "done" and held_job.outputs_withheld
    assert held_job.output_paths == []
    capsys.readouterr()
    _print_call_result(held_job)
    assert capsys.readouterr().out == (
        f"Job '{call.name}' finished with status: done\n"
        "   The data owner's output policy held the outputs back.\n"
        "   The data owner can still share them with you later.\n"
    )


def _policy_entry(kind: str) -> list[FileEntry]:
    return [FileEntry(relative_path=f"policies/{kind}/run.sh", hash="sha256:0")]


def test_api_table_shows_policies(tmp_path):
    definitions = {
        "plain": ApiDefinition(),
        "checked": ApiDefinition(input_policy=_policy_entry("input")),
        "filtered": ApiDefinition(output_policy=_policy_entry("output")),
        "both": ApiDefinition(
            input_policy=_policy_entry("input"),
            output_policy=_policy_entry("output"),
        ),
    }
    apis = [
        Api(name, "do@test.org", tmp_path, definition, client=None)
        for name, definition in definitions.items()
    ]

    lines = api_collection_repr_str(apis).splitlines()[1:]
    assert [line.split(" | ")[1] for line in lines] == [
        "policies: -",
        "policies: input",
        "policies: output",
        "policies: input, output",
    ]
    html = api_collection_repr_html(apis)
    for policies in ("-", "input", "output", "input, output"):
        assert f'<td class="syftapi-td">{policies}</td>' in html
