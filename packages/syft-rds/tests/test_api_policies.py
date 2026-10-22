"""An api with input and output policies, created by a DO and called by a DS."""

import json
import tempfile
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from syft_bg.api import create_api_from_job
from syft_bg.approve.handlers.job import ApprovedJobHandler
from syft_bg.common.config import get_default_paths
from syft_rds import SyftRDSClient

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


def test_api_policies_allow_block_and_show():
    ds_manager, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
    )
    job = _submit_adder_job(ds_manager, do_manager, {"a": 1, "b": 2})

    with _temp_config_paths() as (paths, tmp):
        result = create_api_from_job(
            job,
            input_policy=_write(tmp / "rate_limit.py", RATE_LIMIT_POLICY),
            output_policy=_write(tmp / "block_large.py", BLOCK_LARGE_SUM_POLICY),
        )
        assert result.success and result.policies == ["input", "output"]

        do_manager.sync()
        ds_manager.sync()
        api = ds_manager.api.adder
        assert "Rate limit reached" in repr(api)
        assert "Sum too large to share" in api._repr_html_()

        # A call is a request: it carries the params, not the api's code. The
        # DO's client turns it into an approved job, which the approve service
        # runs. The job the api was built from is a whole job, which nothing
        # approves. A call over 100 passes the input policy and is blocked on
        # output.
        large = api(100, b=1, block=False)
        assert large.is_request
        code_files = sorted(
            p.name for p in (large.job_submission_path / "code").iterdir()
        )
        assert code_files == ["params.json"]
        assert _run_calls(do_manager, paths.config) == {large.name}
        blocked = _ds_job(ds_manager, large.name)
        assert blocked.status == "failed"
        assert blocked.policy_error == "ValueError: Sum too large to share"
        assert blocked.output_paths == []

        # The rate limit of one call is used up.
        second = api(1, b=2, block=False)
        _run_calls(do_manager, paths.config)
        limited = _ds_job(ds_manager, second.name)
        assert limited.status == "failed"
        assert limited.policy_error == "ValueError: Rate limit reached: 1 call"

    # The DO's view of a call shows the api and its policies.
    do_job = next(j for j in do_manager.jobs if j.name == large.name)
    assert do_job.api_name == "adder"
    html = do_job._repr_html_()
    assert "adder" in html and "Input policy" in html and "Output policy" in html
    original = next(j for j in do_manager.jobs if j.name == job.name)
    assert original.status == "pending"
