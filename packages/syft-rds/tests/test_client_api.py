"""client.api: apis shared by a DO, rendered and called by a DS."""

import json
import tempfile
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from syft_bg.api import auto_approve_job
from syft_bg.approve.handlers.job import JobApprovalHandler
from syft_bg.common.config import get_default_paths
from syft_rds import SyftRDSClient
from syft_rds.apis import Api, ApiDefinition, FileEntry
from syft_rds.apis.api import callable_layout

ADDER_CODE = """import json
with open("params.json") as f:
    params = json.load(f)
with open("outputs/result.json", "w") as f:
    json.dump({"sum": params["a"] + params["b"]}, f)
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
            yield patched


def _submit_adder_job(ds_manager, do_manager, params: dict):
    project_dir = Path(tempfile.mkdtemp(prefix="test_client_api_"))
    (project_dir / "main.py").write_text(ADDER_CODE)
    (project_dir / "params.json").write_text(json.dumps(params))
    ds_manager.submit_python_job(
        user=do_manager.email,
        code_path=str(project_dir),
        job_name="adder",
        entrypoint="main.py",
    )
    do_manager.sync()
    return do_manager.jobs[-1]


def _read_sum(ds_manager, job_name: str) -> int:
    job = next(j for j in ds_manager.jobs if j.name == job_name)
    return json.loads(job.output_paths[0].read_text())["sum"]


def test_ds_renders_and_calls_api():
    ds_manager, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
    )
    job = _submit_adder_job(ds_manager, do_manager, {"a": 1, "b": 2})

    with _temp_config_paths() as paths:
        result = auto_approve_job(job)
        assert result.success is True
        do_manager.sync()
        ds_manager.sync()

        # The DS sees the api in a table, with its args and call signature.
        table = ds_manager.api._repr_html_()
        assert "adder" in table and do_manager.email in table
        assert ">a<" in table and ">b<" in table
        assert "client.api.adder(a, b)" in table

        # Accessing the api shows the code a call runs.
        api = ds_manager.api.adder
        assert api.args == ["a", "b"]
        assert api.code == ADDER_CODE
        assert ADDER_CODE in repr(api)

        with pytest.raises(TypeError, match="missing"):
            api(3)
        with pytest.raises(TypeError, match="unexpected"):
            api(3, c=4)

        # Calling it submits a job the DO's auto-approval matches and runs.
        called_job = api(3, b=4)
        assert called_job is not None and called_job.name.startswith("adder-")
        do_manager.sync()
        handler = JobApprovalHandler(
            client=do_manager, config_path=paths.config, verbose=False
        )
        # The original job matches too: the api was built from it.
        approved = {j.name for j in handler.check_and_approve()}
        assert approved == {called_job.name, job.name}

    do_manager.sync()
    ds_manager.sync()
    assert _read_sum(ds_manager, called_job.name) == 7


def _definition(pinned: list[str], name_only: list[str], args=()) -> ApiDefinition:
    return ApiDefinition(
        file_contents=[FileEntry(relative_path=p, hash="sha256:x") for p in pinned],
        file_paths=name_only,
        args=list(args),
    )


@pytest.mark.parametrize(
    "pinned,name_only,expected",
    [
        (["run.sh", "code/main.py"], ["config.yaml", "code/params.json"], True),
        (["code/main.py"], ["config.yaml", "code/params.json"], False),
        (["run.sh", "code/a.py", "code/b.py"], ["config.yaml", "code/p.json"], False),
        (["run.sh", "code/main.py"], ["config.yaml"], False),
        (["run.sh", "code/main.py"], ["config.yaml", "code/p.txt"], False),
    ],
)
def test_callable_layout(pinned, name_only, expected):
    layout = callable_layout(_definition(pinned, name_only))
    assert (layout is not None) is expected
    if expected:
        assert layout.entrypoint == "main.py"
        assert layout.params_file == "code/params.json"


def test_bind_args_positional_and_keyword():
    api = Api(
        "adder", "do@x.com", Path("/nonexistent"), _definition([], [], "ab"), None
    )
    assert api.bind_args((1,), {"b": 2}) == {"a": 1, "b": 2}
    with pytest.raises(TypeError, match="takes 2"):
        api.bind_args((1, 2, 3), {})
    with pytest.raises(TypeError, match="multiple values"):
        api.bind_args((1,), {"a": 2})


def test_non_callable_api_repr_and_call():
    definition = _definition(["run.sh"], ["config.yaml"])
    api = Api("script", "do@x.com", Path("/nonexistent"), definition, None)
    assert "can't be called" in repr(api)
    assert "can't be called" in api._repr_html_()
    assert (
        'client.api["x.y"]'
        in Api("x.y", "do@x.com", Path("/x"), definition, None).call_signature
    )
    with pytest.raises(TypeError, match="can't be called"):
        api()
