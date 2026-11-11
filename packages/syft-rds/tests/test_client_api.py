"""client.api: apis shared by a DO, rendered and called by a DS."""

import json
import tempfile
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from syft_bg.api import create_api_from_job
from syft_bg.approve.handlers.job import ApprovedJobHandler
from syft_bg.common.config import get_default_paths
from syft_rds import SyftRDSClient
from syft_rds.apis import Api, ApiArg, ApiDefinition, FileEntry, infer_args
from syft_rds.apis.models import definition_error

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
        result = create_api_from_job(job)
        assert result.success is True
        do_manager.sync()
        ds_manager.sync()

        # The DS sees the api in a table, with its args and call signature.
        table = ds_manager.api._repr_html_()
        assert "adder" in table and do_manager.email in table
        assert "a: int = 1" in table and "b: int = 2" in table
        assert "client.api.adder(a, b)" in table

        # Accessing the api shows the code a call runs.
        api = ds_manager.api.adder
        # Names, types and defaults are inferred from the example params.json.
        assert [a.describe() for a in api.args] == ["a: int = 1", "b: int = 2"]
        assert api.code == ADDER_CODE
        assert ADDER_CODE in repr(api) and "a: int = 1" in repr(api)

        # Calls are checked client side, before anything is submitted.
        with pytest.raises(TypeError, match="'a' must be int, got str"):
            api("3")
        with pytest.raises(TypeError, match="unexpected"):
            api(3, c=4)

        # Calling it sends a request, which the DO's client turns into an
        # approved job; the approve service runs it.
        request = api(3, b=4, block=False)
        assert request.name.startswith("adder-") and request.status == "received"
        do_manager.sync()
        handler = ApprovedJobHandler(
            client=do_manager, config_path=paths.config, verbose=False
        )
        ran = {j.name for j in handler.run_approved_jobs()}
        # The job the api was built from is a whole job: nothing approves it.
        assert ran == {request.name}

    do_manager.sync()
    finished = ds_manager._wait_for_call(request.name, poll_interval=0)
    assert finished.status == "done"
    assert _read_sum(ds_manager, request.name) == 7


def _definition(pinned: list[str], name_only: list[str], args=()) -> ApiDefinition:
    return ApiDefinition(
        file_contents=[FileEntry(relative_path=p, hash="sha256:x") for p in pinned],
        file_paths=name_only,
        args=list(args),
    )


A_REQUIRED_INT = ApiArg(name="a", type="int", required=True)
B_FLOAT_DEFAULT = ApiArg(name="b", type="float", default=0.5)


PARAMS = "code/params.json"


@pytest.mark.parametrize(
    "pinned, name_only, args, error",
    [
        (["run.sh", "code/main.py"], [PARAMS], [A_REQUIRED_INT], None),
        (["run.sh", "code/a.py"], ["data.csv", "code/x.json"], [], None),
        (["code/main.py"], [PARAMS], [], "pins run.sh"),
        (["run.sh"], ["config.yaml", PARAMS], [], "written for each call"),
        (["run.sh", PARAMS], [PARAMS], [], "pinned and taken"),
        (["run.sh"], [], [A_REQUIRED_INT], "params.json"),
        (["run.sh"], [PARAMS, "data.csv"], [A_REQUIRED_INT], "params.json"),
    ],
)
def test_definition_error(pinned, name_only, args, error):
    reason = definition_error(_definition(pinned, name_only, args))
    assert (reason is None) if error is None else (error in reason)


@pytest.mark.parametrize(
    "name_only, params_file",
    [([PARAMS], PARAMS), (["params.json"], "params.json")]
    + [([PARAMS, "data.csv"], None), (["code/args.json"], None), ([], None)],
)
def test_params_file_is_only_name_only_params_json(name_only, params_file):
    assert _definition(["run.sh"], name_only).params_file == params_file


def test_call_sends_exactly_listed_files():
    definition = _definition(["run.sh"], ["code/data.csv", "code/labels.csv"])
    api = Api("labels", "do@x.com", Path("/x"), definition, None)
    sent = {"code/data.csv": b"x,y", "code/labels.csv": b"a"}

    assert api.request_files({}, sent) == sent
    assert "files={...}" in api.call_signature
    with pytest.raises(TypeError, match="code/labels.csv"):
        api.request_files({}, {"code/data.csv": b""})
    with pytest.raises(TypeError, match="code/extra.py"):
        api.request_files({}, {**sent, "code/extra.py": b""})


def test_bind_args_positional_keyword_and_defaults():
    spec = [A_REQUIRED_INT, B_FLOAT_DEFAULT]
    api = Api("adder", "do@x.com", Path("/x"), _definition([], [], spec), None)
    assert api.bind_args((1,), {"b": 2}) == {"a": 1, "b": 2}
    assert api.bind_args((), {"a": 1}) == {"a": 1, "b": 0.5}
    with pytest.raises(TypeError, match="missing"):
        api.bind_args((), {})
    with pytest.raises(TypeError, match="takes 2"):
        api.bind_args((1, 2, 3), {})
    with pytest.raises(TypeError, match="multiple values"):
        api.bind_args((1,), {"a": 2})
    with pytest.raises(TypeError, match="'a' must be int, got bool"):
        api.bind_args((True,), {})


@pytest.mark.parametrize(
    "value,expected",
    [(True, "bool"), (1, "int"), (1.5, "float"), ("x", "str"), ([1], "list")]
    + [({"k": 1}, "dict"), (None, "any")],
)
def test_infer_type(value, expected):
    assert infer_args({"x": value}) == [ApiArg(name="x", type=expected, default=value)]


@pytest.mark.parametrize(
    "arg_type,value,ok",
    [("float", 1, True), ("int", 1.0, False), ("list", (1, 2), True)]
    + [("any", object(), True), ("dict", [], False), ("str", None, False)],
)
def test_arg_type_check(arg_type, value, ok):
    assert ApiArg(name="x", type=arg_type).matches(value) is ok


def test_describe():
    assert A_REQUIRED_INT.describe() == "a: int"
    assert B_FLOAT_DEFAULT.describe() == "b: float = 0.5"


@pytest.mark.parametrize(
    "kwargs,expected_block", [({}, True), ({"block": False}, False)]
)
def test_call_blocks_by_default(kwargs, expected_block):
    definition = _definition(["run.sh", "code/main.py"], [PARAMS], [A_REQUIRED_INT])
    client = MagicMock()
    api = Api("adder", "do@x.com", Path("/x"), definition, client)

    api(1, **kwargs)

    client._submit_api_call.assert_called_once_with(
        api, {PARAMS: b'{"a": 1}'}, block=expected_block
    )


def test_api_without_params_file_takes_no_arguments():
    definition = _definition(["run.sh"], [])
    api = Api("script", "do@x.com", Path("/nonexistent"), definition, None)
    assert "client.api.script()" in repr(api)
    assert "client.api.script()" in api._repr_html_()
    assert (
        'client.api["x.y"]'
        in Api("x.y", "do@x.com", Path("/x"), definition, None).call_signature
    )
    assert api.request_files({}, {}) == {}
    with pytest.raises(TypeError, match="takes 0"):
        api(1)
