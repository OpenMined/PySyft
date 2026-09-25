"""A single api shared with the current user, as seen from the client."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from syft_rds.apis.args import ApiArg, bind_args
from syft_rds.apis.models import FILES_DIR_NAME, ApiDefinition
from syft_rds.apis.repr import api_repr_html, api_repr_str

if TYPE_CHECKING:
    from syft_rds.client import SyftRDSClient

RUN_SCRIPT = "run.sh"
CONFIG_FILE = "config.yaml"
CODE_PREFIX = "code/"


@dataclass(frozen=True)
class CallableLayout:
    """The files of an api that can be called with arguments."""

    python_file: str  # e.g. "code/main.py"
    params_file: str  # e.g. "code/params.json"

    @property
    def entrypoint(self) -> str:
        return self.python_file.removeprefix(CODE_PREFIX)


def callable_layout(definition: ApiDefinition) -> CallableLayout | None:
    """The layout of an api with arguments, or None if it isn't one.

    Such an api pins run.sh and one Python file by content, and matches
    config.yaml and one JSON file (the arguments) by name only.
    """
    pinned = {e.relative_path for e in definition.file_contents}
    name_only = set(definition.file_paths)
    if RUN_SCRIPT not in pinned or CONFIG_FILE not in name_only:
        return None
    python_files, json_files = pinned - {RUN_SCRIPT}, name_only - {CONFIG_FILE}
    if len(python_files) != 1 or len(json_files) != 1:
        return None
    layout = CallableLayout(python_files.pop(), json_files.pop())
    if not _is_code_file(layout.python_file, ".py"):
        return None
    if not _is_code_file(layout.params_file, ".json"):
        return None
    return layout


def _is_code_file(rel_path: str, suffix: str) -> bool:
    return rel_path.startswith(CODE_PREFIX) and rel_path.endswith(suffix)


class Api:
    """An api from a datasite. Call it to submit a job with arguments."""

    def __init__(
        self,
        name: str,
        datasite: str,
        api_dir: Path,
        definition: ApiDefinition,
        client: SyftRDSClient,
    ):
        self.name = name
        self.datasite = datasite
        self.api_dir = api_dir
        self.definition = definition
        self._client = client

    @property
    def args(self) -> list[ApiArg]:
        return self.definition.args

    @property
    def arg_names(self) -> list[str]:
        return [a.name for a in self.args]

    @property
    def layout(self) -> CallableLayout | None:
        return callable_layout(self.definition)

    @property
    def is_callable(self) -> bool:
        return self.layout is not None

    @property
    def run_script(self) -> bytes:
        return self._stored_file(RUN_SCRIPT).read_bytes()

    @property
    def code(self) -> str | None:
        """The Python code a call runs, or None if the api isn't callable."""
        layout = self.layout
        if layout is None:
            return None
        return self._stored_file(layout.python_file).read_text(encoding="utf-8")

    @property
    def call_signature(self) -> str:
        accessor = (
            f"client.api.{self.name}"
            if self.name.isidentifier()
            else f'client.api["{self.name}"]'
        )
        return f"{accessor}({', '.join(self.arg_names)})"

    def job_code_files(self, params: dict[str, Any]) -> dict[str, bytes]:
        """The files of a job's code/ folder for a call with these params."""
        layout = self.layout
        python_code = self._stored_file(layout.python_file).read_bytes()
        return {
            layout.entrypoint: python_code,
            layout.params_file.removeprefix(CODE_PREFIX): json.dumps(params).encode(),
        }

    def _stored_file(self, rel_path: str) -> Path:
        return self.api_dir / FILES_DIR_NAME / rel_path

    def bind_args(self, args: tuple, kwargs: dict) -> dict[str, Any]:
        """Map a call onto the api's arguments, with defaults and type checks."""
        return bind_args(self.name, self.args, args, kwargs)

    def __call__(self, *args: Any, block: bool = True, **kwargs: Any):
        """Submit a job running this api's code with the given arguments.

        With block=True (the default), waits for the job to finish and returns
        it; with block=False, returns right after submitting. An api argument
        named "block" can only be passed positionally.
        """
        if not self.is_callable:
            raise TypeError(
                f"Api '{self.name}' can't be called: it needs run.sh and one Python "
                f"file pinned by content, and one JSON file matched by name."
            )
        params = self.bind_args(args, kwargs)
        return self._client._submit_api_call(self, params, block=block)

    def __repr__(self) -> str:
        return api_repr_str(self)

    def _repr_html_(self) -> str:
        return api_repr_html(self)
