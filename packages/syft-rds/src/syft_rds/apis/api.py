"""A single api shared with the current user, as seen from the client."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from syft_rds.apis.args import ApiArg, bind_args
from syft_job.policies import PolicyScript

from syft_rds.apis.models import FILES_DIR_NAME, ApiDefinition, policy_dir
from syft_rds.apis.repr import api_repr_html, api_repr_str

if TYPE_CHECKING:
    from syft_rds.client import SyftRDSClient

CODE_PREFIX = "code/"

# What a call can pass for a file: its content, or a path to read it from.
CallFile = bytes | str | Path


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
    def params_file(self) -> str | None:
        """The file a call's arguments go in, e.g. "code/params.json"."""
        return self.definition.params_file

    @property
    def sent_files(self) -> list[str]:
        """The files a call passes in files=: what it sends besides the params."""
        return [f for f in self.definition.call_files if f != self.params_file]

    @property
    def python_file(self) -> str | None:
        """The one Python file the api pins under code/, if it pins one."""
        python_files = [
            e.relative_path
            for e in self.definition.file_contents
            if e.relative_path.startswith(CODE_PREFIX)
            and e.relative_path.endswith(".py")
        ]
        return python_files[0] if len(python_files) == 1 else None

    @property
    def code(self) -> str | None:
        """The Python code a call runs, when the api pins one Python file."""
        if self.python_file is None:
            return None
        return self._stored_file(self.python_file).read_text(encoding="utf-8")

    @property
    def input_policy_code(self) -> str | None:
        """The code of the input policy, or None if the api has none."""
        return self._policy_code("input", self.definition.input_policy)

    @property
    def output_policy_code(self) -> str | None:
        """The code of the output policy, or None if the api has none."""
        return self._policy_code("output", self.definition.output_policy)

    def _policy_code(self, kind: str, entries: list) -> str | None:
        if not entries:
            return None
        return PolicyScript(policy_dir(self.api_dir, kind)).code

    @property
    def call_signature(self) -> str:
        accessor = (
            f"client.api.{self.name}"
            if self.name.isidentifier()
            else f'client.api["{self.name}"]'
        )
        parts = self.arg_names + (["files={...}"] if self.sent_files else [])
        return f"{accessor}({', '.join(parts)})"

    def request_files(
        self, params: dict[str, Any], files: dict[str, CallFile]
    ) -> dict[str, bytes]:
        """Every file of a call: the params file, and the files passed in files=.

        Raises TypeError unless files= holds exactly the files the api takes.
        """
        if set(files) != set(self.sent_files):
            raise TypeError(
                f"{self.name}() sends files={sorted(self.sent_files)}, "
                f"got {sorted(files)}"
            )
        request = {path: _file_bytes(content) for path, content in files.items()}
        if self.params_file is not None:
            request[self.params_file] = json.dumps(params).encode()
        return request

    def _stored_file(self, rel_path: str) -> Path:
        return self.api_dir / FILES_DIR_NAME / rel_path

    def bind_args(self, args: tuple, kwargs: dict) -> dict[str, Any]:
        """Map a call onto the api's arguments, with defaults and type checks."""
        return bind_args(self.name, self.args, args, kwargs)

    def __call__(
        self,
        *args: Any,
        block: bool = True,
        files: dict[str, CallFile] | None = None,
        **kwargs: Any,
    ):
        """Call the api: send its arguments, and any files it takes, as a request.

        files= maps each file the api takes by name, other than the params
        file, to its content or a path, e.g. {"code/data.csv": "data.csv"}.
        With block=True (the default), waits for the job to finish and returns
        it; with block=False, returns right after submitting. An api argument
        named "block" or "files" can only be passed positionally.
        """
        params = self.bind_args(args, kwargs)
        request = self.request_files(params, files or {})
        return self._client._submit_api_call(self, request, block=block)

    def __repr__(self) -> str:
        return api_repr_str(self)

    def _repr_html_(self) -> str:
        return api_repr_html(self)


def _file_bytes(content: CallFile) -> bytes:
    if isinstance(content, bytes):
        return content
    return Path(content).expanduser().read_bytes()
