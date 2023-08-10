# stdlib
from inspect import Parameter
from inspect import Signature
import json
import re
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

# third party
from docker.models.containers import ExecResult
from typing_extensions import Self

# relative
from ...serde.serializable import serializable
from ...service.response import SyftError
from ...service.response import SyftSuccess
from ...types.file import SyftFile
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject

ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


@serializable()
class ContainerVolume(SyftObject):
    # version
    __canonical_name__ = "ContainerVolume"
    __version__ = SYFT_OBJECT_VERSION_1

    __attr_searchable__ = ["name", "internal_mountpath"]
    __attr_unique__ = ["name"]
    __repr_attrs__ = ["name", "internal_mountpath"]

    name: str
    internal_mountpath: str
    mode: str = "ro"


@serializable()
class ContainerImage(SyftObject):
    # version
    __canonical_name__ = "ContainerImage"
    __version__ = SYFT_OBJECT_VERSION_1

    __attr_searchable__ = ["name", "tag"]
    __attr_unique__ = ["name", "tag"]
    __repr_attrs__ = ["name", "tag"]

    name: str
    tag: str
    dockerfile: Optional[str]
    volumes: List[ContainerVolume] = []


@serializable()
class ContainerUpload(SyftObject):
    # version
    __canonical_name__ = "ContainerUpload"
    __version__ = SYFT_OBJECT_VERSION_1

    arg_name: str
    sandbox_path: Optional[str] = "/sandbox"

    def format(self, run_kwargs: Dict[str, Any]) -> str:
        if self.arg_name not in run_kwargs["files"]:
            raise Exception(f"Missing arg_name: {self.arg_name}")
        syft_file = run_kwargs["files"][self.arg_name]
        path = ""
        if self.sandbox_path:
            path = f"{self.sandbox_path}/"
        return f"{path}{syft_file.filename}"


class ContainerCommandArg(SyftObject):
    # version
    __canonical_name__ = "ContainerArg"
    __version__ = SYFT_OBJECT_VERSION_1

    required: bool = False


@serializable()
class ContainerCommandKwarg(ContainerCommandArg):
    # version
    __canonical_name__ = "ContainerCommandKwarg"
    __version__ = SYFT_OBJECT_VERSION_1

    signature_key: Optional[str] = None  # TODO: overrides name
    name: str
    hyphens: str = "--"
    equals: str = "="
    value: Union[str, Type[str], ContainerUpload]
    arg_only: bool = False  # no key= # TODO: refactor to own class

    def format(self, run_kwargs: Dict[str, Any]) -> Optional[str]:
        if self.name in run_kwargs["user_kwargs"]:
            value_str = run_kwargs["user_kwargs"][self.name].strip()
        else:
            value_str = (
                self.value
                if isinstance(self.value, str)
                else self.value.format(run_kwargs)
            )
        if self.arg_only:
            return f"{value_str}"
        return f"{self.hyphens}{self.name}{self.equals}{value_str}"


@serializable()
class ContainerCommandKwargBool(ContainerCommandKwarg):
    # version
    __canonical_name__ = "ContainerCommandKwargBool"
    __version__ = SYFT_OBJECT_VERSION_1

    name: str
    hyphens: str = "--"
    equals: str = "="
    value: bool = True
    flag: bool = True

    def format(self, run_kwargs: Dict[str, Any]) -> Optional[str]:
        value = self.value
        if self.name in run_kwargs["user_kwargs"]:
            value = bool(run_kwargs["user_kwargs"][self.name])
        if self.flag and value:
            return f"{self.hyphens}{self.name}"
        if self.flag and not self.value:
            return None
        return f"{self.hyphens}{self.name}{self.equals}{self.value}"


@serializable()
class ContainerCommandKwargTemplate(ContainerCommandKwarg):
    # version
    __canonical_name__ = "ContainerCommandKwargTemplate"
    __version__ = SYFT_OBJECT_VERSION_1

    name: str
    hyphens: str = "--"
    equals: str = "="
    value: str
    inputs: Dict[str, Union[str, ContainerUpload]]
    append: str = ""

    def format(self, run_kwargs: Dict[str, Any]) -> Optional[str]:
        cmd = ""
        prepared_inputs = {}
        for key, value in self.inputs.items():
            prepared_inputs[key] = (
                value if isinstance(value, str) else value.format(run_kwargs)
            )
        cmd += self.value.format(**prepared_inputs)
        return f"{self.hyphens}{self.name}{self.equals}{cmd}{self.append}"


@serializable()
class ContainerMount(SyftObject):
    # version
    __canonical_name__ = "ContainerMount"
    __version__ = SYFT_OBJECT_VERSION_1

    internal_filepath: str
    file: SyftFile
    mode: str = "ro"
    unix_permission: str = "644"


@serializable()
class ContainerCommand(SyftObject):
    # version
    __canonical_name__ = "ContainerCommand"
    __version__ = SYFT_OBJECT_VERSION_1

    module_name: str
    image_name: str
    doc_string: str = ""
    name: str
    command: str
    args: str
    kwargs: Dict[str, ContainerCommandArg] = {}
    user_kwargs: List[str] = []
    extra_user_kwargs: Dict[str, Type] = {}
    user_files: List[str] = []
    return_filepath: Optional[str] = None
    mounts: List[ContainerMount] = []

    __attr_searchable__ = ["module_name", "name", "image_name"]
    __attr_unique__ = ["name"]
    __repr_attrs__ = ["module_name", "name", "image_name"]

    def cmd(
        self,
        run_user_kwargs: Dict[str, Any],
        run_files: Dict[str, SyftFile],
        run_extra_kwargs: Dict[str, Any],
    ) -> str:
        run_kwargs = {}
        run_kwargs["user_kwargs"] = run_user_kwargs
        run_kwargs["files"] = run_files
        cmd = ""
        cmd += f"{self.command} "
        cmd += f"{self.args} "
        for _k, v in self.kwargs.items():
            value = v.format(run_kwargs)
            if value is not None:
                cmd += f"{value} "
        return cmd.strip()

    def user_signature(self) -> Signature:
        parameters = []
        keys = self.user_kwargs + self.user_files
        for key in keys:
            if key in self.kwargs:
                kwarg = self.kwargs[key]
                param_name = kwarg.name.replace("-", "_")
                if isinstance(kwarg.value, ContainerUpload):
                    param_type = SyftFile
                elif isinstance(kwarg.value, type):
                    param_type = kwarg.value
                else:
                    param_type = type(kwarg.value)
                if not kwarg.required:
                    param_type = Optional[param_type]
            else:
                param_name = key.replace("-", "_")
                param_type = Union[SyftFile, List[SyftFile]]

            parameter = Parameter(
                name=param_name, kind=Parameter.KEYWORD_ONLY, annotation=param_type
            )
            parameters.append(parameter)

        for key, param_type in self.extra_user_kwargs.items():
            param_name = key.replace("-", "_")
            parameter = Parameter(
                name=param_name, kind=Parameter.KEYWORD_ONLY, annotation=param_type
            )
            parameters.append(parameter)

        return Signature(parameters=parameters)


def extract_json_objects(text: str) -> List[Dict[str, Any]]:
    """Generate all JSON objects in a string."""
    decoder = json.JSONDecoder()
    pos = 0
    while True:
        match = text.find("{", pos)
        if match == -1:
            break
        try:
            result, index = decoder.raw_decode(text[match:])
            yield result
            pos = match + index
        except ValueError:
            pos = match + 1


@serializable()
class ContainerResult(SyftObject):
    # version
    __canonical_name__ = "ContainerResult"
    __version__ = SYFT_OBJECT_VERSION_1

    image_name: Optional[str] = None
    image_tag: Optional[str] = None
    command_name: Optional[str] = None
    exit_code: int
    stdout: List[str]
    stderr: List[str]
    jsonstd: List[Dict[str, Any]]
    jsonerr: List[Dict[str, Any]]
    return_file: Optional[SyftFile] = None

    def _repr_html_(self) -> str:
        return_file_name = ""
        if self.return_file:
            return_file_name = f" With file: {self.return_file.filename}."

        if self.exit_code == 0:
            with_json = ""
            if self.jsonstd:
                with_json = " With json."
            return SyftSuccess(
                message=f"Command Succeeded.{return_file_name}{with_json}"
            )._repr_html_()
        else:
            with_json = ""
            if self.jsonerr:
                with_json = " With json."
            return SyftError(
                message=f"Command Failed.{return_file_name}{with_json}"
            )._repr_html_()

    @staticmethod
    def from_execresult(result: ExecResult) -> Self:
        stdout = []
        stderr = []
        jsonstd = []
        jsonerr = []
        if result.output:
            out, err = result.output
            if out is not None:
                stdout = out.decode("utf-8").strip()
                stdout = ansi_escape.sub("", stdout)
                jsonstd = list(extract_json_objects(stdout))
                stdout = stdout.splitlines()
            if err is not None:
                stderr = err.decode("utf-8").strip()
                stderr = ansi_escape.sub("", stderr)
                jsonerr = list(extract_json_objects(stderr))
                stderr = stderr.splitlines()

        return ContainerResult(
            exit_code=result.exit_code,
            stdout=stdout,
            jsonstd=jsonstd,
            stderr=stderr,
            jsonerr=jsonerr,
        )
