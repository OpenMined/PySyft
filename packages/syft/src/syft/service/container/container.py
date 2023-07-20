# stdlib
from inspect import Parameter
from inspect import Signature
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

# relative
from ...serde.serializable import serializable
from ...types.file import SyftFile
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject


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


@serializable()
class ContainerUpload(SyftObject):
    # version
    __canonical_name__ = "ContainerUpload"
    __version__ = SYFT_OBJECT_VERSION_1

    command_file: str
    sandbox_path: Optional[str] = None

    def format(self, run_kwargs: Dict[str, Any]) -> str:
        if self.command_file not in run_kwargs["files"]:
            raise Exception(f"Missing command_file: {self.command_file}")
        syft_file = run_kwargs["files"][self.command_file]
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

    name: str
    hyphens: str = "--"
    equals: str = "="
    value: Union[str, ContainerUpload]

    def format(self, run_kwargs: Dict[str, Any]) -> Optional[str]:
        value_str = (
            self.value if isinstance(self.value, str) else self.value.format(run_kwargs)
        )
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
        if self.flag and self.value:
            return f"{self.hyphens}{self.name}"
        if not self.flag and not self.value:
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
    user_kwargs: List[str] = {}

    __attr_searchable__ = ["module_name", "name", "image_name"]
    __attr_unique__ = ["name", "image_name"]
    __repr_attrs__ = ["module_name", "name", "image_name"]

    def cmd(self, run_files: Dict[str, SyftFile]) -> str:
        run_kwargs = {}
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
        for key in self.user_kwargs:
            kwarg = self.kwargs[key]
            param_name = kwarg.name
            param_type = (
                SyftFile
                if isinstance(kwarg.value, ContainerUpload)
                else type(kwarg.value)
            )
            if not kwarg.required:
                param_type = Optional[param_type]
            parameter = Parameter(
                name=param_name, kind=Parameter.KEYWORD_ONLY, annotation=param_type
            )
            parameters.append(parameter)
        return Signature(parameters=parameters)
