# stdlib
from typing import Any

# relative
from ...serde.serializable import serializable
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...util.markdown import as_markdown_python_code
from ...util.util import get_qualname_for
from ..code.user_code import UserCode


@serializable()
class VerifiableOutput(SyftObject):
    __canonical_name__ = "VerifiableOutput"
    __version__ = SYFT_OBJECT_VERSION_1

    code: UserCode
    enclave_output: Any

    __repr_attrs__ = ["inputs", "code"]

    @property
    def inputs(self) -> list[dict[str, str]]:
        code_init_kwargs = (
            self.code.input_policy_init_kwargs.values()
            if self.code.input_policy_init_kwargs is not None
            else []
        )
        code_kwargs_uid_to_name = {
            uid: name for d in code_init_kwargs for name, uid in d.items()
        }
        code_kwargs_uid_to_hash = self.code.input_id2hash
        inputs = [
            {"id": str(uid), "name": name, "hash": code_kwargs_uid_to_hash[uid]}
            for uid, name in code_kwargs_uid_to_name.items()
        ]
        return inputs

    @property
    def output(self) -> Any:
        return self.enclave_output

    # output_hash: str
    # enclave_key: str
    # enclave_signature: str

    # def _html_repr_() -> str:
    #     # pretty print the table of result and hashesh
    #     # call result.output for real output

    def _repr_markdown_(self, wrap_as_python: bool = True, indent: int = 0) -> str:
        s_indent = " " * indent * 2
        class_name = get_qualname_for(type(self))
        _repr_str = f"{s_indent}class {class_name}:\n"
        _repr_str += f"{s_indent}  id: UID = {self.id}\n"
        _repr_str += f"{s_indent}  inputs:\n"
        _repr_str += (
            "\n".join(
                [
                    f'{s_indent}    - id: UID = {i["id"]}\n'
                    f'{s_indent}      name: str = "{i["name"]}"\n'
                    f'{s_indent}      hash: str = "{i["hash"]}"'
                    for i in self.inputs
                ]
            )
            + "\n"
        )
        _repr_str += f"{s_indent}  code: UserCode\n"
        _repr_str += f"{s_indent}    id: UID = {self.code.id}\n"
        _repr_str += f'{s_indent}    func_name: str = "{self.code.service_func_name}"\n'
        _repr_str += f'{s_indent}    hash: str = "{self.code.code_hash}"\n'
        _repr_str += f"{s_indent}    raw_code: str\n"
        _repr_str += "\n".join(
            [
                f"{'  '*3}{substring}"
                for substring in self.code.raw_code.split("\n")[:-1]
            ]
        )

        if wrap_as_python:
            return (
                as_markdown_python_code(_repr_str)
                + "\n\n**Call `.output` to view the output.**\n"
            )
        return _repr_str + "\n\n**Call `.output` to view the output.**\n"
