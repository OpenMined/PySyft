# stdlib
import operator
from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple

# third party
from packaging import version

operators = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
}


def get_operator(version_string: str) -> Tuple[str, Callable, str]:
    op: Any = operator.ge
    op_char: str = ">="
    if len(version_string) > 2:
        two_char = version_string[0:2]
        one_char = version_string[0:1]
        if two_char in operators.keys():
            op_char = two_char
            version_string = version_string[2:]
        elif one_char in operators.keys():
            op_char = one_char
            version_string = version_string[1:]
    op = operators[op_char]
    return version_string, op, op_char


def make_requires(LATEST_STABLE_SYFT: str, __version__: str) -> Callable:
    def requires(version_string: str, silent: bool = False) -> Optional[bool]:
        version_string, op, op_char = get_operator(version_string)
        syft_version = version.parse(__version__)
        stable_version = version.parse(LATEST_STABLE_SYFT)
        required = version.parse(version_string)
        result = op(syft_version, required)
        if silent:
            return result

        if result:
            print(
                f"✅ The installed version of syft=={syft_version} matches "
                f"the requirement {op_char}{required}"
            )
        else:
            print(
                f"❌ The installed version of syft=={syft_version} doesn't match "
                f"the requirement {op_char}{required}."
            )
            pre = ""
            if required.minor > stable_version.minor:
                pre = " --pre"
            print(
                f"This code or notebook may have issues if APIs have changed.\n"
                f"Alternatively you could try to match {op_char}{required} it with:\n"
            )
            if required > syft_version:
                upgrade = f"pip install -U{pre} syft or "
            else:
                upgrade = ""
            msg = f"{upgrade}pip install syft=={required}"
            print(msg)
        return None

    return requires
