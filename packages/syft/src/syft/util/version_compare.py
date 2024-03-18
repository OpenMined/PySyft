# stdlib
from collections.abc import Callable
import operator
from typing import Any

# third party
from packaging import version

operators = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
}


def get_operator(version_string: str) -> tuple[str, Callable, str]:
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


def check_rule(
    version_string: str, LATEST_STABLE_SYFT: str, __version__: str
) -> tuple[Any, list[str], list[str]]:
    version_string, op, op_char = get_operator(version_string)
    syft_version = version.parse(__version__)
    stable_version = version.parse(LATEST_STABLE_SYFT)
    required = version.parse(version_string)
    result = op(syft_version, required)

    requirements = []
    messages = []

    if result:
        requirements.append(f"the requirement {op_char}{required}")
    else:
        requirements.append(f"the requirement {op_char}{required}")
        pre = ""
        if required.minor > stable_version.minor:
            pre = " --pre"
        msg = f"Alternatively you could try to match {op_char}{required} with:\n"
        if required > syft_version:
            upgrade = f"pip install -U{pre} syft or "
        else:
            upgrade = ""
        msg += f"{upgrade}pip install syft=={required}"
        messages.append(msg)
    return result, requirements, messages


def make_requires(LATEST_STABLE_SYFT: str, __version__: str) -> Callable:
    def requires(version_string: str, silent: bool = False) -> bool | None:
        syft_version = version.parse(__version__)
        parts = version_string.split(",")
        result = True
        all_requirements = []
        all_messages = []
        for part in parts:
            part_result, requirements, messages = check_rule(
                version_string=part,
                LATEST_STABLE_SYFT=LATEST_STABLE_SYFT,
                __version__=__version__,
            )
            all_requirements += requirements
            all_messages += messages
            if not part_result:
                result = False

        if silent:
            return result

        msg_requirements = " and ".join(all_requirements)
        if result:
            print(
                f"✅ The installed version of syft=={syft_version} matches {msg_requirements}"
            )
        else:
            print(
                f"❌ The installed version of syft=={syft_version} doesn't match {msg_requirements}"
            )
        if len(all_messages):
            print("This code or notebook may have issues if APIs have changed\n")
            print("\n\n".join(all_messages))
        return None

    return requires
