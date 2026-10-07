"""Api arguments: names, JSON types and defaults, and checking a call."""

from typing import Any, Literal

from pydantic import BaseModel

ArgType = Literal["int", "float", "str", "bool", "list", "dict", "any"]

# bool before int: True is an int in Python, but not in JSON.
_INFERRED_TYPES: list[tuple[type, ArgType]] = [
    (bool, "bool"),
    (int, "int"),
    (float, "float"),
    (str, "str"),
    (list, "list"),
    (dict, "dict"),
]


class ApiArg(BaseModel):
    """One argument of an api, stored in api.yaml."""

    name: str
    type: ArgType = "any"
    default: Any = None
    required: bool = False  # True: no default, the caller must pass it

    def describe(self) -> str:
        """e.g. "a: int = 1", or "a: int" for a required argument."""
        text = f"{self.name}: {self.type}"
        return text if self.required else f"{text} = {self.default!r}"

    def matches(self, value: Any) -> bool:
        if self.type == "any":
            return True
        if isinstance(value, bool):
            return self.type == "bool"
        if self.type == "float":
            return isinstance(value, (int, float))
        if self.type == "list":
            return isinstance(value, (list, tuple))
        return infer_type(value) == self.type


def infer_type(value: Any) -> ArgType:
    for python_type, arg_type in _INFERRED_TYPES:
        if isinstance(value, python_type):
            return arg_type
    return "any"


def infer_args(params: dict[str, Any]) -> list[ApiArg]:
    """Arguments from an example params dict: its values become the defaults."""
    return [
        ApiArg(name=name, type=infer_type(value), default=value)
        for name, value in params.items()
    ]


def bind_args(
    api_name: str, spec: list[ApiArg], args: tuple, kwargs: dict
) -> dict[str, Any]:
    """Map a call onto the spec, fill in defaults and check the types."""
    names = [a.name for a in spec]
    if len(args) > len(spec):
        raise TypeError(f"{api_name}() takes {len(spec)} arguments, got {len(args)}")
    params = dict(zip(names, args))
    for key, value in kwargs.items():
        if key not in names:
            raise TypeError(f"{api_name}() got an unexpected argument '{key}'")
        if key in params:
            raise TypeError(f"{api_name}() got multiple values for '{key}'")
        params[key] = value
    missing = [a.name for a in spec if a.required and a.name not in params]
    if missing:
        raise TypeError(f"{api_name}() is missing arguments: {missing}")
    return {a.name: _checked(api_name, a, params.get(a.name, a.default)) for a in spec}


def _checked(api_name: str, arg: ApiArg, value: Any) -> Any:
    if not arg.matches(value):
        raise TypeError(
            f"{api_name}() argument '{arg.name}' must be {arg.type}, "
            f"got {type(value).__name__}"
        )
    return value
