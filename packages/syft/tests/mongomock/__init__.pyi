# stdlib
from collections.abc import Sequence
from typing import Literal, TypeAlias
from unittest import mock

# third party

def patch(
    servers: str | tuple[str, int] | Sequence[str | tuple[str, int]] = ...,
    on_new: Literal["error", "create", "timeout", "pymongo"] = ...,
) -> mock._patch: ...

_FeatureName: TypeAlias = Literal["collation", "session"]

def ignore_feature(feature: _FeatureName) -> None: ...
def warn_on_feature(feature: _FeatureName) -> None: ...

SERVER_VERSION: str = ...
