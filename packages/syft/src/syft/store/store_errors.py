# relative
from ..types.errors import SyftException


class StashException(SyftException): ...


class StashNotFoundException(StashException): ...
