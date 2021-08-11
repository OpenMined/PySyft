# stdlib
from typing import Any

# third party
from sqlalchemy.ext.declarative import as_declarative


@as_declarative()
class Base:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    id: Any
    __name__: str
