# stdlib
from typing import Any
from typing import Dict

# third party
from sqlalchemy import inspect
from sqlalchemy.ext.declarative import as_declarative


@as_declarative()
class Base:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    id: Any
    __name__: str

    def _asdict(self) -> Dict:
        return {c.key: getattr(self, c.key) for c in inspect(self).mapper.column_attrs}
