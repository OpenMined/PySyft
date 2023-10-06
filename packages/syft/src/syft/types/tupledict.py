# stdlib
from collections import OrderedDict
from typing import Any
from typing import Union


class TupleDict(OrderedDict):
    def __getitem__(self, key: Union[str, int]) -> Any:
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)
