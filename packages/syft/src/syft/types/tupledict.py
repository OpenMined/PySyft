# stdlib
from collections import OrderedDict
from typing import TypeVar
from typing import Union

_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


class TupleDict(OrderedDict[_KT, _VT]):
    def __getitem__(self, key: Union[int, _KT]) -> _VT:
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)

    def __len__(self) -> int:
        return len(self.keys())
