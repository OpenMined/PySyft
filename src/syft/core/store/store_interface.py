from abc import ABC
from typing import List, Optional

from ...decorators import syft_decorator
from ...common.id import UID
from . import StorableObject


class ObjectStore:
    """Logic to store and retrieve objects within a worker"""

    def __init__(self):
        self._objects = {}

    @syft_decorator(typechecking=True)
    def __sizeof__(self) -> int:
        raise NotImplementedError

    @syft_decorator(typechecking=True)
    def __str__(self) -> int:
        raise NotImplementedError

    @syft_decorator(typechecking=True)
    def __len__(self) -> int:
        raise NotImplementedError

    @syft_decorator(typechecking=True)
    def keys(self) -> [UID]:
        raise NotImplementedError

    @syft_decorator(typechecking=True)
    def values(self) -> [StorableObject]:
        raise NotImplementedError

    @syft_decorator(typechecking=True)
    def __contains__(self, key: UID) -> bool:
        raise NotImplementedError

    @syft_decorator(typechecking=True)
    def __getitem__(self, key: UID) -> StorableObject:
        raise NotImplementedError

    @syft_decorator(typechecking=True)
    def __setitem__(self, key: UID, value: StorableObject) -> None:
        raise NotImplementedError

    @syft_decorator(typechecking=True)
    def __delitem__(self, key: UID) -> None:
        raise NotImplementedError

    @syft_decorator(typechecking=True)
    def clear(self) -> None:
        raise NotImplementedError
