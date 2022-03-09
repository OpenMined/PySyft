# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import List
from typing import Union

# third party
import numpy as np

# relative
from ..common.serde.serializable import serializable
from .entity import Entity


@serializable(recursive_serde=True)
class EntityList:
    __attr_allowlist__ = ("one_hot_lookup", "entities_indexed")
    __slots__ = ("one_hot_lookup", "entities_indexed")

    def __init__(
        self, one_hot_lookup: List[Union[Entity, str]], entities_indexed: np.ndaray
    ) -> None:
        self.one_hot_lookup = one_hot_lookup
        self.entities_indexed = entities_indexed

    @staticmethod
    def from_objs(entities: Union[np.ndarray, list]) -> EntityList:
        if isinstance(entities, list):
            entities = np.array(entities)
        one_hot_lookup, entities_indexed = np.unique(entities, return_inverse=True)

        return EntityList(one_hot_lookup, entities_indexed)

    def __getitem__(self, key: Union[int, slice, str]) -> Union[Entity, str]:
        return self.one_hot_lookup[self.entities_indexed[key]]

    def copy(self, order: str = "K") -> EntityList:
        return EntityList(
            self.one_hot_lookup.copy(), self.entities_indexed.copy(order=order)
        )

    def __len__(self) -> int:
        return len(self.entities_indexed)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, EntityList):
            if (
                self.entities_indexed == other.entities_indexed
            ).all() and self.one_hot_lookup == other.one_hot_lookup:
                return True
            return False
        return self == other
