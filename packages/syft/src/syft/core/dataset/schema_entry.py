# stdlib
from dataclasses import dataclass
from typing import Callable
from typing import Dict
from typing import Optional

# third party
import pyarrow as pa

# relative
from ..common.serde.serializable import bind_protobuf

type_mapping: Dict[type, Callable] = {
    str: pa.string,
    bin: pa.binary,
    bool: pa.uint8,
    int: pa.int64,
    float: pa.float64,
}


class Entry:
    __slots__ = ["observability", "entry_type", "values_set", "transform"]

    def __init__(
        self,
        entry_type: Optional[type] = None,
        observability: str = "public",
        values_set: Optional[set] = None,
        transform: Optional[Callable] = None,
    ):
        self.entry_type = entry_type
        self.observability = observability
        self.values_set = values_set
        self.transform = transform

    def convert_to_pyarrow(self):
        return type_mapping[self.entry_type]()
