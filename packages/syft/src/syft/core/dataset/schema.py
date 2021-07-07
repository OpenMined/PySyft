# stdlib
from collections import OrderedDict
from functools import singledispatch
from typing import Any
from typing import Dict
import warnings

# third party
import pyarrow as pa

# relative
from .schema_entry import Entry


class Schema:
    def __init__(self):
        self.__schema__: Dict[str, Entry] = OrderedDict()
        self.pa_schema = None
        self.observability_metadata = None
        self.transform_mapping = None

    def __setattr__(self, key: str, value: Any):
        if isinstance(value, Entry):
            self.__schema__[key] = value

        warnings.warn(
            "If you want to add an object to the schema, you need to pass an Entry!"
        )

        super().__setattr__(key, value)

    def compile(self, pa_scheme: bool = True):
        if pa_scheme:
            self.pa_schema = pa.schema(
                [
                    (entry_name, entry.convert_to_pyarrow())
                    for entry_name, entry in self.__schema__.items()
                ]
            )

        self.observability_metadata = {
            entry_name: entry.observability
            for entry_name, entry in self.__schema__.items()
        }
        self.transform_mapping = {
            entry_name: entry.transform for entry_name, entry in self.__schema__.items()
        }

    def get_constraints(self) -> Dict[str, Any]:
        if self.pa_schema:
            return {"schema": self.pa_schema}
        else:
            return {"names": list(self.__schema__.keys())}
