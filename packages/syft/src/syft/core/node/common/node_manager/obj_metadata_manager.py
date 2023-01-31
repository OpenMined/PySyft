# stdlib
from typing import Any
from typing import Optional

# relative
from ..node_table.bin_obj_metadata import NoSQLObjectMetadata
from .database_manager import NoSQLDatabaseManager


class NoSQLObjectMetadataManager(NoSQLDatabaseManager):
    """Class to manage binary object metadata database actions."""

    _collection_name = "obj_metadata"
    __canonical_object_name__ = "ObjectMetadata"

    def first(self, **kwargs: Any) -> Optional[NoSQLObjectMetadata]:
        result = super().find_one(kwargs)
        return result

    def create_metadata(self, **kwargs: Any) -> NoSQLObjectMetadata:
        obj = NoSQLObjectMetadata(**kwargs)
        return self.add(obj)
