# stdlib
from typing import Any

# relative
from ..node_table.bin_obj_metadata import NoSQLObjectMetadata
from .database_manager import NoSQLDatabaseManager


class ObjectMetadataNotFoundError(Exception):
    pass


class NoSQLObjectMetadataManager(NoSQLDatabaseManager):
    """Class to manage binary object metadata database actions."""

    _collection_name = "obj_metadata"
    __canonical_object_name__ = "ObjectMetadata"

    def first(self, **kwargs: Any) -> NoSQLObjectMetadata:
        result = super().find_one(kwargs)
        if not result:
            raise ObjectMetadataNotFoundError
        return result

    def create_metadata(self, **kwargs: Any) -> NoSQLObjectMetadata:
        obj = NoSQLObjectMetadata(**kwargs)
        return self.add(obj)
