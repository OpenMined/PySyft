# third party
from pymongo import MongoClient

# relative
from .database_manager import NoSQLDatabaseManager


class NoSQLTaskManager(NoSQLDatabaseManager):
    """Class to manage user database actions."""

    _collection_name = "task"
    __canonical_object_name__ = "Task"

    def __init__(self, client: MongoClient, db_name: str) -> None:
        super().__init__(client, db_name)
