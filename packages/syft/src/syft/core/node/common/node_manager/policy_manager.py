# third party
from pymongo import MongoClient

# relative
from .database_manager import NoSQLDatabaseManager


class NoSQLPolicyManager(NoSQLDatabaseManager):
    """Class to manage user database actions."""

    _collection_name = "policy"
    __canonical_object_name__ = "Policy"

    def __init__(self, client: MongoClient, db_name: str) -> None:
        super().__init__(client, db_name)
