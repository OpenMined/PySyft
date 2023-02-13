"""This file defines classes and methods which are used to manage database queries on the UserApplication table."""

# stdlib
from typing import Any
from typing import List

# relative
from ..node_table.user import NoSQLUserApplication
from .database_manager import NoSQLDatabaseManager


class UserApplicationNotFoundError(Exception):
    pass


class NoSQLUserApplicationManager(NoSQLDatabaseManager):
    """Class to manage user application database actions."""

    _collection_name = "user_application"
    __canonical_object_name__ = "UserApplication"

    def all(self) -> List[NoSQLUserApplication]:
        return super().all()

    def first(self, **kwargs: Any) -> NoSQLUserApplication:
        result = super().find_one(kwargs)
        if not result:
            raise UserApplicationNotFoundError
        return result
