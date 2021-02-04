from .database_manager import DatabaseManager
from ..database.groups.groups import Group

from ..exceptions import (
    AuthorizationError,
    GroupNotFoundError,
    InvalidCredentialsError,
    MissingRequestKeyError,
    PyGridError,
    RoleNotFoundError,
    UserNotFoundError,
)


class GroupManager(DatabaseManager):

    schema = Group

    def __init__(self, database):
        self._schema = GroupManager.schema
        self.db = database
