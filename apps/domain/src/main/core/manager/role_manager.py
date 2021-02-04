from .database_manager import DatabaseManager
from ..database.roles.roles import Role
from ..exceptions import (
    AuthorizationError,
    GroupNotFoundError,
    InvalidCredentialsError,
    MissingRequestKeyError,
    PyGridError,
    RoleNotFoundError,
    UserNotFoundError,
)


class RoleManager(DatabaseManager):

    schema = Role

    def __init__(self, database):
        self._schema = RoleManager.schema
        self.db = database
