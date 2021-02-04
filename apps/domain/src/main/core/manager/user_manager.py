from typing import Dict
from bcrypt import checkpw, gensalt, hashpw

from .database_manager import DatabaseManager
from .role_manager import RoleManager
from ..database.users.user import User
from ..exceptions import (
    AuthorizationError,
    GroupNotFoundError,
    InvalidCredentialsError,
    MissingRequestKeyError,
    PyGridError,
    RoleNotFoundError,
    UserNotFoundError,
)


class UserManager(DatabaseManager):

    schema = User

    def __init__(self, database):
        self._schema = UserManager.schema
        self.roles = RoleManager(database)
        self.db = database

    def signup(self, email: str, password: str, role: int, private_key: str):
        salt, hashed = self.__salt_and_hash_password(password, 12)
        self.register(
            email=email,
            role=role,
            private_key=private_key,
            hashed_password=hashed,
            salt=salt,
        )

    def login(self, email: str, password: str) -> User:
        return self.__login_validation(email, password)

    def set(
        self,
        user_id: str,
        email: str = None,
        hashed_password: str = None,
        role: int = 0,
    ) -> None:
        if email:
            key = "email"
            value = email
        elif hashed_password:
            key = "hashed_password"
            value = hashed_password
        elif role != 0:
            key = "role"
            value = role
        else:
            raise Exception

        self.modify({"id": user_id}, {key: value})

    def role(self, user_id: int):
        query_result = self.query(id=user_id)
        user = query_result[0]
        return self.roles.query(id=user.role)[0]

    def __login_validation(self, email: str, password: str) -> bool:
        query_result = self.query(email=email)

        if len(query_result) != 0:
            user = query_result[0]
        else:
            raise InvalidCredentialsError

        hashed = user.hashed_password.encode("UTF-8")
        salt = user.salt.encode("UTF-8")
        password = password.encode("UTF-8")

        if checkpw(password, salt + hashed):
            return user
        else:
            raise InvalidCredentialsError

    def __salt_and_hash_password(self, password, rounds):
        password = password.encode("UTF-8")
        salt = gensalt(rounds=rounds)
        hashed = hashpw(password, salt)
        hashed = hashed[len(salt) :]
        hashed = hashed.decode("UTF-8")
        salt = salt.decode("UTF-8")
        return salt, hashed
