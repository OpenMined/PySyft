# stdlib
from typing import Dict
from typing import List
from typing import Union

# third party
from bcrypt import checkpw
from bcrypt import gensalt
from bcrypt import hashpw

# grid relative
from ..database.users.user import User
from ..exceptions import AuthorizationError
from ..exceptions import InvalidCredentialsError
from ..exceptions import UserNotFoundError
from .database_manager import DatabaseManager
from .role_manager import RoleManager


class UserManager(DatabaseManager):

    schema = User

    def __init__(self, database):
        self._schema = UserManager.schema
        self.roles = RoleManager(database)
        self.db = database

    @property
    def common_users(self) -> list:
        common_users = []
        for role in self.roles.common_roles:
            common_users = common_users + list(super().query(role=role.id))

        return common_users

    @property
    def org_users(self) -> list:
        org_users = []
        for role in self.roles.org_roles:
            org_users = org_users + list(super().query(role=role.id))
        return org_users

    def signup(
        self, email: str, password: str, role: int, private_key: str, verify_key: str
    ):
        salt, hashed = self.__salt_and_hash_password(password, 12)
        return self.register(
            email=email,
            role=role,
            private_key=private_key,
            verify_key=verify_key,
            hashed_password=hashed,
            salt=salt,
        )

    def query(self, **kwargs) -> Union[None, List]:
        results = super().query(**kwargs)
        if len(results) == 0:
            raise UserNotFoundError
        return results

    def first(self, **kwargs) -> Union[None, User]:
        result = super().first(**kwargs)
        if not result:
            raise UserNotFoundError
        return result

    def login(self, email: str, password: str) -> User:
        return self.__login_validation(email, password)

    def set(
        self,
        user_id: str,
        email: str = None,
        password: str = None,
        role: int = 0,
    ) -> None:
        if not self.contain(id=user_id):
            raise UserNotFoundError

        if email:
            key = "email"
            value = email
        elif password:
            salt, hashed = self.__salt_and_hash_password(password, 12)
            self.modify({"id": user_id}, {"salt": salt, "hashed_password": hashed})
            return
        elif role != 0:
            key = "role"
            value = role
        else:
            raise Exception

        self.modify({"id": user_id}, {key: value})

    def can_create_users(self, user_id: str) -> bool:
        role = self.role(user_id=user_id)
        if role:
            return role.can_create_users
        else:
            return False

    def can_upload_data(self, user_id: str) -> bool:
        role = self.role(user_id=user_id)
        if role:
            return role.can_upload_data
        else:
            return False

    def can_triage_requests(self, user_id: str) -> bool:
        return self.role(user_id=user_id).can_triage_requests

    def can_manage_infrastructure(self, user_id: str) -> bool:
        return self.role(user_id=user_id).can_manage_infrastructure

    def can_edit_roles(self, user_id: str) -> bool:
        return self.role(user_id=user_id).can_edit_roles

    def can_create_groups(self, user_id: str) -> bool:
        return self.role(user_id=user_id).can_create_groups

    def can_edit_settings(self, user_id: str) -> bool:
        return self.role(user_id=user_id).can_edit_settings

    def role(self, user_id: int):
        try:
            user = self.first(id=user_id)
            return self.roles.first(id=user.role)
        except UserNotFoundError:
            return False

    def __login_validation(self, email: str, password: str) -> bool:
        try:
            user = self.first(email=email)

            hashed = user.hashed_password.encode("UTF-8")
            salt = user.salt.encode("UTF-8")
            password = password.encode("UTF-8")

            if checkpw(password, salt + hashed):
                return user
            else:
                raise InvalidCredentialsError
        except UserNotFoundError:
            raise InvalidCredentialsError

    def __salt_and_hash_password(self, password, rounds):
        password = password.encode("UTF-8")
        salt = gensalt(rounds=rounds)
        hashed = hashpw(password, salt)
        hashed = hashed[len(salt) :]
        hashed = hashed.decode("UTF-8")
        salt = salt.decode("UTF-8")
        return salt, hashed
