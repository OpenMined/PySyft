# stdlib
from typing import Any
from typing import List
from typing import Union

# third party
from bcrypt import checkpw
from bcrypt import gensalt
from bcrypt import hashpw
from nacl.encoding import HexEncoder
from nacl.signing import VerifyKey

# relative
from ..exceptions import InvalidCredentialsError
from ..exceptions import UserNotFoundError
from ..node_table.user import SyftUser
from .database_manager import DatabaseManager
from .role_manager import RoleManager


class UserManager(DatabaseManager):

    schema = SyftUser

    def __init__(self, database):
        super().__init__(schema=UserManager.schema, db=database)
        self.roles = RoleManager(database)

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
        self,
        name: str,
        email: str,
        password: str,
        role: int,
        private_key: str,
        verify_key: str,
    ):
        salt, hashed = self.__salt_and_hash_password(password, 12)
        return self.register(
            name=name,
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

    def first(self, **kwargs) -> Union[None, SyftUser]:
        result = super().first(**kwargs)
        if not result:
            raise UserNotFoundError
        return result

    def login(self, email: str, password: str) -> SyftUser:
        return self.__login_validation(email, password)

    def set(
        self,
        user_id: str,
        email: str = None,
        password: str = None,
        role: int = 0,
        name: str = "",
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
        elif name:
            key = "name"
            value = name
        else:
            raise Exception

        self.modify({"id": user_id}, {key: value})

    def can_create_users(self, verify_key: VerifyKey) -> bool:
        try:
            return self.role(verify_key=verify_key).can_create_users
        except UserNotFoundError:
            return False

    def can_upload_data(self, verify_key: VerifyKey) -> bool:
        try:
            return self.role(verify_key=verify_key).can_upload_data
        except UserNotFoundError:
            return False

    def can_triage_requests(self, verify_key: VerifyKey) -> bool:
        try:
            return self.role(verify_key=verify_key).can_triage_requests
        except UserNotFoundError:
            return False

    def can_manage_infrastructure(self, verify_key: VerifyKey) -> bool:
        try:
            return self.role(verify_key=verify_key).can_manage_infrastructure
        except UserNotFoundError:
            return False

    def can_edit_roles(self, verify_key: VerifyKey) -> bool:
        try:
            return self.role(verify_key=verify_key).can_edit_roles
        except UserNotFoundError:
            return False

    def can_create_groups(self, verify_key: VerifyKey) -> bool:
        try:
            return self.role(verify_key=verify_key).can_create_groups
        except UserNotFoundError:
            return False

    def role(self, verify_key: VerifyKey):
        user_id = self.get_user(verify_key).id
        user = self.first(id=user_id)
        return self.roles.first(id=user.role)

    def get_user(self, verify_key: VerifyKey) -> Any:
        return self.first(
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
        )

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
        salt_len = len(salt)
        hashed = hashpw(password, salt)
        hashed = hashed[salt_len:]
        hashed = hashed.decode("UTF-8")
        salt = salt.decode("UTF-8")
        return salt, hashed
