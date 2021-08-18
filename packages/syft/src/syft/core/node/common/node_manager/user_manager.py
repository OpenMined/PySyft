# stdlib
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
from bcrypt import checkpw
from bcrypt import gensalt
from bcrypt import hashpw
from nacl.encoding import HexEncoder
from nacl.signing import VerifyKey
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Query

# syft absolute
from syft.core.node.common.node_table.roles import Role

# relative
from ..exceptions import InvalidCredentialsError
from ..exceptions import UserNotFoundError
from ..node_table.user import SyftUser
from .database_manager import DatabaseManager
from .role_manager import RoleManager


class UserManager(DatabaseManager):

    schema = SyftUser

    def __init__(self, database: Engine) -> None:
        super().__init__(schema=UserManager.schema, db=database)
        self.roles = RoleManager(database)

    @property
    def common_users(self) -> list:
        common_users: List[SyftUser] = []
        for role in self.roles.common_roles:
            common_users = common_users + list(super().query(role=role.id))

        return common_users

    @property
    def org_users(self) -> list:
        org_users: List[SyftUser] = []
        for role in self.roles.org_roles:
            org_users = org_users + list(super().query(role=role.id))
        return org_users

    def signup(
        self,
        name: str,
        email: str,
        password: str,
        budget: float,
        role: int,
        private_key: str,
        verify_key: str,
    ) -> SyftUser:
        salt, hashed = self.__salt_and_hash_password(password, 12)
        return self.register(
            name=name,
            email=email,
            role=role,
            budget=budget,
            private_key=private_key,
            verify_key=verify_key,
            hashed_password=hashed,
            salt=salt,
        )

    def query(self, **kwargs: Any) -> Query:
        results = super().query(**kwargs)
        return results

    def first(self, **kwargs: Any) -> Optional[SyftUser]:
        result = super().first(**kwargs)
        if not result:
            raise UserNotFoundError
        return result

    def login(self, email: str, password: str) -> SyftUser:
        return self.__login_validation(email, password)

    def set(  # nosec
        self,
        user_id: str,
        email: str = "",
        password: str = "",
        role: int = 0,
        name: str = "",
        budget: float = 0.0,
    ) -> None:
        if not self.contain(id=user_id):
            raise UserNotFoundError

        key: str
        value: Union[str, int, float]

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
        elif budget:
            key = "budget"
            value = budget
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

    def role(self, verify_key: VerifyKey) -> Role:
        user = self.get_user(verify_key)
        if not user:
            raise UserNotFoundError
        return self.roles.first(id=user.role)

    def get_user(self, verify_key: VerifyKey) -> Optional[SyftUser]:
        return self.first(
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
        )

    def __login_validation(self, email: str, password: str) -> SyftUser:
        try:
            user = self.first(email=email)
            if not user:
                raise UserNotFoundError

            hashed = user.hashed_password.encode("UTF-8")
            salt = user.salt.encode("UTF-8")
            bytes_pass = password.encode("UTF-8")

            if checkpw(bytes_pass, salt + hashed):
                return user
            else:
                raise InvalidCredentialsError
        except UserNotFoundError:
            raise InvalidCredentialsError

    def __salt_and_hash_password(self, password: str, rounds: int) -> Tuple[str, str]:
        bytes_pass = password.encode("UTF-8")
        salt = gensalt(rounds=rounds)
        salt_len = len(salt)
        hashed = hashpw(bytes_pass, salt)
        hashed = hashed[salt_len:]
        hashed = hashed.decode("UTF-8")
        salt = salt.decode("UTF-8")
        return salt, hashed
