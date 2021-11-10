# stdlib
from datetime import datetime
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
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
from pydantic import BaseModel
from pydantic import EmailStr
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Query
from sqlalchemy.orm import sessionmaker

# relative
from ..exceptions import InvalidCredentialsError
from ..exceptions import UserNotFoundError
from ..node_table.pdf import PDFObject
from ..node_table.roles import Role
from ..node_table.user import SyftUser
from ..node_table.user import UserApplication
from .database_manager import DatabaseManager
from .role_manager import RoleManager


# Shared properties
class UserBase(BaseModel):
    email: Optional[EmailStr] = None
    is_active: Optional[bool] = True
    is_superuser: bool = False
    full_name: Optional[str] = None


# Properties to receive via API on creation
class UserCreate(UserBase):
    email: EmailStr
    password: str


# Properties to receive via API on update
class UserUpdate(UserBase):
    password: Optional[str] = None


class UserInDBBase(UserBase):
    id: Optional[int] = None

    class Config:
        orm_mode = True


# Additional properties to return via API
class User(UserInDBBase):
    pass


# Additional properties stored in DB
class UserInDB(UserInDBBase):
    hashed_password: str


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

    def create_user_application(
        self,
        name: str,
        email: str,
        password: str,
        daa_pdf: Optional[bytes],
        institution: Optional[str] = "",
        website: Optional[str] = "",
        budget: Optional[float] = 0.0,
    ) -> int:
        salt, hashed = self.__salt_and_hash_password(password, 12)
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
        _pdf_obj = PDFObject(binary=daa_pdf)
        session_local.add(_pdf_obj)
        session_local.commit()
        session_local.flush()
        session_local.refresh(_pdf_obj)

        _obj = UserApplication(
            name=name,
            email=email,
            salt=salt,
            hashed_password=hashed,
            daa_pdf=_pdf_obj.id,
            institution=institution,
            website=website,
            budget=budget,
        )
        session_local.add(_pdf_obj)
        session_local.add(_obj)
        session_local.commit()
        _obj_id = _obj.id
        session_local.close()
        return _obj_id

    def get_all_applicant(self) -> List[UserApplication]:
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
        result = list(session_local.query(UserApplication).all())
        session_local.close()
        return result

    def process_user_application(
        self, candidate_id: int, status: str, verify_key: VerifyKey
    ) -> None:
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
        candidate = (
            session_local.query(UserApplication).filter_by(id=candidate_id).first()
        )
        session_local.close()

        if status == "accepted":
            # Generate a new signing key
            _private_key = SigningKey.generate()

            encoded_pk = _private_key.encode(encoder=HexEncoder).decode("utf-8")
            encoded_vk = _private_key.verify_key.encode(encoder=HexEncoder).decode(
                "utf-8"
            )
            added_by = self.get_user(verify_key).name  # type: ignore
            self.register(
                name=candidate.name,
                email=candidate.email,
                role=self.roles.ds_role.id,
                budget=candidate.budget,
                private_key=encoded_pk,
                verify_key=encoded_vk,
                hashed_password=candidate.hashed_password,
                salt=candidate.salt,
                daa_pdf=candidate.daa_pdf,
                added_by=added_by,
                institution=candidate.institution,
                website=candidate.website,
                created_at=datetime.now(),
            )
        else:
            status = "rejected"

        session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
        candidate = (
            session_local.query(UserApplication).filter_by(id=candidate_id).first()
        )
        candidate.status = status
        session_local.flush()
        session_local.commit()
        session_local.close()

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
            created_at=datetime.now(),
        )

    def query(self, **kwargs: Any) -> Query:
        results = super().query(**kwargs)
        return results

    def first(self, **kwargs: Any) -> SyftUser:
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
        website: str = "",
        institution: str = "",
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
        elif website:
            key = "website"
            value = website
        elif budget:
            key = "budget"
            value = budget
        elif institution:
            key = "institution"
            value = institution
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
            return self.role(verify_key=verify_key).can_triage_data_requests
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
