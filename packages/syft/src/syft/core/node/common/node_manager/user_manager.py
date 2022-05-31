"""This file defines classes and methods which are used to manage database queries on the SyftUser table."""

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
from .constants import UserApplicationStatus
from .database_manager import DatabaseManager
from .role_manager import RoleManager


class RefreshBudgetException(Exception):
    pass


class NegativeBudgetException(Exception):
    pass


class NotEnoughBudgetException(Exception):
    pass


class UserManager(DatabaseManager):
    """Class to manage user database actions."""

    schema = SyftUser

    def __init__(self, database: Engine) -> None:
        super().__init__(schema=UserManager.schema, db=database)
        self.roles = RoleManager(database)

    @property
    def common_users(self) -> list:
        """Return users having the common role access."""
        common_users: List[SyftUser] = []
        for role in self.roles.common_roles:
            common_users = common_users + list(super().query(role=role.id))

        return common_users

    @property
    def org_users(self) -> list:
        """Return all the users in the organization."""
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
        """Stores the information of the application submitted by the user.

        Args:
            name (str): name of the user.
            email (str): email of the user.
            password (str): password of the user.
            daa_pdf (Optional[bytes]): data access agreement.
            institution (Optional[str], optional): name of the institution to which the user belongs. Defaults to "".
            website (Optional[str], optional): website link of the institution. Defaults to "".
            budget (Optional[float], optional): privacy budget allocated to the user. Defaults to 0.0.

        Returns:
            int: Id of the application of the user.
        """

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
        """Returns the application data of all the applicants in the database.

        Returns:
            List[UserApplication]: All user applications.
        """
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
        result = list(session_local.query(UserApplication).all())
        session_local.close()
        return result

    def process_user_application(
        self, candidate_id: int, status: str, verify_key: VerifyKey
    ) -> None:
        """Process the application for the given candidate.

        If the application of the user was accepted, then register the user
        and its details in the database. Finally update the application status
        for the given user/candidate in the database.

        Args:
            candidate_id (int): user id of the candidate.
            status (str): application status.
            verify_key (VerifyKey): public digital signature of the user.
        """
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
        candidate = (
            session_local.query(UserApplication).filter_by(id=candidate_id).first()
        )
        session_local.close()

        if (
            status == UserApplicationStatus.ACCEPTED.value
        ):  # If application was accepted
            # Generate a new signing key
            _private_key = SigningKey.generate()

            encoded_pk = _private_key.encode(encoder=HexEncoder).decode("utf-8")
            encoded_vk = _private_key.verify_key.encode(encoder=HexEncoder).decode(
                "utf-8"
            )
            added_by = self.get_user(verify_key).name  # type: ignore

            # Register the user in the database
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
            status = UserApplicationStatus.REJECTED.value

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
        """Registers a user in the database, when they signup on a domain.

        Args:
            name (str): name of the user.
            email (str): email of the user.
            password (str): password set by the user.
            budget (float): privacy budget alloted to the user.
            role (int): role of the user when they signup on the domain.
            private_key (str): private digital signature of the user.
            verify_key (str): public digital signature of the user.

        Returns:
            SyftUser: the registered user object.
        """
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
        """Returns the user object for the given the email and password.

        Args:
            email (str): email of the user.
            password (str): password of the user.

        Returns:
            SyftUser: user object for the given email and password.
        """
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
        """Updates the information for the given user id.

        Args:
            user_id (str): unique id of the user in the database.
            email (str, optional): email of the user. Defaults to "".
            password (str, optional): password of the user. Defaults to "".
            role (int, optional): role of the user. Defaults to 0.
            name (str, optional): name of the user. Defaults to "".
            website (str, optional): website of the institution of the user. Defaults to "".
            institution (str, optional): name of the institution of the user. Defaults to "".
            budget (float, optional): privacy budget allocated to the user. Defaults to 0.0.

        Raises:
            UserNotFoundError: Raised when a user does not exits for the given user id.
            Exception: Raised when an invalid argument/property is passed.
        """
        if not self.contain(id=user_id):
            raise UserNotFoundError

        key: str
        value: Union[str, int, float]

        if email:
            key = "email"
            value = email
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

    def change_password(self, user_id: str, current_pwd: str, new_pwd: str) -> None:
        user = self.first(id=user_id)

        hashed = user.hashed_password.encode("UTF-8")
        salt = user.salt.encode("UTF-8")
        bytes_pass = current_pwd.encode("UTF-8")

        if checkpw(bytes_pass, salt + hashed):
            salt, hashed = self.__salt_and_hash_password(new_pwd, 12)
            self.modify({"id": user_id}, {"salt": salt, "hashed_password": hashed})
        else:
            # Should it warn the user about his wrong current password input?
            raise InvalidCredentialsError

    def can_create_users(self, verify_key: VerifyKey) -> bool:
        """Checks if a user has permissions to create new users."""
        try:
            return self.role(verify_key=verify_key).can_create_users
        except UserNotFoundError:
            return False

    def can_upload_data(self, verify_key: VerifyKey) -> bool:
        """Checks if a user has permissions to upload data to the node."""
        try:
            return self.role(verify_key=verify_key).can_upload_data
        except UserNotFoundError:
            return False

    def can_triage_requests(self, verify_key: VerifyKey) -> bool:
        """Checks if a user has permissions to triage requests."""
        try:
            return self.role(verify_key=verify_key).can_triage_data_requests
        except UserNotFoundError:
            return False

    def can_manage_infrastructure(self, verify_key: VerifyKey) -> bool:
        """Checks if a user has permissions to manage the deployed infrastructure."""
        try:
            return self.role(verify_key=verify_key).can_manage_infrastructure
        except UserNotFoundError:
            return False

    def can_edit_roles(self, verify_key: VerifyKey) -> bool:
        """Checks if a user has permission to edit roles of other users."""
        try:
            return self.role(verify_key=verify_key).can_edit_roles
        except UserNotFoundError:
            return False

    def role(self, verify_key: VerifyKey) -> Role:
        """Returns the role of the given user."""
        user = self.get_user(verify_key)
        if not user:
            raise UserNotFoundError
        return self.roles.first(id=user.role)

    def get_user(self, verify_key: VerifyKey) -> Optional[SyftUser]:
        """Returns the user for the given public digital signature."""
        return self.first(
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
        )

    def __login_validation(self, email: str, password: str) -> SyftUser:
        """Validates and returns the user object for the given credentials.

        Args:
            email (str): email of the user.
            password (str): password of the user.

        Raises:
            UserNotFoundError: Raised if the user does not exist for the email.
            InvalidCredentialsError: Raised if either the password or email is incorrect.

        Returns:
            SyftUser: Returns the user for the given credentials.
        """
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

    def get_budget_for_user(self, verify_key: VerifyKey) -> float:
        encoded_vk = verify_key.encode(encoder=HexEncoder).decode("utf-8")
        user = self.first(verify_key=encoded_vk)
        if user is None:
            raise Exception(f"No user for verify_key: {verify_key}")
        return user.budget

    def deduct_epsilon_for_user(
        self, verify_key: VerifyKey, old_budget: float, epsilon_spend: float
    ) -> bool:
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.db)()
        with session_local.begin():
            encoded_vk = verify_key.encode(encoder=HexEncoder).decode("utf-8")
            user = self.first(verify_key=encoded_vk)
            if user.budget != old_budget:
                raise RefreshBudgetException(
                    "The budget used does not match the current budget in the database."
                    + "Try refreshing again"
                )
            if user.budget < 0:
                raise NegativeBudgetException(
                    f"Budget is somehow negative {user.budget}"
                )

            if old_budget - abs(epsilon_spend) < 0:
                raise NotEnoughBudgetException(
                    f"The user does not have enough budget: {user.budget} for epsilon spend: {epsilon_spend}"
                )

            user.budget = user.budget - epsilon_spend
            session_local.add(user)

        session_local.commit()
        session_local.close()
        return True
