# stdlib
from typing import List
from typing import Optional
from typing import Type
from typing import Union

# third party
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
from pydantic import EmailStr
from typing_extensions import final

# relative
from .....common.serde.serializable import serializable
from ....abstract.node_service_interface import NodeServiceInterface
from ....domain.domain_interface import DomainInterface
from ....domain.registry import DomainMessageRegistry
from ...exceptions import AuthorizationError
from ...exceptions import MissingRequestKeyError
from ...exceptions import UserNotFoundError
from ...node_table.utils import model_to_json
from ...permissions.permissions import BasePermission
from ...permissions.permissions import BinaryOperation
from ...permissions.permissions import UnaryOperation
from ...permissions.user_permissions import IsNodeDaaEnabled
from ...permissions.user_permissions import UserCanCreateUsers
from ...permissions.user_permissions import UserCanTriageRequest
from ...permissions.user_permissions import UserIsOwner
from ..generic_payload.syft_message import NewSyftMessage as SyftMessage
from ..generic_payload.syft_message import ReplyPayload
from ..generic_payload.syft_message import RequestPayload


@serializable(recursive_serde=True)
@final
class CreateUserMessage(SyftMessage, DomainMessageRegistry):

    # Pydantic Inner class to define expected request payload fields.
    class Request(RequestPayload):
        """Payload fields and types used during a User Creation Request."""

        email: EmailStr
        password: str
        name: str
        role: Optional[str] = "Data Scientist"
        institution: Optional[str]
        website: Optional[str]
        budget: Optional[float] = 0.0
        daa_pdf: Optional[bytes] = b""

    # Pydantic Inner class to define expected reply payload fields.
    class Reply(ReplyPayload):
        """Payload fields and types used during a User Creation Response."""

        message: str = "User created successfully!"

    request_payload_type = (
        Request  # Converts generic syft dict into a CreateUserMessage.Request object.
    )
    reply_payload_type = (
        Reply  # Creates a proper Reply payload message structure as a response.
    )

    def run(  # type: ignore
        self, node: DomainInterface, verify_key: Optional[VerifyKey] = None
    ) -> ReplyPayload:  # type: ignore
        """Validates the request parameters and creates a new user.

        Args:
            node (DomainInterface): Domain interface node.
            verify_key (Optional[VerifyKey], optional): User signed verification key. Defaults to None.

        Raises:
            MissingRequestKeyError: If the required request fields are missing.
            AuthorizationError: If user already exists for given email address.

        Returns:
            ReplyPayload: Message on successful user creation.
        """

        # Check if this email was already registered
        try:
            node.users.first(email=self.payload.email)
            # If the email has already been registered, raise exception
            raise AuthorizationError(
                message="You can't create a new User using this email!"
            )
        except UserNotFoundError:
            # If email not registered, a new user can be created.
            pass

        app_id = node.users.create_user_application(
            name=self.payload.name,
            email=self.payload.email,
            password=self.payload.password,
            daa_pdf=self.payload.daa_pdf,
            institution=self.payload.institution,
            website=self.payload.website,
            budget=self.payload.budget,
        )

        node.users.process_user_application(
            candidate_id=app_id, status="accepted", verify_key=verify_key
        )

        return CreateUserMessage.Reply()

    def get_permissions(self) -> List[Type[BasePermission]]:
        """Returns the list of permission classes."""
        return [UserCanCreateUsers, IsNodeDaaEnabled]


@serializable(recursive_serde=True)
@final
class GetUserMessage(SyftMessage, DomainMessageRegistry):

    # Pydantic Inner class to define expected request payload fields.
    class Request(RequestPayload):
        user_id: int

    # Pydantic Inner class to define expected reply payload fields.
    class Reply(ReplyPayload):
        id: int
        name: str
        email: str
        role: Union[int, str]
        budget: float
        created_at: str
        budget_spent: Optional[float] = 0.0
        institution: Optional[str] = ""
        website: Optional[str] = ""
        added_by: Optional[str] = ""

    request_payload_type = Request
    reply_payload_type = Reply

    def run(
        self, node: NodeServiceInterface, verify_key: Optional[VerifyKey] = None
    ) -> ReplyPayload:
        """Returns the user for given user id.

        Args:
            node (NodeServiceInterface): domain/network interface node.
            verify_key (Optional[VerifyKey], optional): user signed verification key. Defaults to None.

        Returns:
            ReplyPayload: Details of the user.
        """
        # Retrieve User Model
        user = node.users.first(id=self.payload.user_id)  # type: ignore

        # Build Reply
        reply = GetUserMessage.Reply(**model_to_json(user))

        # Use role name instead of role ID.
        reply.role = node.roles.first(id=reply.role).name  # type: ignore

        # Get budget spent
        reply.budget_spent = node.acc.user_budget(  # type: ignore
            user_key=VerifyKey(user.verify_key.encode("utf-8"), encoder=HexEncoder)
        )
        return reply

    def get_permissions(self) -> List[Type[BasePermission]]:
        """Returns the list of permission classes."""
        return [UserCanTriageRequest]


@serializable(recursive_serde=True)
@final
class GetUsersMessage(SyftMessage, DomainMessageRegistry):

    # Pydantic Inner class to define expected request payload fields.
    class Request(RequestPayload):
        pass

    # Pydantic Inner class to define expected reply payload fields.
    class Reply(ReplyPayload):
        users: List[GetUserMessage.Reply] = []

    request_payload_type = Request
    reply_payload_type = Reply

    def run(  # type: ignore
        self, node: NodeServiceInterface, verify_key: Optional[VerifyKey] = None
    ) -> ReplyPayload:
        """Returns the list of all users registered to the node.

        Args:
            node (NodeServiceInterface): domain or network interface node.
            verify_key (Optional[VerifyKey], optional): user signed verification key. Defaults to None.

        Returns:
            ReplyPayload: details of all users registered to the node as a list.
        """
        # Get All Users
        users = node.users.all()
        users_list = list()
        for user in users:
            user_model = GetUserMessage.Reply(**model_to_json(user))

            # Use role name instead of role ID.
            user_model.role = node.roles.first(id=user_model.role).name

            # Remaining Budget
            # TODO:
            # Rename it from budget_spent to remaining budget
            user_model.budget_spent = node.acc.get_remaining_budget(  # type: ignore
                user_key=VerifyKey(user.verify_key.encode("utf-8"), encoder=HexEncoder),
                returned_epsilon_is_private=False,
            )
            users_list.append(user_model)

        reply = GetUsersMessage.Reply(users=users_list)
        return reply

    def get_permissions(self) -> List:
        return [UserCanTriageRequest]


@serializable(recursive_serde=True)
@final
class DeleteUserMessage(SyftMessage, DomainMessageRegistry):

    # Pydantic Inner class to define expected request payload fields.
    class Request(RequestPayload):
        user_id: int

    # Pydantic Inner class to define expected reply payload fields.
    class Reply(ReplyPayload):
        message: str = "User deleted successfully!"

    request_payload_type = Request
    reply_payload_type = Reply

    def run(  # type: ignore
        self, node: NodeServiceInterface, verify_key: Optional[VerifyKey] = None
    ) -> ReplyPayload:
        """Delete a user with given user id.

        Args:
            node (NodeServiceInterface): domain or network node interface.
            verify_key (Optional[VerifyKey], optional): user signed verification key. Defaults to None.

        Returns:
            ReplyPayload: message on successful user deletion.
        """

        node.users.delete(id=self.payload.user_id)

        return DeleteUserMessage.Reply()

    def get_permissions(self) -> List[Union[Type[BasePermission], UnaryOperation]]:
        """Returns the list of permission classes applicable to this message."""
        return [UserCanCreateUsers, ~UserIsOwner]


@serializable(recursive_serde=True)
@final
class UpdateUserMessage(SyftMessage, DomainMessageRegistry):

    # Pydantic Inner class to define expected request payload fields.
    class Request(RequestPayload):
        user_id: int
        name: Optional[str] = ""
        email: Optional[EmailStr] = None
        institution: Optional[str] = ""
        website: Optional[str] = ""
        password: Optional[str] = ""
        role: Optional[str] = ""
        budget: Optional[float] = None

    # Pydantic Inner class to define expected reply payload fields.
    class Reply(ReplyPayload):
        message: str = "User updated successfully!"

    request_payload_type = Request
    reply_payload_type = Reply

    def run(  # type: ignore
        self, node: NodeServiceInterface, verify_key: Optional[VerifyKey] = None
    ) -> ReplyPayload:
        """Updates the information for the given user id.

        Args:
            node (NodeServiceInterface): domain or network node interface.
            verify_key (Optional[VerifyKey], optional): user signed verification key. Defaults to None.

        Raises:
            MissingRequestKeyError: If the request parameters are not valid.
            UserNotFoundError: If user with given user id is not present in the database.
            AuthorizationError: If user does not have permission to update their role.

        Returns:
            ReplyPayload: success message on updating the user information.
        """

        _valid_parameters = (
            self.payload.email
            or self.payload.password
            or self.payload.role
            or self.payload.name
            or self.payload.institution
            or self.payload.website
        )

        # Change own information
        _valid_user = node.users.contain(id=self.payload.user_id)

        if not _valid_parameters:
            raise MissingRequestKeyError(
                "Missing json fields ( email,password,role,groups, name )"
            )

        if not _valid_user:
            raise UserNotFoundError

        payload_dict = self.payload.dict(exclude_unset=True)
        user_id = payload_dict.pop("user_id")

        # If Change Role Request, then check if user
        # has proper permissions.
        # TODO: This can also be simplified further.
        if self.payload.role:  # type: ignore
            target_user = node.users.first(id=user_id)
            _allowed = (
                self.payload.role != node.roles.owner_role.name  # Target Role != Owner
                and target_user.role
                != node.roles.owner_role.id  # Target User Role != Owner
                and node.users.can_create_users(
                    verify_key=verify_key
                )  # Key Permissions
            )

            # If all premises were respected
            if _allowed:
                role = payload_dict.pop("role")
                new_role_id = node.roles.first(name=role).id
                node.users.set(user_id=user_id, role=new_role_id)  # type: ignore
            elif (  # Transfering Owner's role
                self.payload.role == node.roles.owner_role.name  # target role == Owner
                and node.users.role(verify_key=verify_key).name
                == node.roles.owner_role.name  # Current user is the current node owner.
            ):
                role = payload_dict.pop("role")
                new_role_id = node.roles.first(name=role).id
                node.users.set(user_id=str(user_id), role=new_role_id)
                current_user = node.users.get_user(verify_key=verify_key)
                node.users.set(user_id=current_user.id, role=node.roles.admin_role.id)  # type: ignore
                # Updating current node keys
                root_key = SigningKey(
                    current_user.private_key.encode("utf-8"), encoder=HexEncoder  # type: ignore
                )
                node.signing_key = root_key
                node.verify_key = root_key.verify_key
                # IDK why, but we also have a different var (node.root_verify_key)
                # defined at ...common.node.py that points to the verify_key.
                # So we need to update it as well.
                node.root_verify_key = root_key.verify_key
            elif target_user.role == node.roles.owner_role.id:
                raise AuthorizationError(
                    "You're not allowed to change Owner user roles!"
                )
            else:
                raise AuthorizationError("You're not allowed to change User roles!")

        # Update values of all other parameters
        for param, val in payload_dict.items():
            update_dict = {"user_id": user_id, param: val}
            node.users.set(**update_dict)

        return UpdateUserMessage.Reply()

    def get_permissions(self) -> List[Union[Type[BasePermission], BinaryOperation]]:
        """Returns the list of permission classes applicable to this message."""
        return [UserCanCreateUsers | UserIsOwner]
