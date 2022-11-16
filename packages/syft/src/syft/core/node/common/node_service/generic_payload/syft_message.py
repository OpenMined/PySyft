# stdlib

# stdlib
from typing import Any
from typing import Dict
from typing import KeysView
from typing import List
from typing import Optional

# third party
from nacl.signing import VerifyKey
from pydantic import BaseModel
from pydantic.error_wrappers import ValidationError as PydanticValidationError

# relative
from .....common.message import ImmediateSyftMessage
from .....common.message import SignedMessage
from .....common.uid import UID
from .....io.address import Address
from ....abstract.node_service_interface import NodeServiceInterface
from ....common.exceptions import AuthorizationError
from ....common.exceptions import BadPayloadException
from ....common.exceptions import PermissionsNotDefined


# Inner Payload message using Pydantic.
class Payload(BaseModel):
    # allows splatting with **payload
    def keys(self) -> KeysView[str]:
        return self.__dict__.keys()

    # allows splatting with **payload
    def __getitem__(self, key: str) -> Any:
        return self.__dict__.__getitem__(key)

    class Config:
        orm_mode = True


class RequestPayload(Payload):
    pass


class ReplyPayload(Payload):
    pass


class NewSyftMessage(ImmediateSyftMessage):
    """A base class from which all message classes should inherit.

    Note:
        This will eventually replace the old `SyftMessage` class.
    """

    __attr_allowlist__ = ["id", "address", "reply_to", "reply", "msg_id", "kwargs"]

    signed_type = SignedMessage
    request_payload_type = RequestPayload
    reply_payload_type = ReplyPayload

    def __init__(
        self,
        address: Address,
        kwargs: Optional[Dict[str, Any]] = None,
        msg_id: Optional[UID] = None,
        reply_to: Optional[Address] = None,
        reply: bool = False,
    ) -> None:
        super().__init__(address=address, msg_id=msg_id)
        self.reply_to = reply_to
        self.reply = reply
        self.kwargs = kwargs if kwargs else {}

    @property
    def payload(self) -> Payload:
        kwargs_dict = {}

        if hasattr(self.kwargs, "upcast"):
            kwargs_dict = self.kwargs.upcast()  # type: ignore
        else:
            kwargs_dict = self.kwargs  # type: ignore

        try:
            # If it's not a reply message then load kwargs as a proper request payload.
            if not self.reply:
                return self.request_payload_type(**kwargs_dict)
            # If it's a reply message, then load kwargs as a proper reply payload.
            else:
                return self.reply_payload_type(**kwargs_dict)
        except PydanticValidationError:
            raise BadPayloadException

    def run(
        self, node: NodeServiceInterface, verify_key: Optional[VerifyKey] = None
    ) -> ReplyPayload:
        raise NotImplementedError

    def get_permissions(self) -> List:
        """Returns the list of permission classes applicable to the given message."""
        raise NotImplementedError

    def check_permissions(
        self, node: NodeServiceInterface, verify_key: Optional[VerifyKey] = None
    ) -> None:
        """Check if the user has relevant permissions to run this message.

        Args:
            node (NodeServiceInterface): node interface used to invoke this message.
            verify_key (Optional[VerifyKey], optional): user signed verification key. Defaults to None.

        Raises:
            AuthorizationError: Error when one of the permission is denied.
        """

        permissions = []
        if len(getattr(self, "permissions", [])):
            permissions = getattr(self, "permissions")
        elif len(self.get_permissions()):
            permissions = self.get_permissions()

        if not len(permissions):
            raise PermissionsNotDefined

        for permission_class in permissions:
            if not permission_class().has_permission(
                msg=self, node=node, verify_key=verify_key
            ):
                raise AuthorizationError(
                    f"You don't have access to perform {self} action."
                )
