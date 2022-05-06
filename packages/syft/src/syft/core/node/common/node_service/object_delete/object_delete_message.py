# stdlib
from typing import List
from typing import Optional
from typing import Type

# third party
from nacl.signing import VerifyKey
from typing_extensions import final

# relative
from .....common.serde.serializable import serializable
from .....common.uid import UID
from ....domain_interface import DomainInterface
from ....domain_msg_registry import DomainMessageRegistry
from ....registry import VMMessageRegistry
from ...permissions.permissions import BasePermission
from ...permissions.user_permissions import NoRestriction
from ..generic_payload.syft_message import NewSyftMessage as SyftMessage
from ..generic_payload.syft_message import ReplyPayload
from ..generic_payload.syft_message import RequestPayload


@serializable(recursive_serde=True)
@final
class ObjectDeleteMessage(SyftMessage, DomainMessageRegistry, VMMessageRegistry):

    # Pydantic Inner class to define expected request payload fields.
    class Request(RequestPayload):
        """Payload fields and types used during a User Creation Request."""

        id_at_location: str

    # Pydantic Inner class to define expected reply payload fields.
    class Reply(ReplyPayload):
        """Payload fields and types used during a User Creation Response."""

        message: str = "Deleted Successfully."

    request_payload_type = (
        Request  # Converts generic syft dict into a ObjectDeleteMessage Request object.
    )
    reply_payload_type = (
        Reply  # Creates a proper Reply payload message structure as a response.
    )

    def run(  # type: ignore
        self, node: DomainInterface, verify_key: Optional[VerifyKey] = None
    ) -> ReplyPayload:  # type: ignore

        # TODO: We can have it run async and have a cron/periodic task to clean up failed deletes.

        # relative
        from ......logger import critical
        from ......logger import debug

        try:
            debug(
                f"Calling delete on Object with ID {self.payload.id_at_location} in store."  # ignore
            )
            id_at_location = UID.from_string(self.payload.id_at_location)
            if not node.store.is_dataset(key=id_at_location):  # type: ignore
                node.store.delete(key=id_at_location)
        except Exception as e:
            log = f"> ObjectDeleteMessage delete exception {self.payload.id_at_location} {e}"
            critical(log)

        return ObjectDeleteMessage.Reply()

    def get_permissions(self) -> List[Type[BasePermission]]:
        """Returns the list of permission classes."""
        # Needs to be replaced with UserHasWritePermissionToData once who has permissions to delete is sorted
        return [NoRestriction]
