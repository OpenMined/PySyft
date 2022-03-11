# stdlib
from typing import List
from typing import Dict
from typing import Any
from typing import Optional
from typing import Type
from typing import Union
import math

# third party
from nacl.signing import VerifyKey
from pydantic import EmailStr
from typing_extensions import final

# relative
from .....common.serde.serializable import serializable
from ....domain.domain_interface import DomainInterface
from ....domain.registry import DomainMessageRegistry
from ...node_table.utils import model_to_json
from ...permissions.permissions import BasePermission
from ...permissions.user_permissions import NoRestriction
from ..generic_payload.syft_message import NewSyftMessage as SyftMessage
from ..generic_payload.syft_message import ReplyPayload
from ..generic_payload.syft_message import RequestPayload
from .....store.util import get_s3_client

@serializable(recursive_serde=True)
@final
class UploadDataMessage(SyftMessage, DomainMessageRegistry):

    # Pydantic Inner class to define expected request payload fields.
    class Request(RequestPayload):
        """Payload fields and types used during a User Creation Request."""
        filename: str
        file_size: int
        chunk_size: int
        content_type: Optional[str]

    # Pydantic Inner class to define expected reply payload fields.
    class Reply(ReplyPayload):
        """Payload fields and types used during a User Creation Response."""
        upload_id: str
        parts: List[Dict[Any,Any]]

    request_payload_type = (
        Request  # Converts generic syft dict into a CreateUserMessage.Request object.
    )
    reply_payload_type = (
        Reply  # Creates a proper Reply payload message structure as a response.
    )

    def run(  # type: ignore
        self, node: DomainInterface, verify_key: Optional[VerifyKey] = None
    ) -> ReplyPayload:  # type: ignore
        # TODO : Move to permissions
        #if not node.users.can_upload_data(verify_key=verify_key):
        #    return {"message": "You're not authorized to do this."}
        key = f"{self.payload.filename}"
        s3_client = get_s3_client(docker_host=True)
        result = s3_client.create_multipart_upload(Bucket=node.name, Key=key)
        total_parts = math.ceil(self.payload.file_size / self.payload.chunk_size)
        upload_id = result["UploadId"]
        signed_urls = []
        s3_client = get_s3_client()

        parts = list()
        for part_no in range(1, total_parts + 1):
            # Creating presigned urls
            signed_url = s3_client.generate_presigned_url(
                ClientMethod="upload_part",
                Params={
                    "Bucket": node.name,
                    "Key": key,
                    "UploadId": upload_id,
                    "PartNumber": part_no,
                },
                ExpiresIn=1800,  # in seconds
            )
            parts.append({"part_no": part_no, "url": signed_url})
        
        return UploadDataMessage.Reply(upload_id=upload_id, parts=parts)

    def get_permissions(self) -> List[Type[BasePermission]]:
        """Returns the list of permission classes."""
        return [NoRestriction]





@serializable(recursive_serde=True)
@final
class UploadDataCompleteMessage(SyftMessage, DomainMessageRegistry):

    # Pydantic Inner class to define expected request payload fields.
    class Request(RequestPayload):
        """Payload fields and types used during a User Creation Request."""
        filename: str
        upload_id: str
        parts: List[Dict[Any,Any]]

    # Pydantic Inner class to define expected reply payload fields.
    class Reply(ReplyPayload):
        """Payload fields and types used during a User Creation Response."""
        message: str = "Upload Complete!"

    request_payload_type = (
        Request  # Converts generic syft dict into a CreateUserMessage.Request object.
    )
    reply_payload_type = (
        Reply  # Creates a proper Reply payload message structure as a response.
    )

    def run(  # type: ignore
        self, node: DomainInterface, verify_key: Optional[VerifyKey] = None
    ) -> ReplyPayload:  # type: ignore
        
        # TODO: Move to permissions
        #user_role = node.roles.first(**{"id": current_user.role})
        #if not user_role.can_upload_data:
        #    return {"message": "You're not authorized to do this."}

        key = f"{self.payload.filename}"
        result = get_s3_client(docker_host=True).complete_multipart_upload(
            Bucket=node.name,
            Key=key,
            MultipartUpload={"Parts": self.payload.parts},
            UploadId=self.payload.upload_id,
        )
        return UploadDataCompleteMessage.Reply()

    def get_permissions(self) -> List[Type[BasePermission]]:
        """Returns the list of permission classes."""
        return [NoRestriction]
