# stdlib
import math
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type

# third party
import boto3
from nacl.signing import VerifyKey
from typing_extensions import final

# relative
from ......grid import GridURL
from .....common.serde.serializable import serializable
from .....common.uid import UID
from ....domain_interface import DomainInterface
from ....domain_msg_registry import DomainMessageRegistry
from ...permissions.permissions import BasePermission
from ...permissions.user_permissions import NoRestriction
from ...util import get_s3_client
from ..generic_payload.syft_message import NewSyftMessage as SyftMessage
from ..generic_payload.syft_message import ReplyPayload
from ..generic_payload.syft_message import RequestPayload


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
        parts: List[Dict[Any, Any]]

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
        # if not node.users.can_upload_data(verify_key=verify_key):
        #    return {"message": "You're not authorized to do this."}
        key = f"{self.payload.filename}"

        # If we're saving the new object using UID keys as its asset name
        # Then we need to check if this UID was registered previously.
        if UID.is_valid_uuid(key.split("/")[-1]):
            id_at_location = UID.from_string(key.split("/")[-1])  # Get Object ID.
            # If if there's another object with the same ID.
            node.store.check_collision(id_at_location)

        s3_client = get_s3_client(settings=node.settings)
        result = s3_client.create_multipart_upload(Bucket=node.id.no_dash, Key=key)
        total_parts = math.ceil(self.payload.file_size / self.payload.chunk_size)
        upload_id = result["UploadId"]

        parts = list()
        for part_no in range(1, total_parts + 1):
            # Creating presigned urls
            signed_url = s3_client.generate_presigned_url(
                ClientMethod="upload_part",
                Params={
                    "Bucket": node.id.no_dash,
                    "Key": key,
                    "UploadId": upload_id,
                    "PartNumber": part_no,
                },
                ExpiresIn=node.settings.S3_PRESIGNED_TIMEOUT_SECS,
            )
            # parse as a URL
            grid_url = GridURL.from_url(url=signed_url)
            # add /blob to path
            grid_url.path = f"/blob{grid_url.path}"
            # only return the path and let the client use its existing public url
            parts.append({"part_no": part_no, "url": grid_url.url_path})

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
        parts: List[Dict[Any, Any]]

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
        # user_role = node.roles.first(**{"id": current_user.role})
        # if not user_role.can_upload_data:
        #    return {"message": "You're not authorized to do this."}

        key = f"{self.payload.filename}"
        client: boto3.client.S3 = get_s3_client(settings=node.settings)
        _ = client.complete_multipart_upload(
            Bucket=node.id.no_dash,
            Key=key,
            MultipartUpload={"Parts": self.payload.parts},
            UploadId=self.payload.upload_id,
        )
        return UploadDataCompleteMessage.Reply()

    def get_permissions(self) -> List[Type[BasePermission]]:
        """Returns the list of permission classes."""
        return [NoRestriction]


@serializable(recursive_serde=True)
@final
class AbortDataUploadMessage(SyftMessage, DomainMessageRegistry):

    # Pydantic Inner class to define expected request payload fields.
    class Request(RequestPayload):
        """Payload fields and types used during Deletion of Incomplete Data Upload Request."""

        upload_id: str
        asset_name: str

    # Pydantic Inner class to define expected reply payload fields.
    class Reply(ReplyPayload):
        """Payload fields and types used during a Deletion of Incomplete Data Upload Response."""

        message: str = "Deletion Complete!"

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
        # user_role = node.roles.first(**{"id": current_user.role})
        # if not user_role.can_upload_data:
        #    return {"message": "You're not authorized to do this."}

        client: boto3.client.S3 = get_s3_client(settings=node.settings)

        # Abort multipart upload
        client.abort_multipart_upload(
            UploadId=self.payload.upload_id,
            Key=self.payload.asset_name,
            Bucket=node.id.no_dash,
        )
        return UploadDataCompleteMessage.Reply()

    def get_permissions(self) -> List[Type[BasePermission]]:
        """Returns the list of permission classes."""
        return [NoRestriction]
