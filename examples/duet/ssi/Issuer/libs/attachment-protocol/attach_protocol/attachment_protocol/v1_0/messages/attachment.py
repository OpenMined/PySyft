""" Protocol for attachment message"""

from ..message_types import PROTOCOL_PACKAGE, ATTACHMENT

import uuid

from typing import Sequence

from aries_cloudagent.wallet.util import bytes_to_b64
from aries_cloudagent.messaging.decorators.attach_decorator import (
    AttachDecorator,
    AttachDecoratorSchema,
    AttachDecoratorData
)

from marshmallow import fields

from aries_cloudagent.messaging.agent_message import AgentMessage, AgentMessageSchema

HANDLER_CLASS = f"{PROTOCOL_PACKAGE}.handlers.attachment_handler.AttachmentHandler"

class Attachment(AgentMessage):
    #Class representing message attachment

    class Meta:
        handler_class = HANDLER_CLASS
        message_type = ATTACHMENT
        schema_class = "AttachmentSchema"

    def __init__(

        self,
        *,
        message: str = None,
        localization: str = None,
        files_attach: Sequence[AttachDecorator] = None,
        **kwargs,
    ):
        super(Attachment, self).__init__(**kwargs)
        self.message = message
        self.files_attach = list(files_attach) if files_attach else []
        if localization:
            self._decorators["l10n"] = localization

    @classmethod
    def wrap_file(cls, content, filename, mime_type, description=None) -> AttachDecorator:
        return AttachDecorator(
            ident=str(uuid.uuid4()),
            filename=filename,
            mime_type=mime_type,
            data=AttachDecoratorData(
                base64_=bytes_to_b64(content)
            ),
            description=description

        )

        
class AttachmentSchema(AgentMessageSchema):
    #Schema for Attachment protocol class

    class Meta:
        #Metadata

        model_class = Attachment
    
    message = fields.Str(required=True, description="A message about the file", example="Hey, check out this image")
    files_attach = fields.Nested(
        AttachDecoratorSchema, required=True, many=True, data_key="files~attach"
    )

