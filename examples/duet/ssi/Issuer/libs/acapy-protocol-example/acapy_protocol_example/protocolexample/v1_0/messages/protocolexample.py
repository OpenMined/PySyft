"""Represents a Privacy Preserving Machine Learning Message"""

from ..message_types import PROTOCOL_EXAMPLE, PROTOCOL_PACKAGE

from marshmallow import fields


from aries_cloudagent.messaging.agent_message import AgentMessage, AgentMessageSchema

HANDLER_CLASS = f"{PROTOCOL_PACKAGE}.handlers.protocolexample_handler.ProtocolExampleHandler"


class ProtocolExample(AgentMessage):
    """Class representing PPML Message"""

    class Meta:

        handler_class = HANDLER_CLASS
        message_type = PROTOCOL_EXAMPLE
        schema_class = "ProtocolExampleSchema"

    def __init__(
            self, *, response_requested: bool = True, example: str = None, **kwargs
    ):
        super(ProtocolExample, self).__init__(**kwargs)
        self.example = example
        self.response_requested = response_requested


class ProtocolExampleSchema(AgentMessageSchema):
    """Schema for Ppml class."""

    class Meta:
        """PpmlSchema metadata."""

        model_class = ProtocolExample

    response_requested = fields.Bool(
        default=True,
        required=False,
        description="Whether response is requested (default True)",
        example=True,
    )
    example = fields.Str(
        required=False,
        description="This is an example of a string field in your message",
        example="this is an example",
        allow_none=False
    )
