"""Ping response handler."""

from aries_cloudagent.messaging.base_handler import (
    BaseHandler,
    BaseResponder,
    RequestContext,
)

from ..messages.protocolexample_response import ProtocolExampleResponse


class ProtocolExampleResponseHandler(BaseHandler):
    """Protocol Example response handler class."""

    async def handle(self, context: RequestContext, responder: BaseResponder):
        """
        Handle protocolexample response message.

        Args:
            context: Request context
            responder: Responder used to reply

        """

        self._logger.info("ProtocolExampleResponseHandler called")
        assert isinstance(context.message, ProtocolExampleResponse)

        self._logger.info(
            "Received protocolexample response from: %s with content - %s", context.message_receipt.sender_did, context.message
        )

