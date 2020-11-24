from aries_cloudagent.messaging.base_handler import (
    BaseHandler,
    BaseResponder,
    RequestContext,
)

from ..messages.protocolexample import ProtocolExample
from ..messages.protocolexample_response import ProtocolExampleResponse

class ProtocolExampleHandler(BaseHandler):

    async def handle(self, context: RequestContext, responder: BaseResponder):

        self._logger.info(f"ProtocolExampleHandler called")
        assert isinstance(context.message, ProtocolExample)

        self._logger.info(
            "Received protocolexample message from: %s with content - %s", context.message_receipt.sender_did, context.message
        )

        if not context.connection_ready:
            self._logger.info(
                "Connection not active, skipping protocolexample response: %s",
                context.message_receipt.sender_did,
            )
            return


        if context.message.response_requested:
            self._logger.info("Response requested, sending ...")
            reply = ProtocolExampleResponse(example_response="Thank you for your protocol message")
            reply.assign_thread_from(context.message)
            reply.assign_trace_from(context.message)
            await responder.send_reply(reply)

        self._logger.info("Send webhook with topic protocolexample")
        await responder.send_webhook(
            "protocolexample",
            {
                "example": context.message.example,
                "connection_id": context.message_receipt.connection_id,
                "state": "received",
                "thread_id": context.message._thread_id,
            },
        )

