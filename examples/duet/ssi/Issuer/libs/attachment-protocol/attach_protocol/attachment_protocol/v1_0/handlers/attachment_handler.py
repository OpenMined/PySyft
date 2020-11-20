#Attachment Handler

from aries_cloudagent.messaging.base_handler import (
    BaseHandler,
    BaseResponder,
    RequestContext,
)

from ..messages.attachment import Attachment

class AttachmentHandler(BaseHandler):

    async def handle(self, context: RequestContext, responder: BaseResponder):

        self._logger.info(f"AttachmentHandler called")
        assert isinstance(context.message, Attachment)

        self._logger.info("Received attached message %s", context.message)

        attachment = context.message
        attachment_message = attachment.message
        attach_decorator = attachment.files_attach[0]
        filename = attach_decorator.filename
        content = attach_decorator.data.base64

        print("Filename", filename)
        print("Content", content)

        # meta ={"content": body}
        
        self._logger.info("Send Webhook with topic attached message")
        await responder.send_webhook(
            "attachment",
            {
                "connection_id": context.connection_record.connection_id,
                "message_id": context.message._id,
                "attachment_message": attachment_message,
                "content": content,
                "filename": filename,
                "state": "received",
            },
        )
        
        # reply = None
        # if body:
        #     if context.settings.get("debug.auto_respond_messages"):
        #         if "received your message" not in body:
        #             reply = f"{context.default_label} received your message"
        #     elif body.startswith("Reply with: "):
        #         reply = body[12:]
        #
        # if reply:
        #     reply_msg = Attachment(message=reply)
        #     reply_msg.assign_thread_from(context.message)
        #     reply_msg.assign_trace_from(context.message)
        #     if "l10n" in context.message._decorators:
        #         reply_msg._decorators["l10n"] = context.message._decorators["l10n"]
        #     await responder.send_reply(reply_msg)
