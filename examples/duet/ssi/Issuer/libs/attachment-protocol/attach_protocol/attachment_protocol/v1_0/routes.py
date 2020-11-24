# Attachment Protocol admin routes

from marshmallow import fields, Schema
from aiohttp import web, FormData
from aiohttp_apispec import docs, match_info_schema, form_schema, request_schema, response_schema
from aries_cloudagent.connections.models.connection_record import ConnectionRecord
from aries_cloudagent.messaging.valid import UUIDFour
from aries_cloudagent.storage.error import StorageNotFoundError

from .messages.attachment import Attachment, AttachmentSchema

class FileSchema(Schema):
    file = fields.Raw(
        description="File upload", required=True, type="file",
    )

class AttachmentMessageSchema(Schema):
    message = fields.Str(description="Message about file", example="Here is the file you asked for")

class ConnIdMatchInfoSchema(Schema):
    """Path parameters and validators for request taking connection id."""

    conn_id = fields.Str(
        description="Connection identifier", required=True, example=UUIDFour.EXAMPLE
    )

@docs(tags=["attachment protocol routes"], summary= "Attach file")
@match_info_schema(ConnIdMatchInfoSchema())
# @request_schema(AttachmentMessageSchema())
@form_schema(FileSchema())
async def send_attachment(request: web.BaseRequest):
    """
    Request Handler to send attachment protocol
    """
    context = request.app["request_context"]
    connection_id = request.match_info["conn_id"]
    outbound_handler = request.app["outbound_message_router"]

    # WARNING: don't do that if you plan to receive large files!
    # TODO change to handle large files??
    data = await request.post()

    # body = await request.json()
    # TODO figure out how to include this in the api. Probable extend the FileSchema
    message = "Here is the data you wanted"

    upfile = data['file']


    # .filename contains the name of the file in string format.
    filename = upfile.filename
    print("filename", filename)
    content_type = upfile.content_type

    # .file contains the actual file data that needs to be stored somewhere.
    file = upfile.file

    content = file.read()

    try:
        connection = await ConnectionRecord.retrieve_by_id(context, connection_id)
    except StorageNotFoundError:
        raise web.HTTPNotFound()

    if not connection.is_ready:
        raise web.HTTPBadRequest()

    attachment = Attachment(
        message=message,
        files_attach=[Attachment.wrap_file(content, filename, content_type)]
    )
    await outbound_handler(attachment, connection_id=connection_id)



    return web.json_response({"thread_id": attachment._thread_id})

async def register(app: web.Application):
    """Register routes."""

    app.add_routes(
        [web.post("/connections/{conn_id}/send-attachment", send_attachment)]
    )