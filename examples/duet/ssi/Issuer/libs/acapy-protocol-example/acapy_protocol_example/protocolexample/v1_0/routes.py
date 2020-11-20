"""Protocol example admin routes."""

from aiohttp import web
from aiohttp_apispec import docs, match_info_schema, request_schema, response_schema

from marshmallow import fields, Schema

from aries_cloudagent.connections.models.connection_record import ConnectionRecord
from aries_cloudagent.messaging.valid import UUIDFour
from aries_cloudagent.storage.error import StorageNotFoundError

from .messages.protocolexample import ProtocolExample

class ProtocolExampleRequestSchema(Schema):

    example = fields.Str(required=True, description="An example field in a http request")
    response_requested = fields.Bool(required=False, description="Set to true if you want for a ProtocolExampleResponse to this message")


class ProtocolExampleRequestResponseSchema(Schema):

    thread_id = fields.Str(required=False, description="Thread ID of the ping message")


class ConnIdMatchInfoSchema(Schema):
    """Path parameters and validators for request taking connection id."""

    conn_id = fields.Str(
        description="Connection identifier", required=True, example=UUIDFour.EXAMPLE
    )


@docs(tags=["protocol example routes"], summary="Tell agent to invoke the protocolexample")
@match_info_schema(ConnIdMatchInfoSchema())
@request_schema(ProtocolExampleRequestSchema())
@response_schema(ProtocolExampleRequestResponseSchema(), 200)
async def connections_send_ping(request: web.BaseRequest):
    """
    Request handler for sending a trust ping to a connection.

    Args:
        request: aiohttp request object

    """
    context = request.app["request_context"]
    connection_id = request.match_info["conn_id"]
    outbound_handler = request.app["outbound_message_router"]
    body = await request.json()
    example = body.get("example")
    response_requested = body.get("response_requested")

    try:
        connection = await ConnectionRecord.retrieve_by_id(context, connection_id)
    except StorageNotFoundError:
        raise web.HTTPNotFound()

    if not connection.is_ready:
        raise web.HTTPBadRequest()

    msg = ProtocolExample(example=example, response_requested=response_requested)
    await outbound_handler(msg, connection_id=connection_id)

    return web.json_response({"thread_id": msg._thread_id})


async def register(app: web.Application):
    """Register routes."""

    app.add_routes(
        [web.post("/connections/{conn_id}/test-protocolexample", connections_send_ping)]
    )