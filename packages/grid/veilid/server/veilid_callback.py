# stdlib
import base64
import json

# third party
import httpx
from loguru import logger
import veilid
from veilid import VeilidUpdate

# relative
from .veilid_connection import get_routing_context
from .veilid_connection import get_veilid_conn
from .veilid_streamer import VeilidStreamer


async def handle_app_message(update: VeilidUpdate) -> None:
    logger.info(f"Received App Message: {update.detail.message}")


async def handle_app_call(message: bytes) -> bytes:
    logger.info(f"Received App Call of {len(message)} bytes.")
    message_dict: dict = json.loads(message)

    async with httpx.AsyncClient() as client:
        data = message_dict.get("data", None)
        # TODO: can we optimize this?
        # We encode the data to base64, as while sending
        # json expects valid utf-8 strings
        if data:
            message_dict["data"] = base64.b64decode(data)
        response = await client.request(
            method=message_dict.get("method"),
            url=message_dict.get("url"),
            data=message_dict.get("data", None),
            params=message_dict.get("params", None),
            json=message_dict.get("json", None),
        )

        # TODO: Currently in `dev` branch, compression is handled by the veilid internals,
        # but we are decompressing it on the client side. Should both the compression and
        # decompression be done either on the client side (for more client control) or by
        # the veilid internals (for abstraction)?

        # compressed_response = lzma.compress(response.content)
        # logger.info(f"Compression response size: {len(compressed_response)}")
        # return compressed_response
        return response.content


# TODO: Handle other types of network events like
# when our private route goes
async def main_callback(update: VeilidUpdate) -> None:
    if VeilidStreamer.is_stream_update(update):
        async with await get_veilid_conn() as conn:
            async with await get_routing_context(conn) as router:
                await VeilidStreamer().receive_stream(
                    conn, router, update, handle_app_call
                )
    elif update.kind == veilid.VeilidUpdateKind.APP_MESSAGE:
        await handle_app_message(update)

    elif update.kind == veilid.VeilidUpdateKind.APP_CALL:
        response = await handle_app_call(update.detail.message)
        async with await get_veilid_conn() as conn:
            await conn.app_call_reply(update.detail.call_id, response)
