# stdlib
import base64
import json
import lzma

# third party
import httpx
from loguru import logger
import veilid
from veilid import VeilidUpdate

# relative
from .veilid_connection import get_veilid_conn


async def handle_app_message(update: VeilidUpdate) -> None:
    logger.info(f"Received App Message: {update.detail.message}")


async def handle_app_call(update: VeilidUpdate) -> None:
    logger.info(f"Received App Call: {update.detail.message}")
    message: dict = json.loads(update.detail.message)

    async with httpx.AsyncClient() as client:
        data = message.get("data", None)
        # TODO: can we optimize this?
        # We encode the data to base64,as while sending
        # json expects valid utf-8 strings
        if data:
            message["data"] = base64.b64decode(data)
        response = await client.request(
            method=message.get("method"),
            url=message.get("url"),
            data=message.get("data", None),
            params=message.get("params", None),
            json=message.get("json", None),
        )

    async with await get_veilid_conn() as conn:
        compressed_response = lzma.compress(response.content)
        logger.info(f"Compression response size: {len(compressed_response)}")
        await conn.app_call_reply(update.detail.call_id, compressed_response)


# TODO: Handle other types of network events like
# when our private route goes
async def main_callback(update: VeilidUpdate) -> None:
    if update.kind == veilid.VeilidUpdateKind.APP_MESSAGE:
        await handle_app_message(update)

    elif update.kind == veilid.VeilidUpdateKind.APP_CALL:
        await handle_app_call(update)
