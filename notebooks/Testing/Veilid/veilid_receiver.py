# stdlib
import asyncio
import logging
import sys
import time

# third party
from utils import RequestType
from utils import VeilidStreamer
from utils import allocate_route
from utils import connect_veilid
from utils import start_veilid_container
import veilid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def process_app_messages(
    app_message_queue: asyncio.Queue[veilid.VeilidUpdate], vs: VeilidStreamer
):
    msg_no = 1
    while True:
        update = await app_message_queue.get()
        if update.detail.message.startswith(RequestType.STREAM_PREFIX.encode()):
            message = await vs.receive_stream(update)
            if message:
                start_time = int.from_bytes(message[:4], byteorder="big")
                logger.info(
                    f"Received {len(message)} bytes in {int(time.time()) - start_time} seconds"
                )
                msg_no += 1
        elif update.kind == veilid.VeilidUpdateKind.APP_MESSAGE:
            logger.info(f"Message received: {update.detail.message}")
        else:
            logger.info(f"Unknown message: {update}")


async def main(host: str, port: int):
    # veilid_container = None
    router = None
    process_messages_task = None
    try:
        app_message_queue: asyncio.Queue[veilid.VeilidUpdate] = asyncio.Queue()
        # veilid_container = start_veilid_container(port=port)
        conn = await connect_veilid(host, port, app_message_queue)
        router: veilid.RoutingContext = await (
            await conn.new_routing_context()
        ).with_default_safety()

        record = await router.create_dht_record(veilid.DHTSchema.dflt(1))
        public_key, private_key = record.owner, record.owner_secret
        await router.close_dht_record(record.key)
        key_pair = veilid.KeyPair.from_parts(key=public_key, secret=private_key)
        record_open = await router.open_dht_record(record.key, key_pair)

        route_id, blob = await allocate_route(conn)
        logger.debug(f"Route ID: {route_id}, Blob: {blob}")
        await router.set_dht_value(record_open.key, 0, blob)
        logger.info(f"Your DHT Key: {record.key}")

        self_remote_private_route = await conn.import_remote_private_route(blob)
        await router.app_message(self_remote_private_route, b"Ready!")

        vs = VeilidStreamer(connection=conn, router=router)
        process_messages_task = asyncio.create_task(
            process_app_messages(app_message_queue, vs), name="app call task"
        )
        await process_messages_task
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.exception(e)
    finally:
        logger.info("Shutting down...")
        if router:
            await router.release()
        if process_messages_task:
            process_messages_task.cancel()
        # if veilid_container:
        #     veilid_container.stop()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python veilid_receiver.py <port>")
        sys.exit(1)
    host = "localhost"
    port = int(sys.argv[1])
    try:
        asyncio.run(main(host=host, port=port))
    except KeyboardInterrupt:
        pass
