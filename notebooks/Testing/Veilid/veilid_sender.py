# stdlib
import asyncio
import logging
import random
import sys
import time

# third party
from utils import VeilidStreamer
from utils import connect_veilid
from utils import get_typed_key
from utils import start_veilid_container
import veilid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main(host: str, port: int):
    # veilid_container = None
    router = None
    try:
        app_message_queue: asyncio.Queue = asyncio.Queue()
        # veilid_container = start_veilid_container(port=port)
        conn = await connect_veilid(host, port, app_message_queue)
        router: veilid.RoutingContext = await (
            await conn.new_routing_context()
        ).with_default_safety()

        dht_key_str = input("Enter DHT Key of the receiver: ")
        dht_key = get_typed_key(dht_key_str.lstrip("VLD0:"))
        try:
            await router.close_dht_record(key=dht_key)
        except veilid.VeilidAPIErrorGeneric:
            pass
        await router.open_dht_record(key=dht_key, writer=None)
        logger.info("DHT record opened")

        record_value = await router.get_dht_value(
            key=dht_key, subkey=0, force_refresh=True
        )
        private_route = await conn.import_remote_private_route(record_value.data)

        vs = VeilidStreamer(connection=conn, router=router)

        for message_size_kb in range(0, 13):  # Powers of two from 1 to 4096
            message_size_kb = 2**message_size_kb
            current_time = int(time.time()).to_bytes(4, byteorder="big")  # 4 bytes
            message_size = (message_size_kb * 1024) - 4
            message = current_time + random.randbytes(message_size)
            logger.info(f"Sending message of size {len(message)} bytes")
            await vs.stream(private_route, message)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.exception(e)
    finally:
        logger.info("Shutting down...")
        if router:
            await router.release()
        # if veilid_container:
        #     veilid_container.stop()


if __name__ == "__main__":
    host = "localhost"
    if len(sys.argv) != 2:
        print("Usage: python veilid_receiver.py <port>")
        sys.exit(1)
    port = int(sys.argv[1])
    try:
        asyncio.run(main(host=host, port=port))
    except KeyboardInterrupt:
        pass
