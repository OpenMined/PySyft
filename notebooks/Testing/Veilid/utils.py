# stdlib
import asyncio
from dataclasses import dataclass
from enum import StrEnum
from functools import wraps
import hashlib
import logging

# third party
import docker
from tqdm import tqdm
import veilid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def retry(retries: int, wait_seconds_before: int):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retries_left = retries
            last_error = None
            while retries_left:
                try:
                    if wait_seconds_before:
                        await asyncio.sleep(wait_seconds_before)
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    retries_left -= 1
            raise Exception(
                f"Retry limit exceeded for '{func.__name__}' with error: {last_error}"
            )

        return wrapper

    return decorator


def start_veilid_container(port: int):
    logger.info("Starting veilid container...")
    docker_client = docker.from_env()
    container = docker_client.containers.run(
        "veilid:latest", ports={"5959/tcp": port, "5959/udp": port}, detach=True
    )
    logger.info("Veilid container started")
    return container


async def veilid_callback(
    update: veilid.VeilidUpdate, app_message_queue: asyncio.Queue
):
    if update.kind in {
        veilid.VeilidUpdateKind.APP_MESSAGE,
        veilid.VeilidUpdateKind.APP_CALL,
    }:
        await app_message_queue.put(update)


@retry(retries=15, wait_seconds_before=2)
async def connect_veilid(host: str, port: int, app_message_queue: asyncio.Queue):
    conn = await veilid.json_api_connect(
        host, port, lambda update: veilid_callback(update, app_message_queue)
    )
    state = await conn.get_state()
    public_internet_ready = state.attachment.public_internet_ready
    if not public_internet_ready:
        raise Exception("Veilid connection failed")
    logger.info(f"Connected to veilid at {host}:{port}")
    return conn


@retry(retries=15, wait_seconds_before=2)
async def allocate_route(conn: veilid.VeilidAPI):
    route = await conn.new_custom_private_route(
        [veilid.CryptoKind.CRYPTO_KIND_VLD0],
        veilid.Stability.RELIABLE,
        veilid.Sequencing.ENSURE_ORDERED,
    )
    logger.debug(f"Route allocated: {route}")
    return route


def get_typed_key(key: str) -> veilid.types.TypedKey:
    return veilid.types.TypedKey.from_value(
        kind=veilid.CryptoKind.CRYPTO_KIND_VLD0, value=key
    )


class RequestType(StrEnum):
    STREAM_PREFIX = "1@"
    STREAM_START = "1@1@"
    STREAM_CHUNK = "1@2@"
    STREAM_END = "1@3@"


class ResponseType(StrEnum):
    OK = "1@"
    ERROR = "2@"


class VeilidStreamer:
    """Handle sending and receiving large messages over Veilid
    Sender side:
        1. Send STREAM_START request -> Get OK
        3. Send all chunks using STREAM_CHUNK requests
        4. Send STREAM_END request -> Get OK
    Receiver side:
        1. Get STREAM_START request
        2. Set up buffers and send OK
        3. Receive all the chunks and fill the buffers
        4. Get STREAM_END request -> Reassemble message -> Send OK
    Chunk structure:
        [RequestType.STREAM_CHUNK][Message hash][Chunk Number][Actual Message Chunk]
    """

    @dataclass
    class MessageMetaData:
        total_size: int
        chunks_count: int
        message_hash: str

    def __init__(
        self,
        connection: veilid.VeilidAPI,
        router: veilid.RoutingContext,
        chunk_size: int = 32 * 1024,
    ):
        self.connection = connection
        self.router = router
        self.chunk_size = chunk_size
        self.receive_buffer = {}  # key is the message hash

    def _get_message_metadata(self, message: bytes) -> MessageMetaData:
        message_size = len(message)
        chunks_count = (message_size + self.chunk_size - 1) // self.chunk_size

        # Each chunk will contain a header of 48 bytes, increasing the total chunk count
        single_chunk_header_size = 48
        total_chunk_headers_size = chunks_count * single_chunk_header_size
        total_bytes = message_size + total_chunk_headers_size
        chunks_count = (total_bytes + self.chunk_size - 1) // self.chunk_size

        message_hash = hashlib.sha256(message).digest()
        return self.MessageMetaData(message_size, chunks_count, message_hash)

    def _serialize_metadata(self, metadata: MessageMetaData) -> bytes:
        """
        Serializes the given metadata object into bytes. The serialized format is:
            - 16 bytes for the total size of the message
            - 8 bytes for the number of chunks
            - 32 bytes for the message hash
            - Total: 56 bytes
        Using big-endian encoding for all fields.

        Args:
            metadata (MessageMetaData): The metadata object to be serialized.

        Returns:
            bytes: The serialized metadata as bytes.
        """
        total_size_bytes = metadata.total_size.to_bytes(16, byteorder="big")
        chunks_count_bytes = metadata.chunks_count.to_bytes(8, byteorder="big")
        message_hash_bytes = metadata.message_hash  # 32 bytes
        return total_size_bytes + chunks_count_bytes + message_hash_bytes

    async def stream(
        self,
        target: veilid.types.TypedKey | veilid.types.RouteId,
        message: bytes,
    ):
        # Send STREAM_START request
        header = RequestType.STREAM_START.ljust(8).encode()
        message_metadata = self._get_message_metadata(message)
        metadata_bytes = header + self._serialize_metadata(message_metadata)

        response = await self.router.app_call(target, metadata_bytes)
        if response.decode() != ResponseType.OK:
            raise Exception("Unexpected response from server")

        # All good, send the chunks
        header = (
            RequestType.STREAM_CHUNK.ljust(8).encode()
            + message_metadata.message_hash  # 32 bytes
        )
        cursor_start = 0
        for chunk_number in tqdm(
            range(message_metadata.chunks_count), desc="Sending chunks"
        ):
            chunk_header = header + chunk_number.to_bytes(8, byteorder="big")

            cursor_end = cursor_start + self.chunk_size - len(chunk_header)
            chunk = message[cursor_start:cursor_end]
            cursor_start = cursor_end

            chunk_data = chunk_header + chunk  # 32768 bytes
            response = await self.router.app_call(target, chunk_data)
            if response.decode() != ResponseType.OK:
                raise Exception("Unexpected response from server")

        # All chunks sent, send STREAM_END request
        header = RequestType.STREAM_END.ljust(8).encode()
        message_metadata = self._get_message_metadata(message)
        metadata_bytes = header + self._serialize_metadata(message_metadata)

        response = await self.router.app_call(target, metadata_bytes)
        if response.decode() != ResponseType.OK:
            raise Exception("Unexpected response from server")

    def _deserialize_metadata(self, metadata_bytes: bytes) -> MessageMetaData:
        """
        Deserializes the given bytes into a metadata object. The serialized format is:
            - 16 bytes for the total size of the message
            - 8 bytes for the number of chunks
            - 32 bytes for the message hash
            - Total: 56 bytes
        Using big-endian encoding for all fields.

        Args:
            metadata_bytes (bytes): The serialized metadata as bytes.

        Returns:
            MessageMetaData: The deserialized metadata object.
        """
        total_size = int.from_bytes(metadata_bytes[:16], byteorder="big")
        chunks_count = int.from_bytes(metadata_bytes[16:24], byteorder="big")
        message_hash = metadata_bytes[24:]
        return self.MessageMetaData(total_size, chunks_count, message_hash)

    async def _handle_receive_stream_start(self, update: veilid.VeilidUpdate):
        message = update.detail.message
        metadata = self._deserialize_metadata(message[8:])
        logger.debug(f"Metadata: {metadata}")
        self.receive_buffer[metadata.message_hash] = (
            metadata,
            [None] * metadata.chunks_count,
        )
        await self.connection.app_call_reply(
            update.detail.call_id, ResponseType.OK.encode()
        )

    async def _handle_receive_stream_chunk(self, update: veilid.VeilidUpdate):
        message = update.detail.message
        message_hash = message[8:40]
        chunk_number = int.from_bytes(message[40:48], byteorder="big")
        chunk = message[48:]
        expected_metadata, chunks = self.receive_buffer[message_hash]
        chunks[chunk_number] = chunk
        logger.debug(
            f"Chunk {chunk_number + 1}/{expected_metadata.chunks_count}, chunk length: {len(chunk)}"
        )
        await self.connection.app_call_reply(
            update.detail.call_id, ResponseType.OK.encode()
        )

    async def _handle_receive_stream_end(self, update: veilid.VeilidUpdate) -> bytes:
        message = update.detail.message
        metadata = self._deserialize_metadata(message[8:])
        chunks = self.receive_buffer[metadata.message_hash][1]
        message = b"".join(chunks)
        await self.connection.app_call_reply(
            update.detail.call_id, ResponseType.OK.encode()
        )
        hash_matches = hashlib.sha256(message).digest() == metadata.message_hash
        logger.info(f"Message reassembled, hash matches: {hash_matches}")
        return message

    async def receive_stream(self, update: veilid.VeilidUpdate) -> bytes:
        message = update.detail.message
        if message.startswith(RequestType.STREAM_START.encode()):
            await self._handle_receive_stream_start(update)
        elif message.startswith(RequestType.STREAM_CHUNK.encode()):
            await self._handle_receive_stream_chunk(update)
        elif message.startswith(RequestType.STREAM_END.encode()):
            return await self._handle_receive_stream_end(update)
        else:
            logger.info(f"Bad message: {update}")
