# stdlib
import asyncio
from enum import Enum
import hashlib
import logging
from struct import Struct
from typing import Any
from typing import Callable
from typing import Coroutine

# third party
import veilid

# relative
from .constants import MAX_MESSAGE_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VEILID_STREAMER_STREAM_PREFIX = b"@VS"

# An asynchronous callable type hint that takes bytes as input and returns bytes
AsyncReceiveStreamCallback = Callable[[bytes], Coroutine[Any, Any, bytes]]


class VeilidStreamer:
    """Pluggable class to make veild server capable of streaming large messages.

    Data flow:
        Sender side:
            1. Send STREAM_START request -> Get OK
            3. Send all chunks using STREAM_CHUNK requests
            4. Send STREAM_END request -> Get OK
        Receiver side:
            1. Get STREAM_START request
            2. Set up buffers and send OK
            3. Receive all the chunks and fill the buffers
            4. Get STREAM_END request -> Reassemble message -> Send OK

    Structs:
        We are using 3 different structs to serialize and deserialize the metadata:

        1. stream_start_struct = Struct("!8s32sQ")  # 48 bytes
            [RequestType.STREAM_START (8 bytes string)] +
            [Message hash (32 bytes string)] +
            [Total chunks count (8 bytes unsigned long long)]

        2. stream_chunk_header_struct = Struct("!8s32sQ")  # 48 bytes
            [RequestType.STREAM_CHUNK (8 bytes string)] +
            [Message hash (32 bytes string)] +
            [Chunk Number (8 bytes unsigned long long)]

        3. stream_end_struct = Struct("!8s32s")  # 40 bytes
            [RequestType.STREAM_END (8 bytes string)] +
            [Message hash (32 bytes string)] = 40 bytes

        The message is divided into chunks of 32720 bytes each, and each chunk is sent
        as a separate STREAM_CHUNK request. This helps in keeping the size of each
        request within the 32KB limit of the Veilid API.
            [stream_chunk_header_struct (48 bytes)] +
            [Actual Message Chunk (32720 bytes)]
            = 32768 bytes

    Usage:
        1. Add this singleton class anwhere in your code, preferably above the update
           callback function for your connection.
           ```
           vs = VeilidStreamer()
           ```

        2. Add a callback function to handle the received message stream:
           ```
           async def handle_receive_stream(message: bytes) -> bytes:
               # Do something with the message once the entire stream is received.
               return b'some response to the sender of the stream.'
           ```

        3. Add the following to your connection's update_callback function to relay
           updates to the VeilidStreamer properly:
           ```
           def update_callback(update: veilid.VeilidUpdate) -> None:
               if VeilidStreamer.is_stream_update(update):
                   vs.receive_stream(connection, update, handle_receive_stream)
               ...other callback code...
           ```

        4. Use the `stream` method to send an app_call with a message of any size.
           ```
           response = await vs.stream(router, dht_key, message)
           ```
    """

    _instance = None

    class RequestType(Enum):
        STREAM_START = VEILID_STREAMER_STREAM_PREFIX + b"@SS"
        STREAM_CHUNK = VEILID_STREAMER_STREAM_PREFIX + b"@SC"
        STREAM_END = VEILID_STREAMER_STREAM_PREFIX + b"@SE"

    class ResponseType(Enum):
        OK = b"@VS@OK"
        ERROR = b"@VS@ER"

    def __new__(cls) -> "VeilidStreamer":
        if cls._instance is None:
            cls._instance = super().__new__(cls)

            # Key is the message hash, value is a list of chunks
            # Dict[bytes, List[bytes | None]]
            cls._instance.receive_buffer = {}
        return cls._instance

    def __init__(self) -> None:
        self.chunk_size = MAX_MESSAGE_SIZE

        MAX_CONCURRENT_REQUESTS = 200
        self._send_request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        self._send_response_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        # Structs for serializing and deserializing metadata as bytes of fixed length
        # '!' - big-endian byte order (recommended for networks as per IETF RFC 1700)
        # '8s' - String of length 8
        # '32s' - String of length 32
        # 'Q' - Unsigned long long (8 bytes)
        # https://docs.python.org/3/library/struct.html#format-characters
        self.stream_start_struct = Struct("!8s32sQ")  # 48 bytes
        self.stream_chunk_header_struct = Struct("!8s32sQ")  # 48 bytes
        self.stream_end_struct = Struct("!8s32s")  # 40 bytes

    @staticmethod
    def is_stream_update(update: veilid.VeilidUpdate) -> bool:
        """Checks if the update is a stream request."""
        return (
            update.kind == veilid.VeilidUpdateKind.APP_CALL
            and update.detail.message.startswith(VEILID_STREAMER_STREAM_PREFIX)
        )

    async def stream(
        self,
        router: veilid.RoutingContext,
        dht_key: str,
        message: bytes,
    ) -> bytes:
        """Streams a message to the given DHT key."""
        message_hash = hashlib.sha256(message).digest()
        chunks_count = self._calculate_chunks_count(message)

        # Send STREAM_START request
        stream_start_request = self.stream_start_struct.pack(
            VeilidStreamer.RequestType.STREAM_START.value,
            message_hash,
            chunks_count,
        )
        await self._send_request(router, dht_key, stream_start_request)

        # Send chunks
        tasks = []
        for chunk_number in range(chunks_count):
            chunk = self._get_chunk(message, message_hash, chunk_number)
            tasks.append(self._send_request(router, dht_key, chunk))
        await asyncio.gather(*tasks)

        # Send STREAM_END request
        stream_end_message = self.stream_end_struct.pack(
            VeilidStreamer.RequestType.STREAM_END.value, message_hash
        )
        response = await self._send_request(router, dht_key, stream_end_message)
        return response

    async def receive_stream(
        self,
        connection: veilid.VeilidAPI,
        update: veilid.VeilidUpdate,
        callback: AsyncReceiveStreamCallback,
    ) -> None:
        """Receives a streamed message."""
        call_id = update.detail.call_id
        message = update.detail.message

        if message.startswith(VeilidStreamer.RequestType.STREAM_START.value):
            await self._handle_receive_stream_start(connection, call_id, message)
        elif message.startswith(VeilidStreamer.RequestType.STREAM_CHUNK.value):
            await self._handle_receive_stream_chunk(connection, call_id, message)
        elif message.startswith(VeilidStreamer.RequestType.STREAM_END.value):
            await self._handle_receive_stream_end(
                connection, call_id, message, callback
            )
        else:
            logger.error(f"Bad message: {message}")

    async def _send_request(
        self, router: veilid.RoutingContext, dht_key: str, request_data: bytes
    ) -> bytes:
        """Send an app call to the Veilid server and return the response."""
        async with self._send_request_semaphore:
            response = await router.app_call(dht_key, request_data)
            ok_prefix = VeilidStreamer.ResponseType.OK.value
            if not response.startswith(ok_prefix):
                raise Exception("Unexpected response from server")
            return response[len(ok_prefix) :]

    async def _send_response(
        self,
        connection: veilid.VeilidAPI,
        call_id: veilid.OperationId,
        response: bytes,
    ) -> None:
        """Send a response to an app call."""
        async with self._send_response_semaphore:
            await connection.app_call_reply(call_id, response)

    def _calculate_chunks_count(self, message: bytes) -> int:
        message_size = len(message)
        chunk_size = self.chunk_size
        chunk_header_size = self.stream_chunk_header_struct.size

        no_of_chunks_in_msg = (message_size + chunk_size - 1) // chunk_size
        total_chunk_headers_size = no_of_chunks_in_msg * chunk_header_size
        size_with_headers = message_size + total_chunk_headers_size
        total_no_of_chunks = (size_with_headers + chunk_size - 1) // chunk_size
        return total_no_of_chunks

    def _get_chunk(
        self,
        message: bytes,
        message_hash: bytes,
        chunk_number: int,
    ) -> bytes:
        chunk_header = self.stream_chunk_header_struct.pack(
            VeilidStreamer.RequestType.STREAM_CHUNK.value,
            message_hash,
            chunk_number,
        )
        message_size = self.chunk_size - self.stream_chunk_header_struct.size
        cursor_start = chunk_number * message_size
        chunk = message[cursor_start : cursor_start + message_size]
        return chunk_header + chunk

    async def _handle_receive_stream_start(
        self, connection: veilid.VeilidAPI, call_id: veilid.OperationId, message: bytes
    ) -> None:
        """Handles receiving STREAM_START request."""
        _, message_hash, chunks_count = self.stream_start_struct.unpack(message)
        logger.debug(f"Receiving stream of {chunks_count} chunks...")
        self.receive_buffer[message_hash] = [None] * chunks_count
        await self._send_response(
            connection, call_id, VeilidStreamer.ResponseType.OK.value
        )

    async def _handle_receive_stream_chunk(
        self, connection: veilid.VeilidAPI, call_id: veilid.OperationId, message: bytes
    ) -> None:
        """Handles receiving STREAM_CHUNK request."""
        chunk_header_len = self.stream_chunk_header_struct.size
        chunk_header, chunk = message[:chunk_header_len], message[chunk_header_len:]
        _, message_hash, chunk_number = self.stream_chunk_header_struct.unpack(
            chunk_header
        )
        buffer = self.receive_buffer[message_hash]
        buffer[chunk_number] = chunk
        logger.debug(
            f"Received chunk {chunk_number + 1}/{len(buffer)}; Length: {len(chunk)}"
        )
        await self._send_response(
            connection, call_id, VeilidStreamer.ResponseType.OK.value
        )

    async def _handle_receive_stream_end(
        self,
        connection: veilid.VeilidAPI,
        call_id: veilid.OperationId,
        message: bytes,
        callback: AsyncReceiveStreamCallback,
    ) -> None:
        """Handles receiving STREAM_END request."""
        _, message_hash = self.stream_end_struct.unpack(message)
        buffer = self.receive_buffer[message_hash]
        message = b"".join(buffer)
        hash_matches = hashlib.sha256(message).digest() == message_hash
        logger.debug(
            f"Message of {len(message) // 1024} KB reassembled, hash matches: {hash_matches}"
        )
        if not hash_matches:
            await self._send_response(
                connection, call_id, VeilidStreamer.ResponseType.ERROR.value
            )
        result = await callback(message)
        response = VeilidStreamer.ResponseType.OK.value + result
        await self._send_response(connection, call_id, response)
        del self.receive_buffer[message_hash]
