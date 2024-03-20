# stdlib
import asyncio
from collections.abc import Callable
from collections.abc import Coroutine
from enum import nonmember
import hashlib
import math
from struct import Struct
from typing import Any
import uuid

# third party
from loguru import logger
import veilid

# relative
from .constants import MAX_SINGLE_VEILID_MESSAGE_SIZE
from .constants import MAX_STREAMER_CONCURRENCY
from .utils import BytesEnum
from .utils import retry

# An asynchronous callable type hint that takes bytes as input and returns bytes
AsyncReceiveStreamCallback = Callable[[bytes], Coroutine[Any, Any, bytes]]
StreamId = bytes


class RequestType(BytesEnum):
    SIZE = nonmember(8)

    STREAM_START = b"@VS@SS"
    STREAM_CHUNK = b"@VS@SC"
    STREAM_END = b"@VS@SE"
    STREAM_SINGLE = b"@VS@S1"  # Special case for handling single chunk messages

    def __init__(self, value: bytes) -> None:
        # Members must be a bytes object of length == SIZE. If length is less than
        # SIZE, it'll be padded with null bytes to make it SIZE bytes long. If it is
        # greater, a ValueError will be raise.
        size = int(self.SIZE)  # type: ignore
        if len(value) > size:
            raise ValueError("Value must not be greater than 8 in length")
        if len(value) < size:
            value = value.ljust(size, b"\x00")
        self._value_ = value

    def __eq__(self, __other: object) -> bool:
        return self._value_ == __other


class ResponseType(BytesEnum):
    OK = b"@VS@OK"
    ERROR = b"@VS@ER"


class Buffer:
    def __init__(self, holds_reply: bool = False) -> None:
        self.msg_hash: bytes
        self.chunks: list[bytes | None]
        self.message: asyncio.Future[bytes] = asyncio.Future()
        self.holds_reply: bool = holds_reply
        # TODO add mechanism to delete/timeout old buffers
        # self.last_updated: float = asyncio.get_event_loop().time()

    def set_metadata(self, message_hash: bytes, chunks_count: int) -> None:
        self.message_hash = message_hash
        self.chunks = [None] * chunks_count

    def add_chunk(self, chunk_number: int, chunk: bytes) -> None:
        self.chunks[chunk_number] = chunk
        # self.last_updated = asyncio.get_event_loop().time()


class VeilidStreamer:
    """Pluggable class to make veild server capable of streaming large messages.

    This class is a singleton and should be used as such. It is designed to be used
    with the Veilid server to stream large messages over the network. It is capable of
    sending and receiving messages of any size by dividing them into chunks and
    reassembling them at the receiver's end.

    Data flow:
        Sender side:
            1. Send STREAM_START request -> Get OK
            3. Send all chunks using STREAM_CHUNK requests
            4. Send STREAM_END request -> Get OK
            ------ Operation for sending the message finished here ------
            5. Await reply from the receiver (the reply could also be >32kb in size)
               This will finish after step 5 of receiver side (See Below section)
            6. Return the reply once received
        Receiver side:
            1. Get STREAM_START request -> Set up buffers and send OK
            2. Receive all the chunks (STREAM_CHUNK request) and fill the buffers
            3. Get STREAM_END request -> Reassemble message -> Send OK
            ------ Operation for receiving the message finished here ------
            4. Pass the reassembled message to the callback function and get the reply
            5. Stream the reply back to the sender

    Structs:
        We are using 3 different structs to serialize and deserialize the metadata:

        1. stream_start_struct = Struct("!8s16s32sQ")  # 64 bytes
            [RequestType.STREAM_START (8 bytes string)] +
            [Stream ID (16 bytes random UUID string)] +
            [Message hash (32 bytes string)] +
            [Total chunks count (8 bytes unsigned long long)]

        2. stream_chunk_header_struct = Struct("!8s16sQ")  # 32 bytes
            [RequestType.STREAM_CHUNK (8 bytes string)] +
            [Stream ID (16 bytes random UUID string)] +
            [Current Chunk Number (8 bytes unsigned long long)]

        3. stream_end_struct = Struct("!8s16s")  # 24 bytes
            [RequestType.STREAM_END (8 bytes string)] +
            [Stream ID (16 bytes random UUID string)]

        The message is divided into chunks of 32736 bytes each, and each chunk is sent
        as a separate STREAM_CHUNK request. This helps in keeping the size of each
        request within the 32KB limit of the Veilid API.
            [stream_chunk_header_struct (32 bytes)] +
            [Actual Message Chunk (32736 bytes)]
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
               if vs.is_stream_update(update):
                   vs.receive_stream(connection, update, handle_receive_stream)
               ...other callback code...
           ```

        4. Use the `stream` method to send an app_call with a message of any size.
           ```
           response = await vs.stream(router, vld_key, message)
           ```

    Special case:
        If the message is small enough to fit in a single chunk, we can send it as a
        single STREAM_SINGLE request. This avoids the additional overhead while still
        allowing large replies containing multiple chunks.

        stream_single_struct = Struct("!8s16s")  # 24 bytes
            [RequestType.STREAM_SINGLE (8 bytes string)] +
            [Stream ID (16 bytes random UUID string)] +

        Therefore, the maximum size of the message that can be sent in a STREAM_SINGLE
        request is 32768 - 24 = 32744 bytes.
            [stream_single_struct (24 bytes)] +
            [Actual Message (32744 bytes)]
            = 32768 bytes

        Data flow for single chunk message:
            Sender side:
                1. Send STREAM_SINGLE request -> Get OK
                2. Await reply from the receiver
                3. Return the reply once received
            Receiver side:
                1. Get STREAM_SINGLE request -> Send OK
                2. Pass the message to the callback function and get the reply
                3. Stream the reply back to the sender

        Usage:
            This is automatically handled by the VeilidStreamer class. You don't need to
            do anything special for this. Just use the `stream` method as usual. If the
            message is small enough to fit in a single chunk, it will be sent as a
            STREAM_SINGLE request automatically.
    """

    _instance = None
    buffers: dict[StreamId, Buffer]

    def __new__(cls) -> "VeilidStreamer":
        # Nothing fancy here, just a simple singleton pattern
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.buffers = {}
        return cls._instance

    def __init__(self) -> None:
        self._init_structs()
        self._init_message_sizes()
        self._init_semaphores()

    @staticmethod
    def is_stream_update(update: veilid.VeilidUpdate) -> bool:
        """Checks if the update is a stream request."""
        if update.kind != veilid.VeilidUpdateKind.APP_CALL:
            return False
        prefix = update.detail.message[:8]
        return prefix in {r.value for r in RequestType}

    async def stream(
        self,
        router: veilid.RoutingContext,
        vld_key: str,
        message: bytes,
        stream_id: bytes | None = None,
    ) -> bytes:
        """Streams a message to the given DHT key."""
        # If stream_id is not present, this is a fresh request stream.
        is_request_stream = stream_id is None

        if is_request_stream:
            # Since this is a new request stream, so we need to generate a new stream_id
            stream_id = uuid.uuid4().bytes
            # Set up a buffer for holding the reply after the end of this request stream
            self.buffers[stream_id] = Buffer(holds_reply=True)

        if len(message) <= self.max_stream_single_msg_size:
            await self._stream_single_chunk_request(router, vld_key, message, stream_id)
        else:
            await self._stream_multi_chunk_request(router, vld_key, message, stream_id)

        if is_request_stream:
            response = await self._wait_for_reply(stream_id)
            self._cleanup_buffer(stream_id)
            return response

        return ResponseType.OK

    async def receive_stream(
        self,
        connection: veilid.VeilidAPI,
        router: veilid.RoutingContext,
        update: veilid.VeilidUpdate,
        callback: AsyncReceiveStreamCallback,
    ) -> None:
        """Receives a streamed message."""
        message = update.detail.message
        prefix = message[:8]

        if prefix == RequestType.STREAM_SINGLE:
            await self._handle_receive_stream_single(
                connection, router, update, callback
            )
        elif prefix == RequestType.STREAM_START:
            await self._handle_receive_stream_start(connection, update)
        elif prefix == RequestType.STREAM_CHUNK:
            await self._handle_receive_stream_chunk(connection, update)
        elif prefix == RequestType.STREAM_END:
            await self._handle_receive_stream_end(connection, router, update, callback)
        else:
            logger.error(f"[Bad Message] Message with unknown prefix: {prefix}")

    def _init_structs(self) -> None:
        # Structs for serializing and deserializing metadata as bytes of fixed length
        # https://docs.python.org/3/library/struct.html#format-characters
        BYTE_ORDER = "!"  # big-endian is recommended for networks as per IETF RFC 1700
        REQUEST_TYPE_PREFIX = f"{RequestType.SIZE}s"
        STREAM_ID = "16s"
        MESSAGE_HASH = "32s"
        TOTAL_CHUNKS_COUNT = "Q"
        CURRENT_CHUNK_NUMBER = "Q"

        self.stream_start_struct = Struct(
            BYTE_ORDER
            + REQUEST_TYPE_PREFIX
            + STREAM_ID
            + MESSAGE_HASH
            + TOTAL_CHUNKS_COUNT
        )
        self.stream_chunk_header_struct = Struct(
            BYTE_ORDER + REQUEST_TYPE_PREFIX + STREAM_ID + CURRENT_CHUNK_NUMBER
        )
        self.stream_end_struct = Struct(BYTE_ORDER + REQUEST_TYPE_PREFIX + STREAM_ID)
        self.stream_single_struct = Struct(BYTE_ORDER + REQUEST_TYPE_PREFIX + STREAM_ID)

    def _init_message_sizes(self) -> None:
        self.max_stream_chunk_msg_size = (
            MAX_SINGLE_VEILID_MESSAGE_SIZE - self.stream_chunk_header_struct.size
        )
        self.max_stream_single_msg_size = (
            MAX_SINGLE_VEILID_MESSAGE_SIZE - self.stream_single_struct.size
        )

    def _init_semaphores(self) -> None:
        self._send_request_semaphore = asyncio.Semaphore(MAX_STREAMER_CONCURRENCY)
        self._send_response_semaphore = asyncio.Semaphore(MAX_STREAMER_CONCURRENCY)

    @retry(veilid.VeilidAPIError, tries=4, delay=1, backoff=2)
    async def _send_request(
        self, router: veilid.RoutingContext, vld_key: str, request_data: bytes
    ) -> None:
        """Send an app call to the Veilid server and return the response."""
        async with self._send_request_semaphore:
            response = await router.app_call(vld_key, request_data)
            if response != ResponseType.OK:
                raise Exception("Unexpected response from server")

    async def _send_response(
        self,
        connection: veilid.VeilidAPI,
        update: veilid.VeilidUpdate,
        response: bytes,
    ) -> None:
        """Send a response to an app call."""
        async with self._send_response_semaphore:
            await connection.app_call_reply(update.detail.call_id, response)

    async def _send_ok_response(
        self, connection: veilid.VeilidAPI, update: veilid.VeilidUpdate
    ) -> None:
        await self._send_response(connection, update, ResponseType.OK)

    async def _send_error_response(
        self, connection: veilid.VeilidAPI, update: veilid.VeilidUpdate
    ) -> None:
        await self._send_response(connection, update, ResponseType.ERROR)

    def _cleanup_buffer(self, stream_id: bytes) -> None:
        del self.buffers[stream_id]

    def _calculate_chunks_count(self, message_size: int) -> int:
        total_no_of_chunks = math.ceil(message_size / self.max_stream_chunk_msg_size)
        return total_no_of_chunks

    def _get_chunk(
        self,
        stream_id: bytes,
        chunk_number: int,
        message: bytes,
    ) -> bytes:
        chunk_header = self.stream_chunk_header_struct.pack(
            RequestType.STREAM_CHUNK,
            stream_id,
            chunk_number,
        )
        cursor_start = chunk_number * self.max_stream_chunk_msg_size
        cursor_end = cursor_start + self.max_stream_chunk_msg_size
        chunk = message[cursor_start:cursor_end]
        return chunk_header + chunk

    async def _stream_single_chunk_request(
        self,
        router: veilid.RoutingContext,
        vld_key: str,
        message: bytes,
        stream_id: bytes,
    ) -> None:
        stream_single_request_header = self.stream_single_struct.pack(
            RequestType.STREAM_SINGLE, stream_id
        )
        stream_single_request = stream_single_request_header + message
        await self._send_request(router, vld_key, stream_single_request)

    async def _stream_multi_chunk_request(
        self,
        router: veilid.RoutingContext,
        vld_key: str,
        message: bytes,
        stream_id: bytes,
    ) -> None:
        message_size = len(message)
        message_hash = hashlib.sha256(message).digest()
        total_chunks_count = self._calculate_chunks_count(message_size)

        # Send STREAM_START request
        stream_start_request = self.stream_start_struct.pack(
            RequestType.STREAM_START,
            stream_id,
            message_hash,
            total_chunks_count,
        )
        await self._send_request(router, vld_key, stream_start_request)

        # Send chunks
        tasks = []
        for chunk_number in range(total_chunks_count):
            chunk = self._get_chunk(stream_id, chunk_number, message)
            tasks.append(self._send_request(router, vld_key, chunk))
        await asyncio.gather(*tasks)

        # Send STREAM_END request
        stream_end_message = self.stream_end_struct.pack(
            RequestType.STREAM_END, stream_id
        )
        await self._send_request(router, vld_key, stream_end_message)

    async def _wait_for_reply(self, stream_id: bytes) -> bytes:
        buffer = self.buffers[stream_id]
        logger.debug("Waiting for reply...")
        response = await buffer.message
        logger.debug("Reply received")
        return response

    async def _handle_receive_stream_single(
        self,
        connection: veilid.VeilidAPI,
        router: veilid.RoutingContext,
        update: veilid.VeilidUpdate,
        callback: AsyncReceiveStreamCallback,
    ) -> None:
        """Handles receiving STREAM_SINGLE request."""
        message = update.detail.message
        header_len = self.stream_single_struct.size
        header, message = message[:header_len], message[header_len:]
        _, stream_id = self.stream_single_struct.unpack(header)
        await self._send_ok_response(connection, update)

        buffer = self.buffers.get(stream_id)
        if buffer and buffer.holds_reply:
            # This message is being received by the sender and the stream() method must
            # be waiting for the reply. So we need to set the result in the buffer.
            logger.debug(f"Received single chunk reply of {len(message)} bytes...")
            buffer.message.set_result(message)
        else:
            # This message is being received by the receiver and we need to send back
            # the reply to the sender. So we need to call the callback function and
            # stream the reply back to the sender.
            logger.debug(f"Received single chunk request of {len(message)} bytes...")
            reply = await callback(message)
            logger.debug(
                f"Replying to {update.detail.sender} with {len(reply)} bytes of msg..."
            )
            await self.stream(router, update.detail.sender, reply, stream_id)
            # Finally delete the buffer
            self._cleanup_buffer(stream_id)

    async def _handle_receive_stream_start(
        self, connection: veilid.VeilidAPI, update: veilid.VeilidUpdate
    ) -> None:
        """Handles receiving STREAM_START request."""
        _, stream_id, message_hash, chunks_count = self.stream_start_struct.unpack(
            update.detail.message
        )
        buffer = self.buffers.get(stream_id)

        if buffer is None:
            # If the buffer is not present, this is a new request stream. So we need to
            # set up a new buffer to hold the chunks.
            buffer = Buffer(holds_reply=False)
            self.buffers[stream_id] = buffer
        buffer.set_metadata(message_hash, chunks_count)
        stream_type = "reply" if buffer.holds_reply else "request"
        logger.debug(f"Receiving {stream_type} stream of {chunks_count} chunks...")
        await self._send_ok_response(connection, update)

    async def _handle_receive_stream_chunk(
        self,
        connection: veilid.VeilidAPI,
        update: veilid.VeilidUpdate,
    ) -> None:
        """Handles receiving STREAM_CHUNK request."""
        message = update.detail.message
        chunk_header_len = self.stream_chunk_header_struct.size
        chunk_header, chunk = message[:chunk_header_len], message[chunk_header_len:]
        _, stream_id, chunk_number = self.stream_chunk_header_struct.unpack(
            chunk_header
        )
        buffer = self.buffers[stream_id]
        buffer.add_chunk(chunk_number, chunk)
        stream_type = "reply" if buffer.holds_reply else "request"
        logger.debug(
            f"Received {stream_type} chunk {chunk_number + 1}/{len(buffer.chunks)}"
        )
        await self._send_ok_response(connection, update)

    async def _handle_receive_stream_end(
        self,
        connection: veilid.VeilidAPI,
        router: veilid.RoutingContext,
        update: veilid.VeilidUpdate,
        callback: AsyncReceiveStreamCallback,
    ) -> None:
        """Handles receiving STREAM_END request."""
        _, stream_id = self.stream_end_struct.unpack(update.detail.message)
        buffer = self.buffers[stream_id]
        reassembled_message = b"".join(buffer.chunks)
        hash_matches = (
            hashlib.sha256(reassembled_message).digest() == buffer.message_hash
        )
        stream_type = "Reply" if buffer.holds_reply else "Request"
        logger.debug(
            f"{stream_type} message of {len(reassembled_message) // 1024} KB reassembled, hash matches: {hash_matches}"
        )

        if hash_matches:
            buffer.message.set_result(reassembled_message)
            await self._send_ok_response(connection, update)
        else:
            buffer.message.set_exception(Exception("Hash mismatch"))
            await self._send_error_response(connection, update)

        is_request_stream = not buffer.holds_reply
        if is_request_stream:
            # This message is being received on the receiver's end and we need to send
            # back the reply to the sender. So we need to call the callback function
            # and stream the reply back to the sender.
            reply = await callback(reassembled_message)
            logger.debug(
                f"Replying to {update.detail.sender} with {len(reply)} bytes of msg..."
            )
            # Stream as the reply itself could be greater than the max chunk size
            await self.stream(router, update.detail.sender, reply, stream_id)
            # Finally delete the buffer
            self._cleanup_buffer(stream_id)
