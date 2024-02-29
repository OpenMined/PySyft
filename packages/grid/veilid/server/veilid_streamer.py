# stdlib
from enum import Enum
import hashlib
import logging
from struct import Struct
from typing import Dict
from typing import List

# third party
import veilid
from veilid_core import app_call
from veilid_core import get_veilid_conn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VEILID_STREAMER_STREAM_PREFIX = b"@VS"


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

    class RequestType(Enum):
        STREAM_START = VEILID_STREAMER_STREAM_PREFIX + b"@SS"
        STREAM_CHUNK = VEILID_STREAMER_STREAM_PREFIX + b"@SC"
        STREAM_END = VEILID_STREAMER_STREAM_PREFIX + b"@SE"

    class ResponseType(Enum):
        OK = b"@VS@OK"
        ERROR = b"@VS@ER"

    def __init__(self, chunk_size=32 * 1024):
        self.chunk_size = chunk_size

        # Key is the message hash, value is a list of chunks
        self.receive_buffer: Dict[bytes, List[bytes]] = {}

        # Structs for serializing and deserializing metadata as bytes of fixed length
        # '!' - big-endian byte order as per IETF RFC 1700
        # '8s' - String of length 8
        # 'Q' - Unsigned long long (8 bytes)
        # '32s' - String of length 32
        # https://docs.python.org/3/library/struct.html#format-characters
        self.stream_start_struct = Struct("!8s32sQ")  # 48 bytes
        self.stream_chunk_header_struct = Struct("!8s32sQ")  # 48 bytes
        self.stream_end_struct = Struct("!8s32s")  # 40 bytes

    async def _send_request(self, dht_key: str, request_data: bytes) -> bytes:
        """Send an app call to the Veilid server and return the response."""
        response = await app_call(dht_key, request_data)
        if response != VeilidStreamer.ResponseType.OK:
            raise Exception("Unexpected response from server")
        return response

    async def _send_response(self, call_id: veilid.OperationId, response: bytes):
        """Send a response to an app call."""
        async with await get_veilid_conn() as conn:
            await conn.app_call_reply(call_id, response)

    def _calculate_chunks_count(self, message: bytes) -> int:
        message_size = len(message)
        chunk_size = self.chunk_size
        chunk_header_size = self.stream_chunk_header_struct.size

        no_of_chunks_in_msg = (message_size + chunk_size - 1) // chunk_size
        total_chunk_headers_size = no_of_chunks_in_msg * chunk_header_size
        size_with_headers = message_size + total_chunk_headers_size
        total_no_of_chunks = (size_with_headers + chunk_size - 1) // chunk_size
        return total_no_of_chunks

    def _get_chunk(self, message: bytes, chunk_number: int) -> bytes:
        message_size = self.chunk_size - len(self.stream_chunk_header_struct.size)
        cursor_start = chunk_number * message_size
        return message[cursor_start : cursor_start + message_size]

    async def stream(self, dht_key: str, message: bytes):
        """Streams a message to the given DHT key."""
        message_hash = hashlib.sha256(message).digest()
        chunks_count = self._calculate_chunks_count(message)

        # Send STREAM_START request
        stream_start_request = self.stream_start_struct.pack(
            VeilidStreamer.RequestType.STREAM_START,
            message_hash,
            chunks_count,
        )
        await self._send_request(dht_key, stream_start_request)

        # Send chunks
        for chunk_number in range(chunks_count):
            chunk_header = self.stream_chunk_header_struct.pack(
                VeilidStreamer.RequestType.STREAM_CHUNK,
                message_hash,
                chunk_number,
            )
            chunk = self._get_chunk(message, chunk_number)
            chunk_data = chunk_header + chunk
            await self._send_request(dht_key, chunk_data)

        # Send STREAM_END request
        stream_end_message = self.stream_start_struct.pack(
            VeilidStreamer.RequestType.STREAM_END, message_hash
        )
        await self._send_request(dht_key, stream_end_message)

    async def _handle_receive_stream_start(
        self, call_id: veilid.OperationId, message: bytes
    ):
        """Handles receiving STREAM_START request."""
        _, message_hash, chunks_count = self.stream_start_struct.unpack(message)
        logger.info(f"Receiving stream of {chunks_count} chunks; Hash {message_hash}")
        self.receive_buffer[message_hash] = [None] * chunks_count
        await self._send_response(call_id, VeilidStreamer.ResponseType.OK)

    async def _handle_receive_stream_chunk(
        self, call_id: veilid.OperationId, message: bytes
    ):
        """Handles receiving STREAM_CHUNK request."""
        chunk_header_len = self.stream_chunk_header_struct.size
        chunk_header, chunk = message[:chunk_header_len], message[chunk_header_len:]
        _, message_hash, chunk_number = self.stream_chunk_header_struct.unpack(
            chunk_header
        )
        buffer = self.receive_buffer[message_hash]
        buffer[chunk_number] = chunk
        logger.info(f"Got chunk {chunk_number + 1}/{len(buffer)}; Length: {len(chunk)}")
        await self._send_response(call_id, VeilidStreamer.ResponseType.OK)

    async def _handle_receive_stream_end(
        self, call_id: veilid.OperationId, message: bytes
    ) -> bytes:
        """Handles receiving STREAM_END request."""
        _, message_hash = self.stream_end_struct.unpack(message)
        buffer = self.receive_buffer[message_hash]
        message = b"".join(buffer)
        hash_matches = hashlib.sha256(message).digest() == message_hash
        logger.info(f"Message reassembled, hash matches: {hash_matches}")
        response = (
            VeilidStreamer.ResponseType.OK
            if hash_matches
            else VeilidStreamer.ResponseType.ERROR
        )
        await self._send_response(call_id, response)
        del self.receive_buffer[message_hash]
        return message

    async def receive_stream(self, update: veilid.VeilidUpdate) -> bytes:
        """Receives a streamed message."""
        call_id = update.detail.call_id
        message = update.detail.message

        if message.startswith(VeilidStreamer.RequestType.STREAM_START):
            await self._handle_receive_stream_start(call_id, message)
        elif message.startswith(VeilidStreamer.RequestType.STREAM_CHUNK):
            await self._handle_receive_stream_chunk(call_id, message)
        elif message.startswith(VeilidStreamer.RequestType.STREAM_END):
            return await self._handle_receive_stream_end(call_id, message)
        else:
            logger.info(f"Bad message: {message}")
