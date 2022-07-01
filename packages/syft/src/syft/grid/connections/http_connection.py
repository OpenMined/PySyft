# stdlib
import json
from typing import Dict as TypeDict
from typing import Optional
from typing import Union

# third party
import requests

# relative
from .. import GridURL
from ...core.common.message import SignedEventualSyftMessageWithoutReply
from ...core.common.message import SignedImmediateSyftMessageWithReply
from ...core.common.message import SignedImmediateSyftMessageWithoutReply
from ...core.common.message import SyftMessage
from ...core.common.serde.deserialize import _deserialize
from ...core.common.serde.serialize import _serialize
from ...core.io.connection import ClientConnection
from ...core.node.enums import RequestAPIFields
from ...core.node.exceptions import RequestAPIException
from ...proto.core.node.common.metadata_pb2 import Metadata as Metadata_PB

DEFAULT_TIMEOUT = 30  # seconds


class HTTPConnection(ClientConnection):
    proxies: TypeDict[str, str] = {}

    def __init__(self, url: Union[str, GridURL]) -> None:
        self.base_url = GridURL.from_url(url) if isinstance(url, str) else url
        if self.base_url is None:
            raise Exception(f"Invalid GridURL. {self.base_url}")

    def send_immediate_msg_with_reply(
        self,
        msg: SignedImmediateSyftMessageWithReply,
        timeout: Optional[float] = None,
        return_signed: bool = False,
    ) -> SignedImmediateSyftMessageWithoutReply:
        """
        Sends high priority messages and wait for their responses.

        This method implements a HTTP version of the
        ClientConnection.send_immediate_msg_with_reply

        :return: returns an instance of SignedImmediateSyftMessageWithReply.
        :rtype: SignedImmediateSyftMessageWithoutReply
        """

        # Serializes SignedImmediateSyftMessageWithReply
        # and send it using HTTP protocol
        response = self._send_msg(msg=msg, timeout=timeout)

        # Deserialize node's response
        if response.status_code == requests.codes.ok:
            # Return SignedImmediateSyftMessageWithoutReply
            return _deserialize(blob=response.content, from_bytes=True)

        try:
            response_json = json.loads(response.content)
            raise RequestAPIException(response_json[RequestAPIFields.ERROR])
        except Exception as e:
            print(f"Unable to json decode message. {e}")
            raise e

    def send_immediate_msg_without_reply(
        self,
        msg: SignedImmediateSyftMessageWithoutReply,
        timeout: Optional[float] = None,
    ) -> None:
        """
        Sends high priority messages without waiting for their reply.

        This method implements a HTTP version of the
        ClientConnection.send_immediate_msg_without_reply

        """
        # Serializes SignedImmediateSyftMessageWithoutReply
        # and send it using HTTP protocol
        self._send_msg(msg=msg, timeout=timeout)

    def send_eventual_msg_without_reply(
        self,
        msg: SignedEventualSyftMessageWithoutReply,
        timeout: Optional[float] = None,
    ) -> None:
        """
        Sends low priority messages without waiting for their reply.

        This method implements a HTTP version of the
        ClientConnection.send_eventual_msg_without_reply
        """
        # Serializes SignedEventualSyftMessageWithoutReply in json format
        # and send it using HTTP protocol
        self._send_msg(msg=msg, timeout=timeout)

    def _send_msg(
        self, msg: SyftMessage, timeout: Optional[float] = None
    ) -> requests.Response:
        """
        Serializes Syft messages in json format and send it using HTTP protocol.

        NOTE: Auxiliary method to avoid code duplication and modularity.

        :return: returns requests.Response object containing a JSON serialized
        SyftMessage
        :rtype: requests.Response
        """

        # timeout = None will wait forever
        timeout = timeout if timeout is not None else DEFAULT_TIMEOUT

        # Perform HTTP request using base_url as a root address
        data_bytes: bytes = _serialize(msg, to_bytes=True)  # type: ignore
        r = requests.post(
            url=str(self.base_url),
            data=data_bytes,
            headers={"Content-Type": "application/octet-stream"},
            timeout=timeout,
            proxies=HTTPConnection.proxies,
        )

        # Return request's response object
        # r.text provides the response body as a str
        return r

    def _get_metadata(self) -> Metadata_PB:
        """
        Request Node's metadata

        :return: returns node metadata
        :rtype: str of bytes
        """
        data: bytes = requests.get(
            str(self.base_url) + "/metadata", timeout=1, proxies=HTTPConnection.proxies
        ).content
        metadata_pb = Metadata_PB()
        metadata_pb.ParseFromString(data)
        return metadata_pb
