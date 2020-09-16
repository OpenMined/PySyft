# third party
import requests

# syft absolute
import syft as sy
from syft.core.common.message import SignedImmediateSyftMessageWithReply
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply
from syft.core.common.message import SyftMessage
from syft.core.io.connection import ClientConnection

# syft relative
from ...decorators.syft_decorator_impl import syft_decorator


class HTTPConnection(ClientConnection):
    @syft_decorator(typechecking=True)
    def __init__(self, url: str) -> None:
        self.base_url = url

    @syft_decorator(typechecking=True)
    def send_immediate_msg_with_reply(
        self, msg: SignedImmediateSyftMessageWithReply
    ) -> SignedImmediateSyftMessageWithoutReply:
        """Sends high priority messages and wait for their responses.

        NOTE: Mandatory abstract method.

        This method aims to implement
        a HTTP version of the
        ClientConnection.send_immediate_msg_with_reply

        :return: returns an instance of SignedImmediateSyftMessageWithReply.
        :rtype: SignedImmediateSyftMessageWithReply
        """

        # Serializes SignedImmediateSyftMessageWithReply in json format
        # and send it using HTTP protocol
        blob = self.__send_msg(msg=msg).text

        # Deserialize node's response
        response = sy.deserialize(blob=blob, from_json=True)

        # Return SignedImmediateSyftMessageWithoutReply
        return response

    @syft_decorator(typechecking=True)
    def send_immediate_msg_without_reply(
        self, msg: SignedImmediateSyftMessageWithoutReply
    ) -> None:
        """Sends high priority messages without waiting for their reply.

        NOTE: Mandatory abstract method.

        This method aims to implement
        a HTTP version of the
        ClientConnection.send_immediate_msg_without_reply

        """
        # Serializes SignedImmediateSyftMessageWithReply in json format
        # and send it using HTTP protocol
        self.__send_msg(msg=msg)

    @syft_decorator(typechecking=True)
    def send_eventual_msg_without_reply(
        self, msg: SignedImmediateSyftMessageWithoutReply
    ) -> None:
        """Sends low priority messages without waiting for their reply.

        NOTE: Mandatory abstract method.

        This method aims to implement
        a HTTP version of the
        ClientConnection.send_eventual_msg_without_reply
        """
        # Serializes SignedImmediateSyftMessageWithReply in json format
        # and send it using HTTP protocol
        self.__send_msg(msg=msg)

    @syft_decorator(typechecking=True)
    def __send_msg(self, msg: SyftMessage) -> requests.Response:
        """Serializes Syft messages in json format and send it using HTTP protocol.

        NOTE: Auxiliar method to provide code modularity and reuse
        This will avoid code duplication.

        :return: returns response body in string format (serialized syft message).
        :rtype: str
        """
        # Serialize SyftMessage object
        json_msg = msg.json()

        # Perform HTTP request using base_url as a root address
        r = requests.post(url=self.base_url, json=json_msg)

        # Return request's response body (serialized
        # syft msg sent by the other peer)
        return r

    @syft_decorator(typechecking=True)
    def _get_metadata(self) -> str:
        """Request Node's metadata

        :return: returns  node metadata.
        :rtype: str
        """
        return requests.get(self.base_url + "/metadata").text
