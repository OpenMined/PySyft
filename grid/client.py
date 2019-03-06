from websocket import create_connection
from syft.workers import BaseWorker
import binascii


class WebsocketClientWorker(BaseWorker):
    def __init__(
        self, hook, host, port, id=0, is_client_worker=False, log_msgs=False, verbose=False, data={}
    ):
        """A client which will forward all messages to a remote worker running a
        WebsocketServerWorker and receive all responses back from the server.
        """

        # TODO get angry when we have no connection params
        self.port = port
        self.host = host
        self.uri = f"ws://{self.host}:{self.port}"

        # creates the connection with the server which gets held open until the
        # WebsocketClientWorker is garbage collected.
        self.ws = create_connection(self.uri)

        super().__init__(hook, id, data, is_client_worker, log_msgs, verbose)

    def _send_msg(self, message: bin) -> bin:
        raise RuntimeError(
            "_send_msg should never get called on a ",
            "WebsocketClientWorker. Did you accidentally "
            "make hook.local_worker a WebsocketClientWorker?",
        )

    def _recv_msg(self, message: bin) -> bin:
        """Forwards a message to the WebsocketServerWorker"""

        self.ws.send(str(binascii.hexlify(message)))
        response = binascii.unhexlify(self.ws.recv()[2:-1])
        return response
