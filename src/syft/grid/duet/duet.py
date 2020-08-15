from typing import Tuple, Optional
from flask import Flask

from nacl.signing import SigningKey, VerifyKey

from ...core.node.domain import Domain
from ...core.node.domain import DomainClient

from flask import request
from syft.core.common.uid import UID
from syft.core.io.address import Address
from syft.core.io.location import SpecificLocation
from syft.core.io.route import Route
from .server import ServerThread
import sys

import requests

from syft.core.io.connection import ClientConnection
from syft.core.io.route import SoloRoute
import time

import syft as sy

from syft.core.common.message import (
    SyftMessage,
    SignedImmediateSyftMessageWithoutReply,
    SignedImmediateSyftMessageWithReply,
)


class GridHttpClientConnection(ClientConnection):
    def __init__(self, base_url: str, domain_id: UID) -> None:
        self.base_url = base_url
        self.domain_id = domain_id

    def send_immediate_msg_with_reply(
        self, msg: SignedImmediateSyftMessageWithReply
    ) -> requests.Response:
        blob = self.send_msg(msg).text
        response = sy.deserialize(blob=blob, from_json=True)
        return response

    def send_immediate_msg_without_reply(
        self, msg: SignedImmediateSyftMessageWithoutReply
    ) -> requests.Response:
        return self.send_msg(msg)

    def send_eventual_msg_without_reply(
        self, msg: SignedImmediateSyftMessageWithoutReply
    ) -> requests.Response:
        return self.send_msg(msg)

    def send_msg(self, msg: SyftMessage) -> requests.Response:
        json_msg = msg.json()
        r = requests.post(url=self.base_url + str(self.domain_id.value), json=json_msg)
        return r


class Duet(DomainClient):
    def __init__(
        self, host: str = "127.0.0.1", port: int = 5000, id: Optional[str] = None
    ) -> None:
        domain_url = "http://" + host + ":" + str(port) + "/"

        # generate a signing key
        self.signing_key = SigningKey.generate()
        self.verify_key = self.signing_key.verify_key

        # start a node on host and port
        if id is None:
            # set the pub_key as the root_key of the server so this user will be root
            self.start_server(host=host, port=port, root_key=self.verify_key)
            print(f"♫♫♫ > URL:{domain_url}")

            time.sleep(0.5)
            print("♫♫♫ > Connecting...")

            time.sleep(0.5)

        address, name, route = self.get_client_params(domain_url=domain_url)

        # dont start a node but connect to it instead
        if id is not None:
            address = SpecificLocation(id=UID.from_string(value=id))

        super().__init__(domain=address, name=name, routes=[route])
        print("♫♫♫ > Connected!")

    def send_signed(self) -> None:
        self.send_immediate_msg_without_reply(msg=sy.ReprMessage(address=self.domain))

    @property
    def id(self) -> UID:
        return self.target_id

    def get_client_params(self, domain_url: str) -> Tuple[Address, str, Route]:
        text = requests.get(domain_url).text
        address, name, client_id = DomainClient.deserialize_client_metadata_from_node(
            metadata=text
        )
        conn = GridHttpClientConnection(base_url=domain_url, domain_id=client_id)
        route = SoloRoute(destination=client_id, connection=conn)
        return address, name, route

    def start_server(self, host: str, port: int, root_key: VerifyKey) -> None:
        app = Flask(__name__)
        domain = Domain(name="duet", root_key=root_key)
        print(f"Domain with Root Key: {domain.root_key}")

        @app.route("/")
        def get_client() -> str:  # pylint: disable=unused-variable
            return domain.get_metadata_for_client()

        @app.route("/" + str(domain.id.value), methods=["POST"])
        def recv() -> str:  # pylint: disable=unused-variable
            json_msg = request.get_json()
            msg = sy.deserialize(blob=json_msg, from_json=True)
            if isinstance(msg, SignedImmediateSyftMessageWithReply):
                reply = domain.recv_immediate_msg_with_reply(msg=msg)
                return reply.json()
            elif isinstance(msg, SignedImmediateSyftMessageWithoutReply):
                domain.recv_immediate_msg_without_reply(msg=msg)
            else:
                domain.recv_eventual_msg_without_reply(msg=msg)

            return str(msg)

        self.server = ServerThread(app, host=host, port=port)
        self.server.start()
        print("🎤  🎸  ♪♪♪ duet started ♫♫♫  🎻  🎹\n")
        sys.stdout.write(
            "♫♫♫ >\033[93m" + " DISCLAIMER" + "\033[0m"
            ":"
            + "\033[1m"
            + " Duet is an experimental feature currently \n♫♫♫ > "
            + "in alpha. Do not use this to protect real-world data.\n"
            + "\033[0m"
        )
        print("♫♫♫ >")
        print(f"♫♫♫ > Duet Node ID:{domain.id.value}")

    def stop(self) -> None:
        self.__del__()

    def __del__(self) -> None:
        print("♫♫♫ > Ending duet...")
        self.server.shutdown()
        print("♫♫♫ > Ended!")
