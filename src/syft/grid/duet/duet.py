from typing import Tuple
import json
from flask import Flask
from ...core.node.domain import Domain
from ...core.node.domain import DomainClient

from flask import request
from syft.core.common.uid import UID
from syft.core.io.address import Address
from syft.core.io.route import Route
from .server import ServerThread
import sys

import binascii
import pickle

import requests

from syft.core.io.connection import ClientConnection
from syft.core.io.route import SoloRoute
import time

import syft as sy

from syft.core.common.message import (
    SyftMessage,
    EventualSyftMessageWithoutReply,
    ImmediateSyftMessageWithoutReply,
    ImmediateSyftMessageWithReply,
)


class GridHttpClientConnection(ClientConnection):
    def __init__(self, base_url: str, domain_id: UID) -> None:
        self.base_url = base_url
        self.domain_id = domain_id

    def send_immediate_msg_with_reply(
        self, msg: ImmediateSyftMessageWithReply
    ) -> requests.Response:
        return self.send_msg(msg)

    def send_immediate_msg_without_reply(
        self, msg: ImmediateSyftMessageWithoutReply
    ) -> requests.Response:
        return self.send_msg(msg)

    def send_eventual_msg_without_reply(
        self, msg: EventualSyftMessageWithoutReply
    ) -> requests.Response:
        return self.send_msg(msg)

    def send_msg(self, msg: SyftMessage) -> requests.Response:
        json_msg = msg.json()
        print("trying to send message", self.base_url, msg)
        r = requests.post(url=self.base_url + str(self.domain_id.value), json=json_msg)
        return r


class Duet(DomainClient):
    def __init__(self, host: str = "127.0.0.1", port: int = 5000) -> None:
        domain_url = "http://" + host + ":" + str(port) + "/"
        self.start_server(host=host, port=port)
        print(f"♫♫♫ > URL:{domain_url}")

        time.sleep(0.5)
        print("♫♫♫ > Connecting...")

        time.sleep(0.5)
        address, name, route = self.get_client_params(domain_url=domain_url)
        super().__init__(domain=address, name=name, routes=[route])
        print("♫♫♫ > Connected!")

    @property
    def id(self) -> UID:
        return self.target_id

    def get_client_params(self, domain_url: str) -> Tuple[Address, str, Route]:
        text = requests.get(domain_url).text
        binary = binascii.unhexlify(text)
        client_metadata = pickle.loads(binary)  # nosec # TODO make less insecure
        address = client_metadata["address"]
        name = client_metadata["name"]
        client_id = client_metadata["id"]
        conn = GridHttpClientConnection(base_url=domain_url, domain_id=client_id)
        route = SoloRoute(destination=client_id, connection=conn)
        return address, name, route

    def start_server(self, host: str, port: int) -> None:
        app = Flask(__name__)
        domain = Domain(name="duet")

        @app.route("/")
        def get_client() -> str:  # pylint: disable=unused-variable
            client_metadata = domain.get_metadata_for_client()
            return pickle.dumps(client_metadata).hex()

        @app.route("/" + str(domain.id.value), methods=["POST"])
        def recv() -> str:  # pylint: disable=unused-variable
            json_msg = request.get_json()
            msg = sy.deserialize(blob=json_msg, from_json=True)
            reply = None
            if isinstance(msg, ImmediateSyftMessageWithReply):
                reply = domain.recv_immediate_msg_with_reply(msg=msg)
                return json.dumps({"data": pickle.dumps(reply).hex()})
            elif isinstance(msg, ImmediateSyftMessageWithoutReply):
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
