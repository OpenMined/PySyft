from flask import Flask
from ...core.node.domain import Domain
from ...core.node.domain import DomainClient
from flask import request
from syft.core.common.message import ImmediateSyftMessageWithReply
from syft.core.common.message import ImmediateSyftMessageWithoutReply
from .server import ServerThread
import sys

import binascii
import json
import pickle

import requests

from syft.core.io.connection import ClientConnection
from syft.core.io.route import SoloRoute
import time


class GridHttpClientConnection(ClientConnection):
    def __init__(self, base_url):
        self.base_url = base_url

    def send_immediate_msg_with_reply(self, msg):
        reply = self.send_msg(msg)
        binary = binascii.unhexlify(json.loads(reply.text)["data"])
        return pickle.loads(binary)  # nosec # TODO make less insecure

    def send_immediate_msg_without_reply(self, msg):
        self.send_msg(msg)

    def send_eventual_msg_without_reply(self, msg):
        self.send_msg(msg)

    def send_msg(self, msg):
        data = pickle.dumps(msg).hex()
        r = requests.post(url=self.base_url + "recv", json={"data": data})
        return r


class Duet(DomainClient):
    def __init__(self, host="127.0.0.1", port=5000):
        pub, pri = self.start_server(host=host, port=port)

        self.private_key = pri
        self.public_key = pub

        domain_url = "http://" + host + ":" + str(port) + "/"

        print("♫♫♫ > URL:{domain_url}")

        time.sleep(0.5)

        print("♫♫♫ > Connecting...")

        time.sleep(0.5)

        address, name, route = self.get_client_params(domain_url=domain_url)

        super().__init__(address=address, name=name, routes=[route])

        print("♫♫♫ > Connected!")

    @property
    def id(self):
        return self.domain_id

    def get_client_params(self, domain_url):

        binary = binascii.unhexlify(requests.get(domain_url).text)
        client_metadata = pickle.loads(binary)  # nosec # TODO make less insecure

        conn = GridHttpClientConnection(base_url=domain_url)
        address = client_metadata["address"]
        name = client_metadata["name"]
        id = client_metadata["id"]
        route = SoloRoute(source=None, destination=id, connection=conn)
        return address, name, route

    def start_server(self, host, port):

        app = Flask(__name__)

        pub, pri = Domain.keygen()

        domain = Domain(name="duet", root_public_key=pub)

        @app.route("/")
        def get_client():
            client_metadata = domain.get_metadata_for_client()
            return pickle.dumps(client_metadata).hex()

        @app.route("/" + str(domain.id.value), methods=["POST"])
        def recv():
            hex_msg = request.get_json()["data"]
            msg = pickle.loads(
                binascii.unhexlify(hex_msg)
            )  # nosec # TODO make less insecure
            reply = None
            print(str(msg))
            if isinstance(msg, ImmediateSyftMessageWithReply):
                reply = domain.recv_immediate_msg_with_reply(msg=msg)
                return {"data": pickle.dumps(reply).hex()}
            elif isinstance(msg, ImmediateSyftMessageWithoutReply):
                domain.recv_immediate_msg_without_reply(msg=msg)
            else:
                domain.recv_eventual_msg_without_reply(msg=msg)

            return str(msg)

        ...
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
        print(f"♫♫♫ > ID:{domain.id.value}")

    def stop(self):
        self.__del__()

    def __del__(self):
        print("♫♫♫ > Ending duet...")
        self.server.shutdown()
        print("♫♫♫ > Ended!")
