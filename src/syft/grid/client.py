import binascii
import json
import pickle

import requests

import syft as sy
from syft.core.io.connection import ClientConnection
from syft.core.io.route import SoloRoute


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


def connect(domain_url="http://localhost:5000/"):
    binary = binascii.unhexlify(requests.get(domain_url).text)
    client_metadata = pickle.loads(binary)  # nosec # TODO make less insecure

    conn = GridHttpClientConnection(base_url=domain_url)
    address = client_metadata["address"]
    name = client_metadata["name"]
    id = client_metadata["id"]
    route = SoloRoute(destination=id, connection=conn)
    client = sy.DomainClient(address=address, name=name, routes=[route])
    return client
