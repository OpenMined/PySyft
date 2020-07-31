"""The purpose of this application is to allow us to dev and test PySyft
functionality on an actual local network. This is NOT meant to be run in
production (that's the *actual* grid's job)."""

from flask import Flask
from flask import request
import pickle

app = Flask(__name__)

import binascii
from syft.core.nodes.domain.domain import Domain
from syft.core.message import ImmediateSyftMessageWithoutReply
from syft.core.message import ImmediateSyftMessageWithReply
from syft.core.message import EventualSyftMessageWithoutReply

domain = Domain(name="ucsf")


@app.route("/")
def get_client():

    client_metadata = domain.get_metadata_for_client()
    return pickle.dumps(client_metadata).hex()


@app.route("/recv", methods=["POST"])
def recv():
    hex_msg = request.get_json()["data"]
    msg = pickle.loads(binascii.unhexlify(hex_msg))
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


def run():
    app.run()
