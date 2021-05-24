"""The purpose of this application is to allow us to dev and test PySyft
functionality on an actual local network. This is NOT meant to be run in
production (that's the *actual* grid's job)."""


# stdlib
import binascii
import json
import pickle  # nosec

# third party
from flask import Flask
from flask import request

# syft absolute
from syft.core.common.message import ImmediateSyftMessageWithReply
from syft.core.common.message import ImmediateSyftMessageWithoutReply
from syft.core.node.domain.domain import Domain
from syft.logger import critical

app = Flask(__name__)


domain = Domain(name="ucsf")


@app.route("/")
def get_client() -> str:
    client_metadata = domain.get_metadata_for_client()
    return pickle.dumps(client_metadata).hex()


@app.route("/recv", methods=["POST"])
def recv() -> str:
    hex_msg = request.get_json()["data"]
    msg = pickle.loads(binascii.unhexlify(hex_msg))  # nosec # TODO make less insecure
    reply = None
    critical(str(msg))
    if isinstance(msg, ImmediateSyftMessageWithReply):
        reply = domain.recv_immediate_msg_with_reply(msg=msg)
        # QUESTION: is this expected to be a json string with the top level key data =>
        return json.dumps({"data": pickle.dumps(reply).hex()})
    elif isinstance(msg, ImmediateSyftMessageWithoutReply):
        domain.recv_immediate_msg_without_reply(msg=msg)
    else:
        domain.recv_eventual_msg_without_reply(msg=msg)

    return str(msg)


def run() -> None:
    app.run()


run()
