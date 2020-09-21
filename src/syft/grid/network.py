"""The purpose of this application is to allow us to dev and test PySyft
functionality on an actual local network. This is NOT meant to be run in
production (that's the *actual* grid's job)."""


# third party
from flask import Flask
from flask import request
from nacl.encoding import HexEncoder

# syft relative
from ..core.common.message import SignedImmediateSyftMessageWithReply

# relative absolute
from ..core.common.message import SignedImmediateSyftMessageWithoutReply
from ..core.common.serde.deserialize import _deserialize
from ..core.node.network.network import Network
from ..grid.services.signaling_service import PullSignalingService
from ..grid.services.signaling_service import PushSignalingService

app = Flask(__name__)

network = Network(name="om-net")

network.immediate_services_without_reply.append(PushSignalingService)
network.immediate_services_with_reply.append(PullSignalingService)
network._register_services()  # re-register all services including SignalingService


@app.route("/metadata")
def get_metadata() -> str:
    return network.get_metadata_for_client()


@app.route("/", methods=["POST"])
def process_network_msgs() -> str:
    json_msg = request.get_json()
    obj_msg = _deserialize(blob=json_msg, from_json=True)
    if isinstance(obj_msg, SignedImmediateSyftMessageWithReply):
        reply = network.recv_immediate_msg_with_reply(msg=obj_msg)
        return reply.json()
    elif isinstance(obj_msg, SignedImmediateSyftMessageWithoutReply):
        network.recv_immediate_msg_without_reply(msg=obj_msg)
    else:
        network.recv_eventual_msg_without_reply(msg=obj_msg)
    return ""


def run() -> None:
    global network
    print("====================================")
    print("========== NODE ROOT KEY ===========")
    print("====================================")
    # this signing_key is to aid in local development and is not used in the real
    # PyGrid implementation
    print(network.signing_key.encode(encoder=HexEncoder).decode("utf-8"), "\n")
    app.run()
