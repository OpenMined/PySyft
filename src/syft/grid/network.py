"""The purpose of this application is to allow us to dev and test PySyft
functionality on an actual local network. This is NOT meant to be run in
production (that's the *actual* grid's job)."""


# third party
from flask import Flask
from flask import request

# syft absolute
from syft.core.common.message import SignedImmediateSyftMessageWithReply
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply
from syft.core.common.serde.deserialize import _deserialize
from syft.core.node.network.network import Network
from syft.grid.services.signaling_service import PullSignalingService
from syft.grid.services.signaling_service import PushSignalingService


app = Flask(__name__)

network = Network(name="om-net")

network.immediate_services_without_reply.append(PushSignalingService)
network.immediate_services_with_reply.append(PullSignalingService)
network._register_services()  # re-register all services including SignalingService


@app.route("/metadata")
def get_metadata() -> str:
    return network.get_metadata_for_client()


@app.route("/", methods=["POST"])
def process_network_msgs():
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
    app.run()

