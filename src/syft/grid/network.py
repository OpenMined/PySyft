"""
The purpose of this application is to allow us to dev and test PySyft
functionality on an actual local network. This is NOT meant to be run in
production (that's the *actual* grid's job).

For example:
$ python src/syft/grid/example_nodes/network.py

"""

# third party
from flask import Flask
from flask import request
from nacl.encoding import HexEncoder

# syft absolute
import syft as sy
from syft.core.common.message import SignedImmediateSyftMessageWithReply
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply
from syft.core.common.serde.deserialize import _deserialize
from syft.core.node.network.network import Network
from syft.grid.services.signaling_service import PullSignalingService
from syft.grid.services.signaling_service import PushSignalingService
from syft.grid.services.signaling_service import RegisterDuetPeerService

sy.VERBOSE = False
app = Flask(__name__)

network = Network(name="om-net")

network.immediate_services_without_reply.append(PushSignalingService)
network.immediate_services_with_reply.append(PullSignalingService)
network.immediate_services_with_reply.append(RegisterDuetPeerService)
network._register_services()  # re-register all services including SignalingService


@app.route("/metadata")
def get_metadata() -> str:
    try:
        return network.get_metadata_for_client()
    except Exception:
        return ""


@app.route("/", methods=["POST"])
def process_network_msgs() -> str:
    try:
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
    except Exception:
        return ""


def run() -> None:
    global network
    print("====================================")
    print("========== NODE ROOT KEY ===========")
    print("====================================")
    # this signing_key is to aid in local development and is not used in the real
    # PyGrid implementation
    print(network.signing_key.encode(encoder=HexEncoder).decode("utf-8"), "\n")
    app.run(host="0.0.0.0", port=5000)  # nosec


run()
