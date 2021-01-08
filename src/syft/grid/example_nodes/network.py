"""
The purpose of this application is to allow us to dev and test PySyft
functionality on an actual local network. This is NOT meant to be run in
production (that's the *actual* grid's job).

For example:
$ python src/syft/grid/example_nodes/network.py

"""
# stdlib
import os
import sys

# third party
import flask
from flask import Flask
from flask import Response
from nacl.encoding import HexEncoder

# syft absolute
from syft.core.common.message import SignedImmediateSyftMessageWithReply
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply
from syft.core.common.serde.deserialize import _deserialize
from syft.core.node.network.network import Network
from syft.grid.services.signaling_service import PullSignalingService
from syft.grid.services.signaling_service import PushSignalingService
from syft.grid.services.signaling_service import RegisterDuetPeerService

app = Flask(__name__)

network = Network(name="om-net")

network.immediate_services_without_reply.append(PushSignalingService)
network.immediate_services_with_reply.append(PullSignalingService)
network.immediate_services_with_reply.append(RegisterDuetPeerService)
network._register_services()  # re-register all services including SignalingService


@app.route("/metadata")
def get_metadata() -> flask.Response:
    metadata = network.get_metadata_for_client()
    metadata_proto = metadata.serialize()
    r = Response(
        response=metadata_proto.SerializeToString(),
        status=200,
    )
    r.headers["Content-Type"] = "application/octet-stream"
    return r


@app.route("/", methods=["POST"])
def process_network_msgs() -> flask.Response:
    data = flask.request.get_data()
    obj_msg = _deserialize(blob=data, from_bytes=True)
    if isinstance(obj_msg, SignedImmediateSyftMessageWithReply):
        print(
            f"Signaling server SignedImmediateSyftMessageWithReply: {obj_msg.message} watch"
        )
        reply = network.recv_immediate_msg_with_reply(msg=obj_msg)
        r = Response(response=reply.serialize(to_bytes=True), status=200)
        r.headers["Content-Type"] = "application/octet-stream"
        return r
    elif isinstance(obj_msg, SignedImmediateSyftMessageWithoutReply):
        print(
            f"Signaling server SignedImmediateSyftMessageWithoutReply: {obj_msg.message} watch"
        )
        network.recv_immediate_msg_without_reply(msg=obj_msg)
        r = Response(status=200)
        return r
    else:
        print(
            f"Signaling server SignedImmediateSyftMessageWithoutReply: {obj_msg.message} watch"
        )
        network.recv_eventual_msg_without_reply(msg=obj_msg)
        r = Response(status=200)
        return r


def run() -> None:
    global network

    num_args = len(sys.argv)
    if num_args > 1:
        if sys.argv[1] == "IPV6":
            os.environ["IP_MODE"] = "IPV6"
        else:
            os.environ["IP_MODE"] = "IPV4"
    else:
        os.environ["IP_MODE"] = "IPV4"

    print("====================================")
    print("========== NODE ROOT KEY ===========")
    print("====================================")
    # this signing_key is to aid in local development and is not used in the real
    # PyGrid implementation
    PORT = os.getenv("PORT", 5000)
    print(f"Starting Node on PORT: {PORT}")
    print(network.signing_key.encode(encoder=HexEncoder).decode("utf-8"), "\n")

    MODE = os.getenv("IP_MODE")
    if MODE == "IPV6":
        print(f"Listening on : {MODE}")
        # used :: so that the server works for IPV6
        app.run(host="::", port=int(PORT))
    else:
        print(f"Listening on : {MODE}")
        app.run(host="0.0.0.0", port=int(PORT))  # nosec


run()
