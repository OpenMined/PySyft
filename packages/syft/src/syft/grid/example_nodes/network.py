"""
The purpose of this application is to allow us to dev and test PySyft
functionality on an actual local network. This is NOT meant to be run in
production (that's the *actual* grid's job).
"""
# third party
import flask
from flask import Flask
from flask import Response

# syft absolute
from syft import serialize
from syft.core.common.message import SignedImmediateSyftMessageWithReply
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply
from syft.core.common.serde.deserialize import _deserialize
from syft.core.node.network.network import Network
from syft.grid.services.signaling_service import PullSignalingService
from syft.grid.services.signaling_service import PushSignalingService
from syft.grid.services.signaling_service import RegisterDuetPeerService
from syft.logger import info


def signaling_server(port: int = 5000, host: str = "127.0.0.1") -> None:  # nosec
    app = Flask(__name__)

    network = Network(name="om-net")
    network.immediate_services_without_reply.append(PushSignalingService)
    network.immediate_services_with_reply.append(PullSignalingService)
    network.immediate_services_with_reply.append(RegisterDuetPeerService)
    network._register_services()  # re-register all services including SignalingService

    @app.route("/metadata")
    def get_metadata() -> flask.Response:
        metadata = network.get_metadata_for_client()
        metadata_proto = serialize(metadata)
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
            info(
                f"Signaling server SignedImmediateSyftMessageWithReply: {obj_msg.message} watch"
            )
            reply = network.recv_immediate_msg_with_reply(msg=obj_msg)
            r = Response(response=serialize(reply, to_bytes=True), status=200)
            r.headers["Content-Type"] = "application/octet-stream"
            return r
        elif isinstance(obj_msg, SignedImmediateSyftMessageWithoutReply):
            info(
                f"Signaling server SignedImmediateSyftMessageWithoutReply: {obj_msg.message} watch"
            )
            network.recv_immediate_msg_without_reply(msg=obj_msg)
            r = Response(status=200)
            return r
        else:
            info(
                f"Signaling server SignedImmediateSyftMessageWithoutReply: {obj_msg.message} watch"
            )
            network.recv_eventual_msg_without_reply(msg=obj_msg)
            r = Response(status=200)
            return r

    app.run(host=host, port=port)
