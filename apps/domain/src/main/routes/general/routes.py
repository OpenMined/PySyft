from .blueprint import root_blueprint as root_route
from ...core.node import node
from ...core.task_handler import executor

# syft absolute
from syft.core.common.message import SignedImmediateSyftMessageWithReply
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply
from syft.core.common.serde.deserialize import _deserialize

from flask import request, Response
import json

executor_running = False


@root_route.route("/pysyft", methods=["POST"])
def root_route():
    global executor_running
    if not executor_running:
        executor.submit(node.run_handlers_thread)
        executor_running = True

    data = request.get_data()
    obj_msg = _deserialize(blob=data, from_bytes=True)
    if isinstance(obj_msg, SignedImmediateSyftMessageWithReply):
        reply = node.recv_immediate_msg_with_reply(msg=obj_msg)
        r = Response(response=reply.serialize(to_bytes=True), status=200)
        r.headers["Content-Type"] = "application/octet-stream"
        return r
    elif isinstance(obj_msg, SignedImmediateSyftMessageWithoutReply):
        node.recv_immediate_msg_without_reply(msg=obj_msg)
    else:
        node.recv_eventual_msg_without_reply(msg=obj_msg)
    return ""
