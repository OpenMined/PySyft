from .blueprint import root_blueprint as root_route
from ...core.node import node

# syft absolute
from syft.core.common.message import SignedImmediateSyftMessageWithReply
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply
from syft.core.common.serde.deserialize import _deserialize

from flask import request, Response
import json


@root_route.route("/pysyft", methods=["POST"])
def root_route():
    json_msg = request.get_json()
    obj_msg = _deserialize(blob=json_msg, from_json=True)
    if isinstance(obj_msg, SignedImmediateSyftMessageWithReply):
        reply = node.recv_immediate_msg_with_reply(msg=obj_msg)
        return reply.json()
    elif isinstance(obj_msg, SignedImmediateSyftMessageWithoutReply):
        node.recv_immediate_msg_without_reply(msg=obj_msg)
    else:
        node.recv_eventual_msg_without_reply(msg=obj_msg)
    return ""
