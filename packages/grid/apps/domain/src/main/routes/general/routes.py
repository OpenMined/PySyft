# stdlib
import json

# third party
from flask import Response
from flask import request
from nacl.encoding import HexEncoder
from syft import deserialize
from syft import serialize
from syft.core.common.message import SignedImmediateSyftMessageWithReply

# syft absolute
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply
from syft.core.common.serde.deserialize import _deserialize
from syft.core.common.serde.serialize import _serialize
from syft import SyModule

# grid relative
from ...core.exceptions import AuthorizationError
from ...core.exceptions import UserNotFoundError
from ...core.database.utils import model_to_json
from ...core.dataset_ops import get_all_datasets
from ...core.dataset_ops import get_all_relations

from ..auth import token_required
from .blueprint import root_blueprint as root_route
import torch as th


@root_route.route("/object-types", methods=["GET"])
def obj_types_route():
    # grid relative
    from ...core.node import get_node  # Tech debt: fix circular import

    # Tech debt : Dummy Approach for PoC/tests purposes
    # TODO: Replace const string values by python structures / abstractions
    return Response(
        json.dumps(["tensor", "dataset", "model"]),
        status=200,
        mimetype="application/json",
    )


@root_route.route("/find", methods=["GET"])
def find_obj_types():
    from ...core.node import get_node  # Tech debt: fix circular import

    obj_type = request.args.get("obj")

    # Tech debt : Dummy Approach for PoC/tests purposes
    # TODO: Replace const string fields by python structures / abstractions

    result = []
    if obj_type == "tensor":
        tensors = get_node().store.get_objects_of_type(obj_type=th.Tensor)

        for tensor in tensors:
            result.append(
                {
                    "id": str(tensor.id.value),
                    "tags": tensor.tags,
                    "description": tensor.description,
                }
            )
    elif obj_type == "dataset":
        for dataset in get_all_datasets():
            ds = model_to_json(dataset)
            objs = get_all_relations(dataset.id)
            ds["data"] = [
                {
                    "name": obj.name,
                    "id": obj.obj,
                    "dtype": obj.dtype,
                    "shape": obj.shape,
                }
                for obj in objs
            ]
            result.append(ds)
    elif obj_type == "model":
        models = get_node().store.get_objects_of_type(obj_type=th.nn.Module)

        for model in models:
            result.append(
                {
                    "id": str(model.id.value),
                    "tags": model.tags,
                    "description": model.description,
                }
            )
    return Response(json.dumps(result), status=200, mimetype="application/json")


@root_route.route("/dashboard", methods=["GET"])
@token_required
def dashboard_route(current_user):
    # grid relative
    from ...core.node import get_node  # TODO: fix circular import

    # Check if current session is the owner user
    _allowed = current_user.role == get_node().roles.owner_role.id
    if not _allowed:
        Response(
            {"error": str(AuthorizationError)}, status=403, mimetype="application/json"
        )
    else:
        response_body = {
            "datasets": len(get_node().disk_store),
            "requests": len(get_node().requests),
            "tensors": len(get_node().store),
            "common_users": len(get_node().users.common_users),
            "org_users": len(get_node().users.org_users),
            "groups": len(get_node().groups),
            "roles": len(get_node().roles),
            "models": len(get_node().store.get_objects_of_type(obj_type=th.nn.Module)),
        }
        return Response(
            json.dumps(response_body), status=200, mimetype="application/json"
        )


@root_route.route("/metadata", methods=["GET"])
def metadata_route():
    # grid relative
    from ...core.node import get_node  # TODO: fix circular import

    response_body = {
        "metadata": serialize(get_node().get_metadata_for_client())
        .SerializeToString()
        .decode("ISO-8859-1")
    }
    return Response(json.dumps(response_body), status=200, mimetype="application/json")


@root_route.route("/pysyft", methods=["POST"])
def syft_route():
    # grid relative
    from ...core.node import get_node  # TODO: fix circular import

    data = request.get_data()
    obj_msg = deserialize(blob=data, from_bytes=True)
    if isinstance(obj_msg, SignedImmediateSyftMessageWithReply):
        reply = get_node().recv_immediate_msg_with_reply(msg=obj_msg)
        r = Response(response=_serialize(obj=reply, to_bytes=True), status=200)
        r.headers["Content-Type"] = "application/octet-stream"
        return r
    elif isinstance(obj_msg, SignedImmediateSyftMessageWithoutReply):
        get_node().recv_immediate_msg_without_reply(msg=obj_msg)
    else:
        get_node().recv_eventual_msg_without_reply(msg=obj_msg)
    return ""


@root_route.route("/pysyft_multipart", methods=["POST"])
def syft_multipart_route():
    # grid relative
    from ...core.node import get_node  # TODO: fix circular import

    if "file" not in request.files:
        response = {"error": "Invalid message!"}
        status_code = 403

    file_obj = request.files["file"]

    msg = file_obj.stream.read()

    obj_msg = deserialize(blob=msg, from_bytes=True)

    if isinstance(obj_msg, SignedImmediateSyftMessageWithReply):
        reply = get_node().recv_immediate_msg_with_reply(msg=obj_msg)
        r = Response(response=_serialize(obj=reply, to_bytes=True), status=200)
        r.headers["Content-Type"] = "application/octet-stream"
        del msg
        del obj_msg
        return r
    elif isinstance(obj_msg, SignedImmediateSyftMessageWithoutReply):
        get_node().recv_immediate_msg_without_reply(msg=obj_msg)
    else:
        get_node().recv_eventual_msg_without_reply(msg=obj_msg)

    return ""
