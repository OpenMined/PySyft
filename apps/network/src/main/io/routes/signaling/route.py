from flask import Blueprint, request, Response

from ....core.signaling import signaling_handler
from syft.core.common.serde.deserialize import _deserialize
from syft.core.common.serde.serialize import _serialize

signaling = Blueprint("signaling", __name__)


@signaling.route("/push", methods=["POST"])
def save_webrtc_msg():
    try:
        msg = request.get_json()

        msg = _deserialize(blob=msg, from_json=True)

        response = signaling_handler.push(msg=msg)

        if response:
            response = response.json()

        return Response(status=200, response=response, mimetype="application/json")
    except Exception as e:
        print("Exception: ", str(e))
        return Response(status=400, mimetype="application/json")


@signaling.route("/pull", methods=["POST"])
def consume_offer():
    try:
        msg = request.get_json()

        msg = _deserialize(blob=msg, from_json=True)

        response = signaling_handler.pull(msg=msg)

        if response:
            response = response.json()

        return Response(status=200, response=response, mimetype="application/json")
    except Exception:
        print("Exception: ".str(e))
        return Response(status=400, mimetype="application/json")
