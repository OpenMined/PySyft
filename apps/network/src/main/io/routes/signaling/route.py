from flask import Blueprint, request, Response

from ....core.signaling import signaling_handler

signaling = Blueprint("signaling", __name__)


@signaling.route("/webrtc-msg", methods=["POST"])
def save_webrtc_msg():
    msg = request.get_json()
    return signaling_handler.store_msg(msg=msg)


@signaling.route("/offer", methods=["GET"])
def consume_offer():
    try:
        msg = request.get_json()
        return signaling_handler.consume_offer(addr=msg)
    except Exception:
        return Response(status=400, mimetype="application/json")


@signaling.route("/answer", methods=["GET"])
def consume_answer():
    try:
        msg = request.get_json()
        return signaling_handler.consume_answer(addr=msg)
    except Exception:
        return Response(status=400, mimetype="application/json")
