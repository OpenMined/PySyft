from flask import Blueprint
from flask import request

from ....core.data_centric import duet_handler

duet = Blueprint("duet", __name__)


@duet.route("/metadata", methods=["GET"])
def get_metadata():
    return duet_handler.register()


@duet.route("/webrtc-msg", methods=["POST"])
def save_webrtc_msg():
    msg = request.get_json()
    return duet_handler.store_msg(msg=msg)


@duet.route("/offer", methods=["GET"])
def consume_offer():
    msg = request.get_json()
    return duet_handler.consume_offer(msg=msg)


@duet.route("/answer", methods=["GET"])
def consume_offer():
    msg = request.get_json()
    return duet_handler.consume_answer(msg=msg)
