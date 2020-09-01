from flask import Blueprint, request, Response
from ....core import node

common = Blueprint("common", __name__)


@common.route("/metadata", methods=["GET"])
def get_metadata():
    return node.get_metadata_for_client()
