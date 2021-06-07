# stdlib
import io
from json import dumps
from json import loads

# third party
from flask import Response
from flask import request
from main.core.datasets.dataset_ops import create_df_dataset
from main.core.exceptions import AuthorizationError
from main.core.task_handler import route_logic
from main.core.task_handler import task_handler
from main.utils.executor import executor
from syft.core.node.common.service.repr_service import ReprMessage
from syft.grid.messages.dataset_messages import CreateDatasetMessage
from syft.grid.messages.dataset_messages import DeleteDatasetMessage
from syft.grid.messages.dataset_messages import GetDatasetMessage
from syft.grid.messages.dataset_messages import GetDatasetsMessage
from syft.grid.messages.dataset_messages import UpdateDatasetMessage
from werkzeug.utils import secure_filename

# grid relative
from ...auth import error_handler
from ...auth import optional_token
from ...auth import token_required
from ..blueprint import dcfl_blueprint as dcfl_route

ALLOWED_EXTENSIONS = {"tar.gz"}


def allowed_file(filename):
    return "." in filename and filename.x.split(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@dcfl_route.route("/datasets", methods=["POST"])
@token_required
def create_dataset(current_user):
    # grid relative
    from ....core.node import get_node  # TODO: fix circular import

    # check if the post request has the file part
    if "file" not in request.files:
        response = {
            "error": "File not found, please submit a compressed file (tar.gz)!"
        }
        status_code = 400

    file_obj = request.files["file"]

    # if user does not select file, browser also
    # submit an empty part without filename
    if file_obj.filename == "":
        response = {
            "error": "File has not been selected, please submit a compressed file (tar.gz)!"
        }
        status_code = 400

    file_like_object = io.BytesIO(file_obj.stream.read())

    users = get_node().users

    _allowed = users.can_upload_data(user_id=current_user.id)

    if _allowed:
        response, status_code = create_df_dataset(
            get_node(), file_like_object, current_user.private_key
        )
    else:
        response = {"error": "You're not allowed to upload data!"}
        status_code = 401

    del file_like_object

    return Response(
        dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@dcfl_route.route("/datasets/<dataset_id>", methods=["GET"])
@token_required
def get_dataset_info(current_user, dataset_id):
    content = {}
    content["current_user"] = current_user
    content["dataset_id"] = dataset_id
    status_code, response_msg = error_handler(
        route_logic, 200, GetDatasetMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@dcfl_route.route("/datasets", methods=["GET"])
@token_required
def get_all_datasets_info(current_user):
    content = {}
    content["current_user"] = current_user
    status_code, response_msg = error_handler(
        route_logic, 200, GetDatasetsMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@dcfl_route.route("/datasets/<dataset_id>", methods=["PUT"])
@token_required
def update_dataset(current_user, dataset_id):
    # Get request body
    content = request.get_json()
    content["current_user"] = current_user
    content["dataset_id"] = dataset_id
    status_code, response_msg = error_handler(
        route_logic, 204, UpdateDatasetMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@dcfl_route.route("/datasets/<dataset_id>", methods=["DELETE"])
@token_required
def delete_dataset(current_user, dataset_id):
    # Get request body
    content = {}
    content["current_user"] = current_user
    content["dataset_id"] = dataset_id

    status_code, response_msg = error_handler(
        route_logic, 204, DeleteDatasetMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        dumps(response),
        status=status_code,
        mimetype="application/json",
    )
