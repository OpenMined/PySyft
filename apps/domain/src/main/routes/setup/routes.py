from .blueprint import setup_blueprint as setup_route
from flask import request, Response
import json
from syft.core.node.common.service.repr_service import ReprMessage

from ..auth import error_handler, token_required
from ...core.node import node


@setup_route.route("/", methods=["POST"])
#@token_required
def initial_setup():
    def route_logic():
        # Get request body
        content = loads(request.data)

        syft_message = {}
        syft_message["message_class"] = ReprMessage  # TODO: InitialSetupSettingsMessage
        syft_message["message_content"] = content
        syft_message[
            "sign_key"
        ] = node.signing_key  # TODO: Method to map token into sign-key

        # Execute task
        status_code, response_body = task_handler(
            route_function=process_as_syft_message,
            data=syft_message,
            mandatory={
                "message_class": MissingRequestKeyError,
                "message_content": MissingRequestKeyError,
                "sign_key": MissingRequestKeyError,
            },
        )
        return response_body

    #status_code, response_body = error_handler(process_as_syft_message)
    status_code, response_body = 200, {"msg": "Running initial setup!"}

    return Response(
        json.dumps(response_body), status=status_code, mimetype="application/json"
    )


@setup_route.route("/", methods=["GET"])
#@token_required
def get_setup():
    def route_logic():
        # Get request body
        content = loads(request.data)

        syft_message = {}
        syft_message["message_class"] = ReprMessage  # TODO: Get Initial Setup Messages
        syft_message["message_content"] = content
        syft_message[
            "sign_key"
        ] = node.signing_key  # TODO: Method to map token into sign-key

        # Execute task
        status_code, response_body = task_handler(
            route_function=process_as_syft_message,
            data=syft_message,
            mandatory={
                "message_class": MissingRequestKeyError,
                "message_content": MissingRequestKeyError,
                "sign_key": MissingRequestKeyError,
            },
        )
        return response_body

    #status_code, response_body = error_handler(process_as_syft_message)
    status_code, response_body = 200, {"setup": {}}

    return Response(
        json.dumps(response_body), status=status_code, mimetype="application/json"
    )
