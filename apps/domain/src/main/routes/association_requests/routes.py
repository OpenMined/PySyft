from .blueprint import association_requests_blueprint as association_request_route
from flask import request, Response
import json

from syft.core.node.common.service.repr_service import ReprMessage
from ..auth import error_handler, token_required

@association_request_route.route("/request", methods=["POST"])
@token_required
def send_association_request():
    def route_logic():
        # Get request body
        content = loads(request.data)

        syft_message = {}
        syft_message["message_class"] = ReprMessage # TODO: Send Association Request Message
        syft_message["message_content"] = content
        syft_message["sign_key"] =  # TODO: Method to map token into sign-key

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

    status_code, response_body = error_handler(process_as_syft_message)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@association_request_route.route("/receive", methods=["POST"])
@token_required
def recv_association_request():
    def route_logic():
        # Get request body
        content = loads(request.data)

        syft_message = {}
        syft_message["message_class"] = ReprMessage # TODO: Retrieve Association Request Messages
        syft_message["message_content"] = content
        syft_message["sign_key"] =  # TODO: Method to map token into sign-key

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

    status_code, response_body = error_handler(process_as_syft_message)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@association_request_route.route("/respond", methods=["POST"])
@token_required
def reply_association_request():
    def route_logic():
        # Get request body
        content = loads(request.data)

        syft_message = {}
        syft_message["message_class"] = ReprMessage # TODO: ReplyAssociationRequestMessage
        syft_message["message_content"] = content
        syft_message["sign_key"] =  # TODO: Method to map token into sign-key

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

    status_code, response_body = error_handler(process_as_syft_message)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@association_request_route.route("/", methods=["GET"])
@token_required
def get_all_association_requests():
    def route_logic():
        # Get request body
        content = loads(request.data)

        syft_message = {}
        syft_message["message_class"] = ReprMessage # TODO: Get All Association Request Message
        syft_message["message_content"] = content
        syft_message["sign_key"] =  # TODO: Method to map token into sign-key

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

    status_code, response_body = error_handler(process_as_syft_message)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )

@association_request_route.route("/<association_request_id>", methods=["GET"])
@token_required
def get_specific_association_requests(association_request_id):
    def route_logic():
        # Get request body
        content = loads(request.data)

        syft_message = {}
        syft_message["message_class"] = ReprMessage # TODO: Get Specific Association Request Message
        syft_message["message_content"] = content
        syft_message["sign_key"] =  # TODO: Method to map token into sign-key

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

    status_code, response_body = error_handler(process_as_syft_message)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )

@association_request_route.route("/<association_request_id>", methods=["DELETE"])
@token_required
def delete_association_requests(association_request_id):
    def route_logic():
        # Get request body
        content = loads(request.data)

        syft_message = {}
        syft_message["message_class"] = ReprMessage # TODO: Delete Association Request Messages
        syft_message["message_content"] = content
        syft_message["sign_key"] =  # TODO: Method to map token into sign-key

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

    status_code, response_body = error_handler(process_as_syft_message)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )