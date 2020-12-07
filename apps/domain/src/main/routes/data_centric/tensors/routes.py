from ..blueprint import dcfl_blueprint as dcfl_route
from flask import request, Response
import json


from syft.core.node.common.service.repr_service import ReprMessage
from ...auth import error_handler, token_required

@dcfl_route.route("/tensors", methods=["POST"])
@token_required
def create_tensor():
    def route_logic():
        # Get request body
        content = loads(request.data)

        syft_message = {}
        syft_message["message_class"] = ReprMessage  # TODO: CreateNewTensorMessage
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


@dcfl_route.route("/tensors/<tensor_id>", methods=["GET"])
@token_required
def get_tensor(tensor_id):
    def route_logic():
        # Get request body
        content = loads(request.data)

        syft_message = {}
        syft_message["message_class"] = ReprMessage # TODO: GetTensorMessage
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

@dcfl_route.route("/tensors", methods=["GET"])
@token_required
def get_all_tensors():
    def route_logic():
        # Get request body
        content = loads(request.data)

        syft_message = {}
        syft_message["message_class"] = ReprMessage # TODO: GetTensorsMessage
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


@dcfl_route.route("/tensors/<tensor_id>", methods=["PUT"])
@token_required
def update_tensor(tensor_id):
    def route_logic():
        # Get request body
        content = loads(request.data)

        syft_message = {}
        syft_message["message_class"] = ReprMessage # TODO: UpdateTensorMessage
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


@dcfl_route.route("/tensors/<tensor_id>", methods=["DELETE"])
@token_required
def delete_tensor(tensor_id):
    def route_logic():
        # Get request body
        content = loads(request.data)

        syft_message = {}
        syft_message["message_class"] = ReprMessage # TODO: DeleteTensor
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
