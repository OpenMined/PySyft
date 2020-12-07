from ..blueprint import dcfl_blueprint as dcfl_route
from flask import request, Response
import json

from syft.core.node.common.service.repr_service import ReprMessage
from ...auth import error_handler, token_required
from ....core.node import node

## Nodes CRUD
@dcfl_route.route("/nodes", methods=["POST"])
@token_required
def create_node():
    def route_logic():
        # Get request body
        content = loads(request.data)

        syft_message = {}
        syft_message["message_class"] = ReprMessage  # TODO: CreateWorkerMessage
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

    status_code, response_body = error_handler(process_as_syft_message)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@dcfl_route.route("/nodes/<node_id>", methods=["GET"])
@token_required
def get_node(node_id):
    def route_logic():
        # Get request body
        content = loads(request.data)

        syft_message = {}
        syft_message["message_class"] = ReprMessage  # TODO: GetWorkerMessage
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

    status_code, response_body = error_handler(process_as_syft_message)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@dcfl_route.route("/nodes", methods=["GET"])
@token_required
def get_all_nodes():
    def route_logic():
        # Get request body
        content = loads(request.data)

        syft_message = {}
        syft_message["message_class"] = ReprMessage  # TODO: GetAllWorkersMessage
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

    status_code, response_body = error_handler(process_as_syft_message)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@dcfl_route.route("/nodes/<node_id>", methods=["PUT"])
@token_required
def update_node(node_id):
    def route_logic():
        # Get request body
        content = loads(request.data)

        syft_message = {}
        syft_message["message_class"] = ReprMessage  # TODO: UpdateWorkerMessage
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

    status_code, response_body = error_handler(process_as_syft_message)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@dcfl_route.route("/nodes/<node_id>", methods=["DELETE"])
@token_required
def delete_node(node_id):
    def route_logic():
        # Get request body
        content = loads(request.data)

        syft_message = {}
        syft_message["message_class"] = ReprMessage  # TODO: DeleteWorkerMessage
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

    status_code, response_body = error_handler(process_as_syft_message)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


## Auto-scaling CRUD


@dcfl_route.route("/nodes/autoscaling", methods=["POST"])
@token_required
def create_autoscaling():
    def route_logic():
        # Get request body
        content = loads(request.data)

        syft_message = {}
        syft_message["message_class"] = ReprMessage  # TODO: CreateAutoScalingMessage
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

    status_code, response_body = error_handler(process_as_syft_message)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@dcfl_route.route("/nodes/autoscaling", methods=["GET"])
@token_required
def get_all_autoscaling_conditions():
    def route_logic():
        # Get request body
        content = loads(request.data)

        syft_message = {}
        syft_message[
            "message_class"
        ] = ReprMessage  # TODO: GetAutoScalingConditionsMessage
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

    status_code, response_body = error_handler(process_as_syft_message)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@dcfl_route.route("/nodes/autoscaling/<autoscaling_id>", methods=["GET"])
@token_required
def get_specific_autoscaling_condition(autoscaling_id):
    def route_logic():
        # Get request body
        content = loads(request.data)

        syft_message = {}
        syft_message["message_class"] = ReprMessage  # TODO: GetAutoScalingCondition
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

    status_code, response_body = error_handler(process_as_syft_message)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@dcfl_route.route("/nodes/autoscaling/<autoscaling_id>", methods=["PUT"])
@token_required
def update_autoscaling_condition(autoscaling_id):
    def route_logic():
        # Get request body
        content = loads(request.data)

        syft_message = {}
        syft_message["message_class"] = ReprMessage  # TODO: UpdateAutoScalingCondition
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

    status_code, response_body = error_handler(process_as_syft_message)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@dcfl_route.route("/nodes/autoscaling/<autoscaling_id>", methods=["DELETE"])
@token_required
def delete_autoscaling_condition(autoscaling_id):
    def route_logic():
        # Get request body
        content = loads(request.data)

        syft_message = {}
        syft_message["message_class"] = ReprMessage  # TODO: DeleteAutoScalingCondition
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

    status_code, response_body = error_handler(process_as_syft_message)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


## Workers CRUD


@dcfl_route.route("/workers", methods=["POST"])
@token_required
def create_worker():
    def route_logic():
        # Get request body
        content = loads(request.data)

        syft_message = {}
        syft_message["message_class"] = ReprMessage  # TODO: CreateVirtualMachineMessage
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

    status_code, response_body = error_handler(process_as_syft_message)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@dcfl_route.route("/workers", methods=["GET"])
@token_required
def get_all_workers():
    def route_logic():
        # Get request body
        content = loads(request.data)

        syft_message = {}
        syft_message["message_class"] = ReprMessage  # TODO: GetVirtualMachinesMessage
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

    status_code, response_body = error_handler(process_as_syft_message)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@dcfl_route.route("/workers/<worker_id>", methods=["GET"])
@token_required
def get_specific_worker(worker_id):
    def route_logic():
        # Get request body
        content = loads(request.data)

        syft_message = {}
        syft_message["message_class"] = ReprMessage  # TODO: GetVirtualMachineMessage
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

    status_code, response_body = error_handler(process_as_syft_message)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@dcfl_route.route("/workers/<worker_id>", methods=["DELETE"])
@token_required
def delete_worker(worker_id):
    def route_logic():
        # Get request body
        content = loads(request.data)

        syft_message = {}
        syft_message["message_class"] = ReprMessage  # TODO: DeleteVirtualMachineMessage
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

    status_code, response_body = error_handler(process_as_syft_message)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )
