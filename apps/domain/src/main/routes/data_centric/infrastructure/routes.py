from ..blueprint import dcfl_blueprint as dcfl_route
from flask import request, Response
import json

from syft.core.node.common.service.repr_service import ReprMessage
from ...auth import error_handler, token_required
from ....core.node import node

## Nodes CRUD
@dcfl_route.route("/nodes", methods=["POST"])
# @token_required
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

    # status_code, response_body = error_handler(process_as_syft_message)

    status_code, response_body = 200, {"msg": "Node created succesfully!"}

    return Response(
        json.dumps(response_body), status=status_code, mimetype="application/json"
    )


@dcfl_route.route("/nodes/<node_id>", methods=["GET"])
# @token_required
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

    # status_code, response_body = error_handler(process_as_syft_message)

    status_code, response_body = 200, {
        "node": {"id": "464615", "tags": ["node-a"], "description": "node sample"}
    }

    return Response(
        json.dumps(response_body), status=status_code, mimetype="application/json"
    )


@dcfl_route.route("/nodes", methods=["GET"])
# @token_required
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

    # status_code, response_body = error_handler(process_as_syft_message)
    status_code, response_body = 200, {
        "nodes": [
            {"id": "35654sad6ada", "address": "175.89.0.170"},
            {"id": "adfarf3f1af5", "address": "175.55.22.150"},
            {"id": "fas4e6e1fas", "address": "195.74.128.132"},
        ]
    }

    return Response(
        json.dumps(response_body), status=status_code, mimetype="application/json"
    )


@dcfl_route.route("/nodes/<node_id>", methods=["PUT"])
# @token_required
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

    # status_code, response_body = error_handler(process_as_syft_message)
    status_code, response_body = 200, {"msg": "Node changed succesfully!"}

    return Response(
        json.dumps(response_body), status=status_code, mimetype="application/json"
    )


@dcfl_route.route("/nodes/<node_id>", methods=["DELETE"])
# @token_required
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

    # status_code, response_body = error_handler(process_as_syft_message)

    status_code, response_body = 200, {"msg": "Node deleted succesfully!"}

    return Response(
        json.dumps(response_body), status=status_code, mimetype="application/json"
    )


## Auto-scaling CRUD


@dcfl_route.route("/nodes/autoscaling", methods=["POST"])
# @token_required
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

    # status_code, response_body = error_handler(process_as_syft_message)
    status_code, response_body = 200, {
        "msg": "Autoscaling condition created succesfully!"
    }

    return Response(
        json.dumps(response_body), status=status_code, mimetype="application/json"
    )


@dcfl_route.route("/nodes/autoscaling", methods=["GET"])
# @token_required
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

    # status_code, response_body = error_handler(process_as_syft_message)

    status_code, response_body = 200, {
        "condition_a": {"mem_usage": "80%", "cpu_usage": "90%", "disk_usage": "75%"},
        "condition_b": {"mem_usage": "50%", "cpu_usage": "70%", "disk_usage": "95%"},
        "condition_c": {"mem_usage": "92%", "cpu_usage": "77%", "disk_usage": "50%"},
    }

    return Response(
        json.dumps(response_body), status=status_code, mimetype="application/json"
    )


@dcfl_route.route("/nodes/autoscaling/<autoscaling_id>", methods=["GET"])
# @token_required
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

    # status_code, response_body = error_handler(process_as_syft_message)

    status_code, response_body = 200, {
        "mem_usage": "80%",
        "cpu_usage": "90%",
        "disk_usage": "75%",
    }

    return Response(
        json.dumps(response_body), status=status_code, mimetype="application/json"
    )


@dcfl_route.route("/nodes/autoscaling/<autoscaling_id>", methods=["PUT"])
# @token_required
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

    # status_code, response_body = error_handler(process_as_syft_message)

    status_code, response_body = 200, {
        "msg": "Autoscaling condition updated succesfully!"
    }

    return Response(
        json.dumps(response_body), status=status_code, mimetype="application/json"
    )


@dcfl_route.route("/nodes/autoscaling/<autoscaling_id>", methods=["DELETE"])
# @token_required
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

    # status_code, response_body = error_handler(process_as_syft_message)

    status_code, response_body = 200, {
        "msg": "Autoscaling condition deleted succesfully!"
    }

    return Response(
        json.dumps(response_body), status=status_code, mimetype="application/json"
    )


## Workers CRUD


@dcfl_route.route("/workers", methods=["POST"])
# @token_required
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

    # status_code, response_body = error_handler(process_as_syft_message)

    status_code, response_body = 200, {"msg": "Worker created succesfully!"}

    return Response(
        json.dumps(response_body), status=status_code, mimetype="application/json"
    )


@dcfl_route.route("/workers", methods=["GET"])
# @token_required
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

    # status_code, response_body = error_handler(process_as_syft_message)

    status_code, response_body = 200, {
        "workers": [
            {"id": "546513231a", "address": "159.156.128.165", "datasets": 25320},
            {"id": "asfa16f5aa", "address": "138.142.125.125", "datasets": 2530},
            {"id": "af61ea3a3f", "address": "19.16.98.146", "datasets": 2320},
            {"id": "af4a51adas", "address": "15.59.18.165", "datasets": 5320},
        ]
    }

    return Response(
        json.dumps(response_body), status=status_code, mimetype="application/json"
    )


@dcfl_route.route("/workers/<worker_id>", methods=["GET"])
# @token_required
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

    # status_code, response_body = error_handler(process_as_syft_message)
    status_code, response_body = 200, {
        "worker": {"id": "9846165", "address": "159.156.128.165", "datasets": 25320}
    }

    return Response(
        json.dumps(response_body), status=status_code, mimetype="application/json"
    )


@dcfl_route.route("/workers/<worker_id>", methods=["DELETE"])
# @token_required
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

    # status_code, response_body = error_handler(process_as_syft_message)

    status_code, response_body = 200, {"msg": "Worker was deleted succesfully!"}

    return Response(
        json.dumps(response_body), status=status_code, mimetype="application/json"
    )
