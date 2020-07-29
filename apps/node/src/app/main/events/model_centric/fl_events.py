# Standard python imports
import base64
import json
import traceback
import uuid
from binascii import unhexlify

from ...core.codes import CYCLE, MODEL_CENTRIC_FL_EVENTS, MSG_FIELD, RESPONSE_MSG
from ...core.exceptions import (
    CycleNotFoundError,
    MaxCycleLimitExceededError,
    PyGridError,
)
from ...model_centric.auth.federated import verify_token
from ...model_centric.controller import processes
from ...model_centric.processes import process_manager
from ...model_centric.workers import worker_manager

# Local imports
# Local imports
from ..socket_handler import SocketHandler

# Singleton socket handler
handler = SocketHandler()


def host_federated_training(message: dict, socket=None) -> str:
    """This will allow for training cycles to begin on end-user devices.

    Args:
        message : Message body sent by some client.
        socket: Socket descriptor.
    Returns:
        response : String response to the client
    """
    data = message[MSG_FIELD.DATA]
    response = {}

    try:
        # Retrieve JSON values
        serialized_model = unhexlify(
            data.get(MSG_FIELD.MODEL, None).encode()
        )  # Only one
        serialized_client_plans = {
            k: unhexlify(v.encode()) for k, v in data.get(CYCLE.PLANS, {}).items()
        }  # 1 or *
        serialized_client_protocols = {
            k: unhexlify(v.encode()) for k, v in data.get(CYCLE.PROTOCOLS, {}).items()
        }  # 0 or *
        serialized_avg_plan = unhexlify(
            data.get(CYCLE.AVG_PLAN, None).encode()
        )  # Only one
        client_config = data.get(CYCLE.CLIENT_CONFIG, None)  # Only one
        server_config = data.get(CYCLE.SERVER_CONFIG, None)  # Only one

        # Create a new FL Process
        processes.create_process(
            model=serialized_model,
            client_plans=serialized_client_plans,
            client_protocols=serialized_client_protocols,
            server_averaging_plan=serialized_avg_plan,
            client_config=client_config,
            server_config=server_config,
        )
        response[CYCLE.STATUS] = RESPONSE_MSG.SUCCESS
    except Exception as e:  # Retrieve exception messages such as missing JSON fields.
        response[RESPONSE_MSG.ERROR] = str(e) + traceback.format_exc()

    response = {
        MSG_FIELD.TYPE: MODEL_CENTRIC_FL_EVENTS.HOST_FL_TRAINING,
        MSG_FIELD.DATA: response,
    }

    return json.dumps(response)


def assign_worker_id(message: dict, socket=None) -> str:
    """New workers should receive a unique worker ID after authenticate on
    PyGrid platform.

    Args:
        message : Message body sended after token verification.
        socket: Socket descriptor.
    Returns:
        response : String response to the client
    """
    response = {}

    # Create a new worker instance and bind it with the socket connection.
    try:
        # Create new worker id
        worker_id = str(uuid.uuid4())

        # Create a link between worker id and socket descriptor
        handler.new_connection(worker_id, socket)

        # Create worker instance
        worker_manager.create(worker_id)

        requires_speed_test = True

        response[CYCLE.STATUS] = RESPONSE_MSG.SUCCESS
        response[MSG_FIELD.WORKER_ID] = worker_id

    except Exception as e:  # Retrieve exception messages such as missing JSON fields.
        response[CYCLE.STATUS] = RESPONSE_MSG.ERROR
        response[RESPONSE_MSG.ERROR] = str(e)

    return response


def requires_speed_test(model_name, model_version):

    kwargs = {"name": model_name}
    if model_version is not None:
        kwargs["version"] = model_version

    server_config, _ = process_manager.get_configs(**kwargs)

    #
    return (
        True
        if (
            server_config.get("minimum_upload_speed", None) is not None
            or server_config.get("minimum_download_speed", None) is not None
        )
        else False
    )


def authenticate(message: dict, socket=None) -> str:
    """Check the submitted token and assign the worker a new id.

    Args:
        message : Message body sended by some client.
        socket: Socket descriptor.
    Returns:
        response : String response to the client
    """
    data = message.get("data")
    response = {}

    try:
        _auth_token = data.get("auth_token")
        model_name = data.get("model_name", None)
        model_version = data.get("model_version", None)

        verification_result = verify_token(_auth_token, model_name, model_version)

        if verification_result["status"] == RESPONSE_MSG.SUCCESS:
            response = assign_worker_id({"auth_token": _auth_token}, socket)
            # check if requires speed test
            response[MSG_FIELD.REQUIRES_SPEED_TEST] = requires_speed_test(
                model_name, model_version
            )
        else:
            response[RESPONSE_MSG.ERROR] = verification_result["error"]

    except Exception as e:
        response[RESPONSE_MSG.ERROR] = str(e) + "\n" + traceback.format_exc()

    response = {
        MSG_FIELD.TYPE: MODEL_CENTRIC_FL_EVENTS.AUTHENTICATE,
        MSG_FIELD.DATA: response,
    }
    return json.dumps(response)


def cycle_request(message: dict, socket=None) -> str:
    """This event is where the worker is attempting to join an active federated
    learning cycle.

    Args:
        message : Message body sent by some client.
        socket: Socket descriptor.
    Returns:
        response : String response to the client
    """
    data = message[MSG_FIELD.DATA]
    response = {}

    try:
        # Retrieve JSON values
        worker_id = data.get(MSG_FIELD.WORKER_ID, None)
        name = data.get(MSG_FIELD.MODEL, None)
        version = data.get(CYCLE.VERSION, None)

        # Retrieve the worker
        worker = worker_manager.get(id=worker_id)

        # Request fields to worker's DB fields mapping
        fields_map = {
            CYCLE.PING: "ping",
            CYCLE.DOWNLOAD: "avg_download",
            CYCLE.UPLOAD: "avg_upload",
        }
        requires_speed_fields = requires_speed_test(name, version)

        # Check and save connection speed to DB
        for request_field, db_field in fields_map.items():
            if request_field in data:
                value = data.get(request_field)
                if not isinstance(value, (float, int)) or value < 0:
                    raise PyGridError(
                        f"'{request_field}' needs to be a positive number"
                    )
                setattr(worker, db_field, float(value))
            elif requires_speed_fields:
                # Require fields to present when FL model has speed req's
                raise PyGridError(f"'{request_field}' is required")

        worker_manager.update(worker)  # Update database worker attributes

        # The last time this worker was assigned for this model/version.
        last_participation = processes.last_cycle(worker_id, name, version)

        # Assign
        response = processes.assign(name, version, worker, last_participation)
    except CycleNotFoundError:
        # Nothing to do
        response[CYCLE.STATUS] = CYCLE.REJECTED
    except MaxCycleLimitExceededError as e:
        response[CYCLE.STATUS] = CYCLE.REJECTED
        response[MSG_FIELD.MODEL] = e.name
    except Exception as e:
        print("Exception: ", str(e))
        response[CYCLE.STATUS] = CYCLE.REJECTED
        response[RESPONSE_MSG.ERROR] = str(e) + traceback.format_exc()

    response = {
        MSG_FIELD.TYPE: MODEL_CENTRIC_FL_EVENTS.CYCLE_REQUEST,
        MSG_FIELD.DATA: response,
    }
    return json.dumps(response)


def report(message: dict, socket=None) -> str:
    """This method will allow a worker that has been accepted into a cycle and
    finished training a model on their device to upload the resulting model
    diff.

    Args:
        message : Message body sent by some client.
        socket: Socket descriptor.
    Returns:
        response : String response to the client
    """
    data = message[MSG_FIELD.DATA]
    response = {}

    try:
        worker_id = data.get(MSG_FIELD.WORKER_ID, None)
        request_key = data.get(CYCLE.KEY, None)

        # It's simpler for client (and more efficient for bandwidth) to use base64
        # diff = unhexlify()
        diff = base64.b64decode(data.get(CYCLE.DIFF, None).encode())

        # Submit model diff and run cycle and task async to avoid block report request
        # (for prod we probably should be replace this with Redis queue + separate worker)
        processes.submit_diff(worker_id, request_key, diff)

        response[CYCLE.STATUS] = RESPONSE_MSG.SUCCESS
    except Exception as e:  # Retrieve exception messages such as missing JSON fields.
        response[RESPONSE_MSG.ERROR] = str(e) + traceback.format_exc()

    response = {
        MSG_FIELD.TYPE: MODEL_CENTRIC_FL_EVENTS.REPORT,
        MSG_FIELD.DATA: response,
    }
    return json.dumps(response)
