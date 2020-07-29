# Standard Python Imports
import io
import json
import logging
from math import floor
from random import random

import numpy as np
from flask import Response, current_app, render_template, request, send_file

# External modules imports
from requests_toolbelt import MultipartEncoder

# Dependencies used by req_join endpoint
# It's a mockup endpoint and should be removed soon.
from scipy.stats import poisson

# Local imports
from ... import model_centric_routes
from ...core.codes import CYCLE, MSG_FIELD, RESPONSE_MSG
from ...core.exceptions import InvalidRequestKeyError, ModelNotFoundError, PyGridError
from ...events.model_centric.fl_events import (
    assign_worker_id,
    cycle_request,
    report,
    requires_speed_test,
)
from ...model_centric.auth.federated import verify_token
from ...model_centric.controller import processes
from ...model_centric.cycles import cycle_manager
from ...model_centric.models import model_manager
from ...model_centric.processes import process_manager
from ...model_centric.syft_assets import plans, protocols
from ...model_centric.workers import worker_manager


@model_centric_routes.route("/cycle-request", methods=["POST"])
def worker_cycle_request():
    """" This endpoint is where the worker is attempting to join an active
    federated learning cycle."""
    response_body = {}
    status_code = None

    try:
        body = json.loads(request.data)
        response_body = cycle_request({MSG_FIELD.DATA: body}, None)
    except (PyGridError, json.decoder.JSONDecodeError) as e:
        status_code = 400  # Bad Request
        response_body[RESPONSE_MSG.ERROR] = str(e)
    except Exception as e:
        status_code = 500  # Internal Server Error
        response_body[RESPONSE_MSG.ERROR] = str(e)

    if isinstance(response_body, str):
        # Consider just data field as a response
        response_body = json.loads(response_body)[MSG_FIELD.DATA]

    response_body = json.dumps(response_body)
    return Response(response_body, status=status_code, mimetype="application/json")


@model_centric_routes.route("/speed-test", methods=["GET", "POST"])
def connection_speed_test():
    """Connection speed test."""
    response_body = {}
    status_code = None

    try:
        _worker_id = request.args.get("worker_id", None)
        _random = request.args.get("random", None)
        _is_ping = request.args.get("is_ping", None)

        if not _worker_id or not _random:
            raise PyGridError

        # If GET method
        if request.method == "GET":
            if _is_ping is None:
                # Download data sample (64MB)
                data_sample = b"x" * 67108864  # 64 Megabyte
                response = {"sample": data_sample}
                form = MultipartEncoder(response)
                return Response(form.to_string(), mimetype=form.content_type)
            else:
                status_code = 200  # Success
        elif request.method == "POST":  # Otherwise, it's POST method
            status_code = 200  # Success

    except PyGridError as e:
        status_code = 400  # Bad Request
        response_body[RESPONSE_MSG.ERROR] = str(e)
    except Exception as e:
        status_code = 500  # Internal Server Error
        response_body[RESPONSE_MSG.ERROR] = str(e)

    return Response(
        json.dumps(response_body), status=status_code, mimetype="application/json"
    )


@model_centric_routes.route("/report", methods=["POST"])
def report_diff():
    """Allows reporting of (agg/non-agg) model diff after worker completes a
    cycle."""
    response_body = {}
    status_code = None

    try:
        body = json.loads(request.data)
        response_body = report({MSG_FIELD.DATA: body}, None)
    except (PyGridError, json.decoder.JSONDecodeError) as e:
        status_code = 400  # Bad Request
        response_body[RESPONSE_MSG.ERROR] = str(e)
    except Exception as e:
        status_code = 500  # Internal Server Error
        response_body[RESPONSE_MSG.ERROR] = str(e)

    if isinstance(response_body, str):
        # Consider just data field as a response
        response_body = json.loads(response_body)[MSG_FIELD.DATA]

    response_body = json.dumps(response_body)
    return Response(response_body, status=status_code, mimetype="application/json")


@model_centric_routes.route("/get-protocol", methods=["GET"])
def download_protocol():
    """Request a download of a protocol."""

    response_body = {}
    status_code = None
    try:
        worker_id = request.args.get("worker_id", None)
        request_key = request.args.get("request_key", None)
        protocol_id = request.args.get("protocol_id", None)

        # Retrieve Process Entities
        _protocol = protocols.get(id=protocol_id)
        _cycle = cycle_manager.last(_protocol.fl_process_id)
        _worker = worker_manager.get(id=worker_id)
        _accepted = cycle_manager.validate(_worker.id, _cycle.id, request_key)

        if not _accepted:
            raise InvalidRequestKeyError

        status_code = 200  # Success
        response_body[CYCLE.PROTOCOLS] = _protocol.value
    except InvalidRequestKeyError as e:
        status_code = 401  # Unauthorized
        response_body[RESPONSE_MSG.ERROR] = str(e)
    except PyGridError as e:
        status_code = 400  # Bad request
        response_body[RESPONSE_MSG.ERROR] = str(e)
    except Exception as e:
        status_code = 500  # Internal Server Error
        response_body[RESPONSE_MSG.ERROR] = str(e)

    return Response(
        json.dumps(response_body), status=status_code, mimetype="application/json"
    )


@model_centric_routes.route("/get-model", methods=["GET"])
def download_model():
    """Request a download of a model."""

    response_body = {}
    status_code = None
    try:
        worker_id = request.args.get("worker_id", None)
        request_key = request.args.get("request_key", None)
        model_id = request.args.get("model_id", None)

        # Retrieve Process Entities
        _model = model_manager.get(id=model_id)
        _cycle = cycle_manager.last(_model.fl_process_id)
        _worker = worker_manager.get(id=worker_id)
        _accepted = cycle_manager.validate(_worker.id, _cycle.id, request_key)

        if not _accepted:
            raise InvalidRequestKeyError

        _last_checkpoint = model_manager.load(model_id=model_id)

        return send_file(
            io.BytesIO(_last_checkpoint.value), mimetype="application/octet-stream"
        )

    except InvalidRequestKeyError as e:
        status_code = 401  # Unauthorized
        response_body[RESPONSE_MSG.ERROR] = str(e)
    except PyGridError as e:
        status_code = 400  # Bad request
        response_body[RESPONSE_MSG.ERROR] = str(e)
    except Exception as e:
        status_code = 500  # Internal Server Error
        response_body[RESPONSE_MSG.ERROR] = str(e)

    return Response(
        json.dumps(response_body), status=status_code, mimetype="application/json"
    )


@model_centric_routes.route("/get-plan", methods=["GET"])
def download_plan():
    """Request a download of a plan."""

    response_body = {}
    status_code = None

    try:
        worker_id = request.args.get("worker_id", None)
        request_key = request.args.get("request_key", None)
        plan_id = request.args.get("plan_id", None)
        receive_operations_as = request.args.get("receive_operations_as", None)

        # Retrieve Process Entities
        _plan = process_manager.get_plan(id=plan_id, is_avg_plan=False)
        _cycle = cycle_manager.last(fl_process_id=_plan.fl_process_id)
        _worker = worker_manager.get(id=worker_id)
        _accepted = cycle_manager.validate(_worker.id, _cycle.id, request_key)

        if not _accepted:
            raise InvalidRequestKeyError

        status_code = 200  # Success

        if receive_operations_as == "torchscript":
            response_body = _plan.value_ts
        elif receive_operations_as == "tfjs":
            response_body = _plan.value_tfjs
        else:
            response_body = _plan.value

        return send_file(io.BytesIO(response_body), mimetype="application/octet-stream")

    except InvalidRequestKeyError as e:
        status_code = 401  # Unauthorized
        response_body[RESPONSE_MSG.ERROR] = str(e)
    except PyGridError as e:
        status_code = 400  # Bad request
        response_body[RESPONSE_MSG.ERROR] = str(e)
    except Exception as e:
        status_code = 500  # Internal Server Error
        response_body[RESPONSE_MSG.ERROR] = str(e)

    return Response(
        json.dumps(response_body), status=status_code, mimetype="application/json"
    )


@model_centric_routes.route("/authenticate", methods=["POST"])
def auth():
    """uses JWT (HSA/RSA) to authenticate."""
    response_body = {}
    status_code = 200
    data = json.loads(request.data)
    _auth_token = data.get("auth_token", None)
    model_name = data.get("model_name", None)
    model_version = data.get("model_version", None)

    try:
        verification_result = verify_token(_auth_token, model_name, model_version)

        if verification_result["status"] == RESPONSE_MSG.SUCCESS:
            resp = assign_worker_id({"auth_token": _auth_token}, None)
            # check if requires speed test
            resp[MSG_FIELD.REQUIRES_SPEED_TEST] = requires_speed_test(
                model_name, model_version
            )
            response_body = resp

        elif verification_result["status"] == RESPONSE_MSG.ERROR:
            status_code = 400
            response_body[RESPONSE_MSG.ERROR] = verification_result["error"]

    except Exception as e:
        status_code = 401
        response_body[RESPONSE_MSG.ERROR] = str(e)

    return Response(
        json.dumps(response_body), status=status_code, mimetype="application/json"
    )


@model_centric_routes.route("/req-join", methods=["GET"])
def fl_cycle_application_decision():
    """use the temporary req_join endpoint to mockup:

    - reject if worker does not satisfy 'minimum_upload_speed' and/or 'minimum_download_speed'
    - is a part of current or recent cycle according to 'do_not_reuse_workers_until_cycle'
    - selects according to pool_selection
    - is under max worker (with some padding to account for expected percent of workers so do not report successfully)
    """

    # parse query strings (for now), eventually this will be parsed from the request body
    model_id = request.args.get("model_id")
    up_speed = request.args.get("up_speed")
    down_speed = request.args.get("down_speed")
    worker_id = request.args.get("worker_id")
    worker_ping = request.args.get("ping")
    _cycle = cycle_manager.last(model_id)
    _accept = False
    """
    MVP variable stubs:
        we will stub these with hard coded numbers first, then make functions to dynaically query/update in subsquent PRs
    """
    # this will be replaced with a function that check for the same (model_id, version_#) tuple when the worker last participated
    last_participation = 1
    # how late is too late into the cycle time to give a worker "new work", if only 5 seconds left probably don't bother, set this intelligently later
    MINIMUM_CYCLE_TIME_LEFT = 500
    # the historical amount of workers that fail to report (out of time, offline, too slow etc...),
    # could be modified to be worker/model specific later, track across overall pygrid instance for now
    EXPECTED_FAILURE_RATE = 0.2

    dummy_server_config = {
        "max_workers": 100,
        "pool_selection": "random",  # or "iterate"
        "num_cycles": 5,
        "do_not_reuse_workers_until_cycle": 4,
        "cycle_length": 8 * 60 * 60,  # 8 hours
        "minimum_upload_speed": 2000,  # 2 mbps
        "minimum_download_speed": 4000,  # 4 mbps
    }
    """  end of variable stubs """

    _server_config = dummy_server_config

    up_speed_check = up_speed > _server_config["minimum_upload_speed"]
    down_speed_check = down_speed > _server_config["minimum_download_speed"]
    cycle_valid_check = (
        (
            last_participation + _server_config["do_not_reuse_workers_until_cycle"]
            >= _cycle.get(
                "cycle_sequence", 99999
            )  # this should reuturn current cycle sequence number
        )
        * (_cycle.get("cycle_sequence", 99999) <= _server_config["num_cycles"])
        * (_cycle.cycle_time > MINIMUM_CYCLE_TIME_LEFT)
        * (worker_id not in _cycle._workers)
    )

    if up_speed_check * down_speed_check * cycle_valid_check:
        if _server_config["pool_selection"] == "iterate" and len(
            _cycle._workers
        ) < _server_config["max_workers"] * (1 + EXPECTED_FAILURE_RATE):
            """first come first serve selection mode."""
            _accept = True
        elif _server_config["pool_selection"] == "random":
            """probabilistic model for rejection rate:

                - model the rate of worker's request to join as lambda in a poisson process
                - set probabilistic reject rate such that we can expect enough workers will request to join and be accepted
                    - between now and ETA till end of _server_config['cycle_length']
                    - such that we can expect (,say with 95% confidence) successful completion of the cycle
                    - while accounting for EXPECTED_FAILURE_RATE (% of workers that join cycle but never successfully report diff)

            EXPECTED_FAILURE_RATE = moving average with exponential decay based on historical data (maybe: noised up weights for security)

            k' = max_workers * (1+EXPECTED_FAILURE_RATE) # expected failure adjusted max_workers = var: k_prime

            T_left = T_cycle_end - T_now # how much time is left (in the same unit as below)

            normalized_lambda_actual = (recent) historical rate of request / unit time

            lambda' = number of requests / unit of time that would satisfy the below equation

            probability of receiving at least k' requests per unit time:
                P(K>=k') = 0.95 = e ^ ( - lambda' * T_left) * ( lambda' * T_left) ^ k' / k'! = 1 - P(K<k')

            var: lambda_approx = lambda' * T_left

            solve for lambda':
                use numerical approximation (newton's method) or just repeatedly call prob = poisson.sf(x, lambda') via scipy

            reject_probability = 1 - lambda_approx / (normalized_lambda_actual * T_left)
            """

            # time base units = 1 hr, assumes lambda_actual and lambda_approx have the same unit as T_left
            k_prime = _server_config["max_workers"] * (1 + EXPECTED_FAILURE_RATE)
            T_left = _cycle.get("cycle_time", 0)

            # TODO: remove magic number = 5 below... see block comment above re: how
            normalized_lambda_actual = 5
            lambda_actual = (
                normalized_lambda_actual * T_left
            )  # makes lambda_actual have same unit as lambda_approx
            # @hyperparam: valid_range => (0, 1) | (+) => more certainty to have completed cycle, (-) => more efficient use of worker as computational resource
            confidence = 0.95  # P(K>=k')
            pois = lambda l: poisson.sf(k_prime, l) - confidence
            """
            _bisect_approximator because:
                - solving for lambda given P(K>=k') has no algebraic solution (that I know of) => need approxmiation
                - scipy's optimizers are not stable for this problem (I tested a few) => need custom approxmiation
                - at this MVP stage we are not likely to experince performance problems, binary search is log(N)
            refactor notes:
                - implmenting a smarter approximiator using lambert's W or newton's methods will take more time
                - if we do need to scale then we can refactor to the above ^
            """
            # @hyperparam: valid_range => (0, 1) | (+) => get a faster but lower quality approximation
            _search_tolerance = 0.01

            def _bisect_approximator(arr, search_tolerance=_search_tolerance):
                """uses binary search to find lambda_actual within
                search_tolerance."""
                n = len(arr)
                L = 0
                R = n - 1

                while L <= R:
                    mid = floor((L + R) / 2)
                    if pois(arr[mid]) > 0 and pois(arr[mid]) < search_tolerance:
                        return mid
                    elif pois(arr[mid]) > 0 and pois(arr[mid]) > search_tolerance:
                        R = mid - 1
                    else:
                        L = mid + 1
                return None

            """
            if the number of workers is relatively small:
                - approximiation methods is not neccessary / we can find exact solution fast
                - and search_tolerance is not guaranteed because lambda has to be int()
            """
            if k_prime < 50:
                lambda_approx = np.argmin(
                    [abs(pois(x)) for x in range(floor(k_prime * 3))]
                )
            else:
                lambda_approx = _bisect_approximator(range(floor(k_prime * 3)))

            rej_prob = (
                (1 - lambda_approx / lambda_actual)
                if lambda_actual > lambda_approx
                else 0  # don't reject if we expect to be short on worker requests
            )

            # additional security:
            if (
                k_prime > 50
                and abs(poisson.sf(k_prime, lambda_approx) - confidence)
                > _search_tolerance
            ):
                """something went wrong, fall back to safe default."""
                rej_prob = 0.1
                WARN = "_bisect_approximator failed unexpectedly, reset rej_prob to default"
                logging.exception(WARN)  # log error

            if random.random_sample() < rej_prob:
                _accept = True

    if _accept:
        return Response(
            json.dumps(
                {"status": "accepted"}
            ),  # leave out other accpet keys/values for now
            status=200,
            mimetype="application/json",
        )

    # reject by default
    return Response(
        json.dumps(
            {"status": "rejected"}
        ),  # leave out other accpet keys/values for now
        status=400,
        mimetype="application/json",
    )


@model_centric_routes.route("/retrieve-model", methods=["GET"])
def get_model():
    """Request a download of a model."""

    response_body = {}
    status_code = None
    try:
        name = request.args.get("name", None)
        version = request.args.get("version", None)
        checkpoint = request.args.get("checkpoint", None)

        process_query = {"name": name}
        if version:
            process_query["version"] = version
        _fl_process = process_manager.last(**process_query)
        _model = model_manager.get(fl_process_id=_fl_process.id)

        checkpoint_query = {"model_id": _model.id}
        if checkpoint:
            if checkpoint.isnumeric():
                checkpoint_query["number"] = int(checkpoint)
            else:
                checkpoint_query["alias"] = checkpoint
        else:
            checkpoint_query["alias"] = "latest"

        logging.info(f"Looking for checkpoint: {checkpoint_query}")
        _model_checkpoint = model_manager.load(**checkpoint_query)

        return send_file(
            io.BytesIO(_model_checkpoint.value), mimetype="application/octet-stream"
        )

    except ModelNotFoundError as e:
        status_code = 404
        response_body[RESPONSE_MSG.ERROR] = str(e)
        logging.warning("Model not found in get-model", exc_info=e)

    except Exception as e:
        status_code = 500  # Internal Server Error
        response_body[RESPONSE_MSG.ERROR] = str(e)
        logging.error("Exception in get-model", exc_info=e)

    return Response(
        json.dumps(response_body), status=status_code, mimetype="application/json"
    )
