# Database object controllers
# Generic imports
import hashlib
import logging
import uuid
from datetime import datetime

from ...core.codes import CYCLE, MSG_FIELD
from ...core.exceptions import ProtocolNotFoundError
from ..cycles import cycle_manager
from ..models import model_manager
from ..processes import process_manager
from ..workers import worker_manager


class FLController:
    """This class implements controller design pattern over the federated
    learning processes."""

    def __init__(self):
        pass

    def create_process(
        self,
        model,
        client_plans,
        client_config,
        server_config,
        server_averaging_plan,
        client_protocols=None,
    ):
        """ Register a new federated learning process
            Args:
                model: Model object.
                client_plans : an object containing syft plans.
                client_protocols : an object containing syft protocols.
                client_config: the client configurations
                server_averaging_plan: a function that will instruct PyGrid on how to average model diffs that are returned from the workers.
                server_config: the server configurations
            Returns:
                process : FLProcess Instance.
            Raises:
                FLProcessConflict (PyGridError) : If Process Name/Version already exists.
        """
        cycle_len = server_config["cycle_length"]

        # Create a new federated learning process
        # 1 - Create a new process
        # 2 - Save client plans/protocols
        # 3 - Save Server AVG plan
        # 4 - Save Client/Server configs
        _process = process_manager.create(
            client_config,
            client_plans,
            client_protocols,
            server_config,
            server_averaging_plan,
        )

        # Save Model
        # Define the initial version (first checkpoint)
        _model = model_manager.create(model, _process)

        # Create the initial cycle
        _cycle = cycle_manager.create(_process.id, _process.version, cycle_len)

        return _process

    def last_cycle(self, worker_id: str, name: str, version: str) -> int:
        """Retrieve the last time the worker participated from this cycle.

        Args:
            worker_id: Worker's ID.
            name: Federated Learning Process Name.
            version: Model's version.
        Return:
            last_participation: Index of the last cycle assigned to this worker.
        """
        process = process_manager.first(name=name, version=version)
        return cycle_manager.last_participation(process, worker_id)

    def assign(self, name: str, version: str, worker, last_participation: int):
        """ Assign a new worker the specified federated training worker cycle
            Args:
                name: Federated learning process name.
                version: Federated learning process version.
                worker: Worker Object.
                last_participation: The last time that this worker worked on this fl process.
            Return:
                last_participation: Index of the last cycle assigned to this worker.
        """
        _accepted = False

        if version:
            _fl_process = process_manager.first(name=name, version=version)
        else:
            _fl_process = process_manager.last(name=name)

        server_config, client_config = process_manager.get_configs(
            name=name, version=version
        )

        # Retrieve the last cycle used by this fl process/ version
        _cycle = cycle_manager.last(_fl_process.id, None)

        # Check if already exists a relation between the worker and the cycle.
        _assigned = cycle_manager.is_assigned(worker.id, _cycle.id)
        logging.info(
            f"Worker {worker.id} is already assigned to cycle {_cycle.id}: {_assigned}"
        )

        # Check bandwidth
        _comp_bandwidth = worker_manager.is_eligible(worker.id, server_config)

        # Check if the current worker is allowed to join into this cycle
        _allowed = True

        # TODO wire intelligence
        # (
        #     last_participation + server.config["do_not_reuse_workers_until_cycle"]
        #     >= _cycle.sequence
        # )

        _accepted = (not _assigned) and _comp_bandwidth and _allowed
        logging.info(f"Worker is accepted: {_accepted}")

        if _accepted:
            # Assign
            # 1 - Generate new request key
            # 2 - Assign the worker with the cycle.
            key = self._generate_hash_key(uuid.uuid4().hex)
            _worker_cycle = cycle_manager.assign(worker, _cycle, key)

            # Create a plan dictionary
            _plans = process_manager.get_plans(
                fl_process_id=_fl_process.id, is_avg_plan=False
            )

            # Create a protocol dictionary
            try:
                _protocols = process_manager.get_protocols(fl_process_id=_fl_process.id)
            except ProtocolNotFoundError:
                # Protocols are optional
                _protocols = {}

            # Get model ID
            _model = model_manager.get(fl_process_id=_fl_process.id)
            return {
                CYCLE.STATUS: "accepted",
                CYCLE.KEY: _worker_cycle.request_key,
                CYCLE.VERSION: _cycle.version,
                MSG_FIELD.MODEL: name,
                CYCLE.PLANS: _plans,
                CYCLE.PROTOCOLS: _protocols,
                CYCLE.CLIENT_CONFIG: client_config,
                MSG_FIELD.MODEL_ID: _model.id,
            }
        else:
            n_completed_cycles = cycle_manager.count(
                fl_process_id=_fl_process.id, is_completed=True
            )

            _max_cycles = server_config["num_cycles"]

            response = {
                CYCLE.STATUS: "rejected",
                MSG_FIELD.MODEL: name,
                CYCLE.VERSION: _cycle.version,
            }

            # If it's not the last cycle, add the remaining time to the next cycle.
            if n_completed_cycles < _max_cycles:
                remaining = _cycle.end - datetime.now()
                response[CYCLE.TIMEOUT] = str(remaining)

            return response

    def _generate_hash_key(self, primary_key: str) -> str:
        """Generate SHA256 Hash to give access to the cycle.

        Args:
            primary_key : Used to generate hash code.
        Returns:
            hash_code : Hash in string format.
        """
        return hashlib.sha256(primary_key.encode()).hexdigest()

    def submit_diff(self, worker_id: str, request_key: str, diff: bin):
        """Submit worker model diff to the assigned cycle.

        Args:
            worker_id: Worker's ID.
            request_key: request (token) used by this worker during this cycle.
            diff: Model params trained by this worker.
        Raises:
            ProcessLookupError : If Not found any relation between the worker/cycle.
        """
        return cycle_manager.submit_worker_diff(worker_id, request_key, diff)
