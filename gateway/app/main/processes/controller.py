from .federated_learning_process import FLProcess
from .federated_learning_cycle import FederatedLearningCycle
from ..storage import models
from ..storage.warehouse import Warehouse
from datetime import datetime, timedelta
import hashlib
import uuid
from ..codes import MSG_FIELD, CYCLE
from ..exceptions import (
    CycleNotFoundError,
    ProtocolNotFoundError,
    PlanNotFoundError,
    ModelNotFoundError,
    ProcessFoundError,
    FLProcessConflict,
)


class FLController:
    """ This class implements controller design pattern over the federated learning processes. """

    def __init__(self):
        self._processes = Warehouse(models.FLProcess)
        self._cycles = Warehouse(models.Cycle)
        self._worker_cycle = Warehouse(models.WorkerCycle)
        self._configs = Warehouse(models.Config)
        self._plans = Warehouse(models.Plan)
        self._protocols = Warehouse(models.Protocol)
        self._models = Warehouse(models.Model)
        self._model_checkpoints = Warehouse(models.ModelCheckPoint)

    def create_cycle(self, model_id: str, version: str, cycle_time: int = 2500):
        """ Create a new federated learning cycle.
            Args:
                model_id: Model's ID.
                worker_id: Worker's ID.
                cycle_time: Remaining time to finish this cycle.
            Returns:
                fd_cycle: Cycle Instance.
        """
        _fl_process = self._processes.query(model=model_id)

        _new_cycle = None
        if _fl_process:
            # Retrieve a list of cycles using the same model_id/version
            sequence_number = len(
                self._cycles.query(fl_process_id=_fl_process.id, version=version)
            )

            _new_cycle = self._cycles.register(
                start=datetime.now(),
                end=datetime.now(),
                sequence=sequence_number + 1,
                version=version,
                fl_process_id=_fl_process,
            )

        return _new_cycle

    def get_cycle(self, fl_process_id: int, version: str = None):
        """ Retrieve a registered cycle.
            Args:
                model_id: Model's ID.
                version: Model's version.
            Returns:
                cycle: Cycle Instance / None
        """
        if version:
            _cycle = self._cycles.last(fl_process_id=fl_process_id, version=version)
        else:
            _cycle = self._cycles.last(fl_process_id=fl_process_id)

        if not _cycle:
            raise CycleNotFoundError

        return _cycle

    def get_protocol(self, **kwargs):
        _protocol = self._protocols.first(**kwargs)
        if not _protocol:
            raise ProtocolNotFoundError

        return _protocol

    def get_plan(self, **kwargs):
        _protocol = self._plans.first(**kwargs)
        if not _protocol:
            raise PlanNotFoundError

        return _protocol

    def get_model(self, **kwargs):
        _model = self._models.first(**kwargs)

        if not _model:
            raise ModelNotFoundError

        return _model

    def get_model_checkpoint(self, **kwargs):
        _check_point = self._model_checkpoints.last(**kwargs)

        if not _check_point:
            raise ModelNotFoundError

        return _check_point

    def validate(self, worker_id: str, cycle_id: str, request_key: str):
        _worker_cycle = self._worker_cycle.first(worker_id=worker_id, cycle_id=cycle_id)

        if not _worker_cycle:
            raise ProcessFoundError

        return _worker_cycle.request_key == request_key

    def delete_cycle(self, **kwargs):
        """ Delete a registered Cycle.
            Args:
                model_id: Model's ID.
        """
        self._cycles.delete(**kwargs)

    def last_participation(self, worker_id: str, name: str, version: str) -> int:
        """ Retrieve the last time the worker participated from this cycle.
            Args:
                worker_id: Worker's ID.
                name: Federated Learning Process Name.
                version: Model's version.
            Return:
                last_participation: Index of the last cycle assigned to this worker.
        """
        _fl_process = self._processes.first(name=name, version=version)
        _cycles = self._cycles.query(fl_process_id=_fl_process.id)

        last = 0
        if not len(_cycles):
            return last

        for cycle in _cycles:
            worker_cycle = self._worker_cycle.first(
                cycle_id=cycle.id, worker_id=worker_id
            )
            if worker_cycle and cycle.sequence > last:
                last = cycle.sequence

        return last

    def assign(self, name: str, version: str, worker, last_participation: int):
        """ Assign a new worker  the speficied federated training worker cycle
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
            _fl_process = self._processes.first(name=name, version=version)
        else:
            _fl_process = self._processes.last(name=name)

        # Retrieve model to tracked federated learning process id
        _model = self._models.first(fl_process_id=_fl_process.id)

        # Retrieve server configs
        server = self._configs.first(
            fl_process_id=_fl_process.id, is_server_config=True
        )

        # Retrieve the last cycle used by this fl process/ version
        _cycle = self.get_cycle(_fl_process.id, None)

        # Retrieve an WorkerCycle instance if this worker is already registered on this cycle.
        _worker_cycle = self._worker_cycle.query(
            worker_id=worker.id, cycle_id=_cycle.id
        )

        # Check bandwith
        _comp_bandwith = (
            worker.avg_upload > server.config["minimum_upload_speed"]
        ) and (worker.avg_download > server.config["minimum_download_speed"])

        # Check if the current worker is allowed to join into this cycle
        _allowed = (
            last_participation + server.config["do_not_reuse_workers_until_cycle"]
            >= _cycle.sequence
        )

        _accepted = (not _worker_cycle) and _comp_bandwith and _allowed
        if _accepted:
            _worker_cycle = self._worker_cycle.register(
                worker=worker,
                cycle=_cycle,
                request_key=self._generate_hash_key(uuid.uuid4().hex),
            )
            # Create a plan dictionary
            _plans = {
                plan.name: plan.id
                for plan in self._plans.query(
                    fl_process_id=_fl_process.id, is_avg_plan=False
                )
            }
            # Create a protocol dictionary
            _protocols = {
                protocol.name: protocol.id
                for protocol in self._protocols.query(fl_process_id=_fl_process.id)
            }

            return {
                CYCLE.STATUS: "accepted",
                CYCLE.KEY: _worker_cycle.request_key,
                MSG_FIELD.MODEL: name,
                CYCLE.PLANS: _plans,
                CYCLE.PROTOCOLS: _protocols,
                CYCLE.CLIENT_CONFIG: self._configs.first(
                    fl_process_id=_fl_process.id, is_server_config=False
                ).config,
                MSG_FIELD.MODEL_ID: _model.id,
            }
        else:
            remaining = _cycle.end - datetime.now()
            return {CYCLE.STATUS: "rejected", CYCLE.TIMEOUT: str(remaining)}

    def _generate_hash_key(self, primary_key: str) -> str:
        """ Generate SHA256 Hash to give access to the cycle.
            Args:
                primary_key : Used to generate hash code.
            Returns:
                hash_code : Hash in string format.
        """
        return hashlib.sha256(primary_key.encode()).hexdigest()

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
        """

        # Register a new FL Process
        name = client_config["name"]
        version = client_config["version"]

        # Check if already exists
        if self._processes.contains(name=name, version=version):
            raise FLProcessConflict

        fl_process = self._processes.register(name=name, version=version)

        # Register new model
        _model = self._models.register(flprocess=fl_process)

        # Save model initial weights into ModelCheckpoint
        self._model_checkpoints.register(values=model, model=_model)

        # Register new Protocols into the database
        for key, value in client_protocols.items():
            self._protocols.register(
                name=key, value=value, protocol_flprocess=fl_process
            )

        # Register the average plan into the database
        self._plans.register(
            value=server_averaging_plan, avg_flprocess=fl_process, is_avg_plan=True
        )

        # Register new Plans into the database
        for key, value in client_plans.items():
            self._plans.register(name=key, value=value, plan_flprocess=fl_process)

        # Register the client/server setup configs
        self._configs.register(config=client_config, server_flprocess_config=fl_process)

        self._configs.register(
            config=server_config,
            is_server_config=True,
            client_flprocess_config=fl_process,
        )

        # Create a new cycle
        _now = datetime.now()
        _end = _now + timedelta(seconds=server_config["cycle_length"])
        self._cycles.register(
            start=_now, end=_end, sequence=0, version=None, cycle_flprocess=fl_process
        )
        return fl_process

    def delete_process(self, **kwargs):
        """ Remove a registered federated learning process.
            Args:
                pid : Id used identify the desired process. 
        """
        self._processes.delete(**kwargs)

    def get_process(self, **kwargs):
        """ Retrieve the desired federated learning process.
            Args:
                pid : Id used to identify the desired process.
            Returns:
                process : FLProcess Instance or None if it wasn't found.
        """
        _process = self._processes.query(**kwargs)

        if not _process:
            raise ProtocolNotFoundError

        return _process
