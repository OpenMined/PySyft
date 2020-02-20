import uuid
import hashlib
from .federated_learning_process import FLProcess


class FederatedLearningCycle:
    """ An abstraction of a federated learning process cycle """

    def __init__(
        self,
        fl_process: "FLProcess",
        model_id: str,
        sequence_number: int,
        cycle_time: int = 2500,
    ):
        """ Create a federated learning cycle instance.
            Args:
                fl_process: Federated Learning Process.
                cycle_time: Remaining time to execute this cycle.
                sequence_number: Sequence number of the current cycle.
                model_id: Model's ID.
                version: Model's version.
        """
        self.fl_process = fl_process
        self.cycle_time = cycle_time
        self._sequence_number = sequence_number
        self._workers = []

    @property
    def hash(self) -> str:
        """ Generate  and store hash code as a form of "authenticating" the download requests.
            *** This is specific to the relationship between the worker AND the cycle
            and cannot be reused for future cycles or other workers.***
            Args:
                worker_id: Worker's ID.
            Returns:
                hash_code: SHA256 code in string format.
        """
        return self._generate_hash_key(uuid.uuid4())

    def contains(self, worker_id):
        """ Check if a specific worker already exist on this cycle.
            Args:
                worker_id: Worker's ID.
            Returns:
                result: Boolean flag.
        """
        return worker_id in self._workers

    def assign(
        self,
        worker_id: str,
        up_speed: float,
        down_speed: float,
        last_participation: int,
    ) -> bool:
        """ Include a new worker in this cycle.
            
            Args:
                worker_id:  Worker's ID.
                up_speed: Worker's upload speed.
                down_speed: Worker's download speed.
            Returns:
                result: Boolean flag.
        """
        # Check if worker was assigned previously
        _contains = self.contains(worker_id)

        # Check if upload/download rate are conformable
        _compatible = self.accept(up_speed, down_speed)

        # Check if the current worker is allowed to join into this cycle
        _process = self.fl_process
        _allowed = (
            last_participation
            + _process.server_config["do_not_reuse_workers_until_cycle"]
            >= self._sequence_number
        )

        # If everything is ok, add the new worker.
        if _contains and _compatible and _allowed:
            self._workers.append(worker_id)

        return _contains and _compatible and _allowed

    def accept(self, up_speed: float, down_speed: float) -> bool:
        """ Check upload/download speed to accept/reject new workers.
            Args:
                up_speed: Worker's upload speed.
                down_speed: Worker's download speed.
            Returns:
                result: Boolean flag.
        """
        server_config = self.fl_process.server_config

        return (up_speed > server_config["minimum_upload_speed"]) and (
            down_speed > server_config["minimum_download_speed"]
        )

    def _generate_hash_key(self, primary_key: str) -> str:
        """ Generate SHA256 Hash to indentify this Cycle.
            Args:
                primary_key : Used to generate hash code.
            Returns:
                hash_code : Hash in string format.
        """
        return hashlib.sha256(bytes(primary_key)).hexdigest()
