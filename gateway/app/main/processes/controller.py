from .federated_learning_process import FLProcess
from .federated_learning_cycle import FederatedLearningCycle


class FLController:
    """ This class implements controller design pattern over the federated learning processes. """

    def __init__(self):
        self.processes = {}
        self._cycles = {}

    def create_cycle(self, model_id: str, version: str, cycle_time: int = 2500):
        """ Create a new federated learning cycle.
            Args:
                model_id: Model's ID.
                worker_id: Worker's ID.
                cycle_time: Remaining time to finish this cycle.
            Returns:
                fd_cycle: Cycle Instance.
        """
        _fl_process = self.processes.get(model_id, None)

        if _fl_process:
            # Retrieve a list of cycles using the same model_id/version
            cycle_sequence = self._cycles.get((model_id, version), None)

            # If already exists, create and append a new cycle
            if cycle_sequence:
                new_cycle = FederatedLearningCycle(
                    _fl_process, cycle_time, len(cycle_sequence) + 1
                )
                self.cycle_sequence.append(new_cycle)
            else:  # otherwise, create a new list
                new_cycle = FederatedLearningCycle(
                    _fl_process, cycle_time, 1
                )  # Create the first one.
                self._cycles[(model_id, version)] = [new_cycle]

    def get_cycle(self, model_id: str, version: str, sequence_number: int = None):
        """ Retrieve a registered cycle.
            Args:
                model_id: Model's ID.
                version: Model's version.
                sequence_number: Cycle index.
            Returns:
                cycle: Cycle Instance / None
        """
        # Retrive a list of cycles used by this model_id/version
        _cycle_sequences = self._cycles.get((model_id, version), None)

        # Select the cycle (By default, will return the last cycle)
        cycle_index = sequence_number if sequence_number else len(_cycle_sequences) - 1

        if _cycle_sequences:
            return _cycle_sequences[cycle_index]

    def delete_cycle(self, model_id: str, version: str):
        """ Delete a registered Cycle.
            Args:
                model_id: Model's ID.
        """
        if model_id in self._cycles:
            del self._cycles[(model_id, version)]

    def last_participation(self, worker_id: str, model_id: str, version: str) -> int:
        """ Retrieve the last time the worker participated from this cycle.
            Args:
                worker_id: Worker's ID.
                model_id: Model's ID.
                version: Model's version.
            Return:
                last_participation: Index of the last cycle assigned to this worker.
        """
        _cycle_sequences = self._cycles.get((model_id, version), None)

        if _cycle_sequences:
            for i in range(len(_cycle_sequences)):
                if _cycle_sequences[i].contains(worker_id):
                    return i
        return 0

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
                model: The model that will be hosted.
                client_plans : an object containing syft plans.
                client_protocols : an object containing syft protocols.
                client_config: the client configurations
                server_averaging_plan: a function that will instruct PyGrid on how to average model diffs that are returned from the workers.
                server_config: the server configurations
            Returns:
                process : FLProcess Instance.
        """
        process = FLProcess(
            model=model,
            client_plans=client_plans,
            client_config=client_config,
            server_config=server_config,
            client_protocols=client_protocols,
            server_averaging_plan=server_averaging_plan,
        )

        self.processes[process.id] = process
        return self.processes[process.id]

    def delete_process(self, pid):
        """ Remove a registered federated learning process.
            Args:
                pid : Id used identify the desired process. 
        """
        del self.processes[pid]

    def get_process(self, pid):
        """ Retrieve the desired federated learning process.
            Args:
                pid : Id used to identify the desired process.
            Returns:
                process : FLProcess Instance or None if it wasn't found.
        """
        return self.processes.get(pid, None)
