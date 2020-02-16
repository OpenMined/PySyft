from .federated_learning_process import FLProcess


class FLPController:
    """ This class implements controller design pattern over the federated learning processes."""

    def __init__(self):
        self.processes = {}

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
