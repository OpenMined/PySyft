import uuid


class FLProcess:
    """ An abstraction of a federated learning process """

    def __init__(
        self,
        model,
        client_plans,
        client_config,
        server_config,
        server_averaging_plan,
        client_protocols=None,
    ):
        """ Create a federated learning process instance.
            
            Args:
                model: The model that will be hosted.
                client_plans : an object containing syft plans.
                client_protocols : an object containing syft protocols.
                client_config: the client configurations
                server_averaging_plan: a function that will instruct PyGrid on how to average model diffs that are returned from the workers.
                server_config: the server configurations
        """

        self.id = str(uuid.uuid4())
        self.model = model
        self.client_plans = client_plans
        self.client_protocols = client_protocols
        self.client_config = client_config
        self.server_averaging_plan = server_averaging_plan
        self.server_config = server_config

    def json(self):
        """ Convert a federated learning process instances into a JSON/Dictionary structure.
            Returns:
                federated_learning_process: a JSON representation of a Federated Learning Process
        """

        return {
            "model": self.model,
            "client_plans": self.client_plans,
            "client_protocols": self.client_protocols,
            "client_config": self.client_config,
            "server_averaging_plan": self.server_averaging_plan,
            "server_config": self.server_config,
        }

    def __str__(self):
        return "< FLProcess - ID: " + self.id + " >"

    def __eq__(self, other):
        if isinstance(other, FLProcess):
            return self.id == other.id
        return False
