import abc

from .warehouse import Warehouse


class ProcessManager(metaclass=abc.ABCMeta):
    """Abstract Process Manager for both Model and Data Centric Federated
    Learning."""

    def __init__(self, FLProcess, Config) -> None:
        self._processes = Warehouse(FLProcess)
        self._configs = Warehouse(Config)

    @abc.abstractmethod
    def create(
        self,
        client_config,
        client_plans,
        client_protocols,
        server_config,
        server_avg_plan,
    ):
        """ Register a new federated learning process
            Args:
                client_config: the client configurations
                client_plans: an object containing syft plans.
                client_protocols: an object containing syft protocols.
                server_config: the server configurations
                server_avg_plan: a function that will instruct PyGrid on how to average model
                    diffs that are returned from the workers.
            Returns:
                process : FLProcess Instance.
            Raises:
                FLProcessConflict (PyGridError) : If Process Name/Version already exists.
        """
        pass

    @abc.abstractmethod
    def get_configs(self, **kwargs):
        """Return FL Process Configs.

        Args:
            query: Query attributes used to identify and retrieve the FL Process.
        Returns:
            configs (Tuple) : Tuple of Process Configs (Server, Client)
        Raises:
            ProcessFoundError (PyGridError) : If FL Process not found.
        """
        pass

    @abc.abstractmethod
    def get_plans(self, **kwargs):
        """Return FL Process Plans.

        Args:
            query: Query attributes used to identify and retrieve the Plans.
        Returns:
            plans (Dict) : Dict of Plans
        Raises:
            PlanNotFoundError (PyGridError) : If Plan not found.
        """
        pass

    @abc.abstractmethod
    def get_plan(self, **kwargs):
        pass

    @abc.abstractmethod
    def get_protocols(self, **kwargs):
        """Return FL Process Protocols.

        Args:
            query: Query attributes used to identify and retrieve the FL Process.
        Returns:
            plans (Dict) : Dict of Protocols
        Raises:
            ProtocolNotFoundError (PyGridError) : If Protocol not found.
        """
        pass

    @abc.abstractmethod
    def get(self, **kwargs):
        """Retrieve the desired federated learning process.

        Args:
            query : query used to identify the desired process.
        Returns:
            process : FLProcess Instance or None if it wasn't found.
        Raises:
            ProcessNotFoundError (PyGridError) : If Process not found.
        """
        pass

    @abc.abstractmethod
    def first(self, **kwargs):
        """Retrieve the desired federated learning process.

        Args:
            query : query used to identify the desired process.
        Returns:
            process : FLProcess Instance or None if it wasn't found.
        Raises:
            ProcessNotFoundError (PyGridError) : If Process not found.
        """
        pass

    @abc.abstractmethod
    def last(self, **kwargs):
        """Retrieve the desired federated learning process.

        Args:
            query : query used to identify the desired process.
        Returns:
            process : FLProcess Instance or None if it wasn't found.
        Raises:
            ProcessNotFound (PyGridError) : If Process not found.
        """
        pass

    @abc.abstractmethod
    def delete(self, **kwargs):
        """Delete a registered Process.

        Args:
            model_id: Model's ID.
        """
        pass
