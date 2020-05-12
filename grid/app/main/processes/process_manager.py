# Processes module imports
from .fl_process import FLProcess
from .config import Config

# PyGrid imports
from ..storage.warehouse import Warehouse
from ..syft_assets import plans, protocols
from ..exceptions import (
    FLProcessConflict,
    ProcessFoundError,
    PlanNotFoundError,
    ProtocolNotFoundError,
)


class ProcessManager:
    def __init__(self):
        self._processes = Warehouse(FLProcess)
        self._configs = Warehouse(Config)

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
                name : FL Process name.
                version : Process version.
            Returns:
                process : FLProcess Instance.
            Raises:
                FLProcessConflict (PyGridError) : If Process Name/Version already exists.
        """

        name = client_config["name"]
        version = client_config["version"]

        # Check if already exists
        if self._processes.contains(name=name, version=version):
            raise FLProcessConflict

        # Create a new process
        fl_process = self._processes.register(name=name, version=version)

        # Register client protocols
        protocols.register(fl_process, client_protocols)

        # Register Server avg plan
        plans.register(fl_process, server_avg_plan, avg_plan=True)

        # Register client plans
        plans.register(fl_process, client_plans, avg_plan=False)

        # Register the client setup configs
        self._configs.register(config=client_config, server_flprocess_config=fl_process)

        # Register the server setup configs
        self._configs.register(
            config=server_config,
            is_server_config=True,
            client_flprocess_config=fl_process,
        )

        return fl_process

    def get_configs(self, **kwargs):
        """ Return FL Process Configs.
            Args:
                query: Query attributes used to identify and retrieve the FL Process.
            Returns:
                configs (Tuple) : Tuple of Process Configs (Server, Client)
            Raises:
                ProcessFoundError (PyGridError) : If FL Process not found.
        """
        _process = self._processes.last(**kwargs)

        if not _process:
            raise ProcessFoundError

        # Server configs
        server = self._configs.first(fl_process_id=_process.id, is_server_config=True)

        # Client configs
        client = self._configs.first(fl_process_id=_process.id, is_server_config=False)

        return (server.config, client.config)

    def get_plans(self, **kwargs):
        """ Return FL Process Plans.
            Args:
                query: Query attributes used to identify and retrieve the Plans.
            Returns:
                plans (Dict) : Dict of Plans
            Raises:
                PlanNotFoundError (PyGridError) : If Plan not found.
        """
        _plans = plans.get(**kwargs)

        if not _plans:
            raise PlanNotFoundError

        # Build a plan dictionary
        plan_dict = {_plan.name: _plan.id for _plan in _plans}

        return plan_dict

    def get_plan(self, **kwargs):
        return plans.first(**kwargs)

    def get_protocols(self, **kwargs):
        """ Return FL Process Protocols.
            Args:
                query: Query attributes used to identify and retrieve the FL Process.
            Returns:
                plans (Dict) : Dict of Protocols
            Raises:
                ProtocolNotFoundError (PyGridError) : If Protocol not found.
        """
        _protocols = protocols.get(**kwargs)

        if not _protocols:
            raise ProtocolNotFoundError

        # Build a protocol dictionary
        protocol_dict = {_protocol.name: _protocol.id for _protocol in _protocols}

        return protocol_dict

    def get(self, **kwargs):
        """ Retrieve the desired federated learning process.
            Args:
                query : query used to identify the desired process.
            Returns:
                process : FLProcess Instance or None if it wasn't found.
            Raises:
                ProcessNotFound (PyGridError) : If Process not found.
        """
        _process = self._processes.query(**kwargs)

        if not _process:
            raise ProcessFoundError

        return _process

    def first(self, **kwargs):
        """ Retrieve the desired federated learning process.
            Args:
                query : query used to identify the desired process.
            Returns:
                process : FLProcess Instance or None if it wasn't found.
            Raises:
                ProcessNotFound (PyGridError) : If Process not found.
        """
        _process = self._processes.first(**kwargs)

        if not _process:
            raise ProcessFoundError

        return _process

    def last(self, **kwargs):
        """ Retrieve the desired federated learning process.
            Args:
                query : query used to identify the desired process.
            Returns:
                process : FLProcess Instance or None if it wasn't found.
            Raises:
                ProcessNotFound (PyGridError) : If Process not found.
        """
        _process = self._processes.last(**kwargs)

        if not _process:
            raise ProcessFoundError

        return _process

    def delete(self, **kwargs):
        """ Delete a registered Process.
            Args:
                model_id: Model's ID.
        """
        self._processes.delete(**kwargs)
