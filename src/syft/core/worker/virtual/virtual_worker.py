from ..worker import Worker
from ..virtual.virtual_client import VirtualClient
from typing import final
from .. import service


class VirtualWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = VirtualClient(self.id, self, verbose=False)

    def _recv_msg(self, msg):
        return self.recv_msg(msg=msg)

    def get_client(self, verbose=False):
        self.client.verbose = verbose
        return self.client

    def _register_services(self) -> None:
        services = list()
        services.append(service.get_object_service.GetObjectService)
        services.append(service.save_object_service.SaveObjectService)
        services.append(service.run_class_service.RunClassMethodService)
        services.append(service.delete_object_service.DeleteObjectService)
        services.append(service.run_function_or_constructor_service.RunFunctionOrConstructorService)

        for s in services:
            self.msg_router[s.message_type_handler()] = s()
