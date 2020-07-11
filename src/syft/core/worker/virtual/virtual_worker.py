from ..worker import Worker
from ..virtual.virtual_client import VirtualClient
from typing import final
from .. import service


class VirtualWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = VirtualClient(self.id, self, verbose=False)

    def _recv_msg(self, msg):
        return self.recv_msg(msg=msg)

    def get_client(self, debug=False):
        self._client.debug = debug
        return self._client

    def _register_services(self) -> None:
        services = list()
        services.append(service.get_object_service.GetObjectService)
        services.append(service.save_object_service.SaveObjectService)
        services.append(service.run_class_service.RunClassMethodService)
        services.append(service.delete_object_service.DeleteObjectService)
        services.append(service.run_function_or_constructor_service.RunFunctionOrConstructorService)

        for s in services:
            self.msg_router[s.message_type_handler()] = s()

    def _register_frameworks(self) -> None:

        from ....lib import supported_frameworks

        for fw in supported_frameworks:
            for name, ast in fw.ast.attrs.items():
                if name in self.frameworks.attrs:
                    raise KeyError(
                        "Framework already imported. Why are you importing it twice?"
                    )
                self.frameworks.attrs[name] = ast
