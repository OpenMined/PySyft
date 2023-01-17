# stdlib
from typing import Any
from typing import Optional

# third party
from nacl.signing import SigningKey
from pydantic import BaseSettings
from typing_extensions import final

# relative
from ...logger import traceback_and_raise
from ..common.serde.serializable import serializable
from ..io.location import Location
from ..io.location import SpecificLocation
from .common.node import Node
from .common.node_manager.dict_store import DictStore
from .service import VMServiceClass
from .vm_client import VirtualMachineClient


@serializable(recursive_serde=True)
@final
class VirtualMachine(Node):
    client_type = VirtualMachineClient
    vm: SpecificLocation  # redefine the type of self.vm to not be optional
    signing_key: Optional[SigningKey]
    child_type_client_type = None

    def __init__(
        self,
        *,  # Trasterisk
        name: Optional[str] = None,
        network: Optional[Location] = None,
        domain: Optional[Location] = None,
        device: Optional[Location] = None,
        vm: SpecificLocation = SpecificLocation(),
        signing_key: Optional[SigningKey] = None,
        store_type: type = DictStore,
        settings: Optional[BaseSettings] = None,
    ):
        super().__init__(
            name=name,
            network=network,
            domain=domain,
            device=device,
            vm=vm,
            signing_key=signing_key,
            store_type=store_type,
        )

        # specific location with name
        self.vm = SpecificLocation(name=self.name)

        # relative
        from .common.node_service.vm_request_service.vm_service import (
            VMRequestAnswerService,
        )

        self.immediate_services_with_reply.append(VMServiceClass)
        self.immediate_services_with_reply.append(VMRequestAnswerService)
        # All node subclasses have to call this at the end of their __init__
        self._register_services()
        self.post_init()

    def post_init(self) -> None:
        super().post_init()

    @property
    def icon(self) -> str:
        return "🍰"

    def _register_frameworks(self) -> None:
        traceback_and_raise(NotImplementedError)
        # TODO: it doesn't at the moment but it needs to in the future,
        #  mostly because nodes should be able to choose waht framweorks they
        #  want to support (and more importantly what versions of those frameworks
        #  they want to support).
        # QUESTION: Does this exist?
        # from ....lib import supported_frameworks
        # for fw in supported_frameworks:
        #     for name, ast in fw.ast.attrs.items():
        #         if name in self.frameworks.attrs:
        #             traceback_and_raise(KeyError(
        #                 "Framework already imported. Why are you importing it twice?"
        #             ))
        #         self.frameworks.attrs[name] = ast

    def __hash__(self) -> int:
        return hash(self.vm.id)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, VirtualMachine):
            return False

        if self.node_uid != other.node_uid:
            return False

        return True
