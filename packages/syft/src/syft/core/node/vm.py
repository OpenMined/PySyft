# stdlib
from typing import Any
from typing import Optional
from typing import Union

# third party
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
from pydantic import BaseSettings
from typing_extensions import final

# relative
from ...logger import critical
from ...logger import traceback_and_raise
from ..common.message import SignedMessage
from ..common.message import SyftMessage
from ..common.uid import UID
from ..io.location import Location
from ..io.location import SpecificLocation
from .common.node import Node
from .common.node_manager.dict_store import DictStore
from .service import VMServiceClass
from .vm_client import VirtualMachineClient


@final
class VirtualMachine(Node):
    client_type = VirtualMachineClient
    vm: SpecificLocation  # redefine the type of self.vm to not be optional
    signing_key: Optional[SigningKey]
    verify_key: Optional[VerifyKey]
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
        verify_key: Optional[VerifyKey] = None,
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
            verify_key=verify_key,
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
        Node.set_keys(node=self)
        super().post_init()

    @property
    def icon(self) -> str:
        return "🍰"

    @property
    def id(self) -> UID:
        return self.vm.id

    def message_is_for_me(self, msg: Union[SyftMessage, SignedMessage]) -> bool:
        # this needs to be defensive by checking vm_id NOT vm.id or it breaks
        try:
            return msg.address.vm_id == self.id
        except Exception as e:
            critical(f"Error checking if {msg.pprint} is for me on {self.pprint}. {e}")
            return False

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

        if self.vm.id != other.vm.id:
            return False

        return True
