# stdlib
from typing import Optional
from typing import Union

# third party
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
from typing_extensions import final

# syft relative
from ....decorators import syft_decorator
from ....logger import critical
from ....logger import traceback_and_raise
from ...common.message import SignedMessage
from ...common.message import SyftMessage
from ...common.uid import UID
from ...io.location import Location
from ...io.location import SpecificLocation
from ..common.node import Node
from .client import VirtualMachineClient


@final
class VirtualMachine(Node):
    client_type = VirtualMachineClient
    vm: SpecificLocation  # redefine the type of self.vm to not be optional
    signing_key: Optional[SigningKey]
    verify_key: Optional[VerifyKey]
    child_type_client_type = None

    @syft_decorator(typechecking=True)
    def __init__(
        self,
        name: Optional[str] = None,
        network: Optional[Location] = None,
        domain: Optional[Location] = None,
        device: Optional[Location] = None,
        vm: SpecificLocation = SpecificLocation(),
        signing_key: Optional[SigningKey] = None,
        verify_key: Optional[VerifyKey] = None,
    ):
        super().__init__(
            name=name,
            network=network,
            domain=domain,
            device=device,
            vm=vm,
            signing_key=signing_key,
            verify_key=verify_key,
        )

        # specific location with name
        self.vm = SpecificLocation(name=self.name)
        # syft relative
        from ..domain.service.vm_service import VMRequestAnswerMessageService
        from ..domain.service.vm_service import VMRequestService

        self.immediate_services_without_reply.append(VMRequestService)
        self.immediate_services_with_reply.append(VMRequestAnswerMessageService)
        # All node subclasses have to call this at the end of their __init__
        self._register_services()
        self.post_init()

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

    @syft_decorator(typechecking=True)
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
