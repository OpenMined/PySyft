# stdlib
from typing import List
from typing import Optional

# third party
from nacl.signing import SigningKey
from nacl.signing import VerifyKey

# relative
from ...common.message import SignedImmediateSyftMessageWithoutReply
from ...io.location import Location
from ...io.location import SpecificLocation
from ..common.action.common import Action
from .vm import VirtualMachine


class PlanVirtualMachine(VirtualMachine):  # type: ignore
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
        self.record = False
        self.recorded_actions: List[Action] = []

    def recv_immediate_msg_without_reply(
        self, msg: SignedImmediateSyftMessageWithoutReply
    ) -> None:
        if self.record and isinstance(msg.message, Action):
            self.recorded_actions.append(msg.message)
        super().recv_immediate_msg_without_reply(msg)

    def record_actions(self) -> None:
        self.record = True

    def stop_recording(self) -> None:
        self.record = False
