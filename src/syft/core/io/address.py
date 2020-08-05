from typing import Union, Optional
from typing_extensions import final

from syft.core.common.uid import UID

from ...decorators import syft_decorator


@final
class PublicAddress(object):
    @syft_decorator(typechecking=True)
    def __init__(
        self, network: Optional[Union[str, UID]], domain: Optional[Union[str, UID]]
    ):
        self.network = network
        self.domain = domain


@final
class PrivateAddress(object):
    @syft_decorator(typechecking=True)
    def __init__(
        self, device: Optional[Union[str, UID]], vm: Optional[Union[str, UID]]
    ):
        self.device = device
        self.vm = vm


@final
class Address(object):
    @syft_decorator(typechecking=True)
    def __init__(self, pub_address: PublicAddress, pri_address: PrivateAddress):
        self.pub_address = pub_address
        self.pri_address = pri_address

    @property
    def target_id(self) -> Union[str, UID]:
        """Return the address of the node which lives at this address.

        Note that this id is simply the most granular id available to the
        address."""
        if self.pri_address.vm is not None:
            return self.pri_address.vm
        elif self.pri_address.device is not None:
            return self.pri_address.device
        elif self.pub_address.domain is not None:
            return self.pub_address.domain
        elif self.pub_address.network is not None:
            return self.pub_address.network

        raise Exception("Address has no valid parts")

    def __repr__(self) -> str:
        out = ""
        out += f" Network:{self.pub_address.network},"  # OpenGrid
        out += f" Domain:{self.pub_address.domain}) "  # UCSF
        out += f" Device:{self.pri_address.device},"  # One of UCSF's Dell Servers
        out += f" VM:{self.pri_address.vm})"  # 8GB RAM set aside @Trask - UCSF-Server-5
        return out


@syft_decorator(typechecking=True)
def address(
    network: Optional[Union[str, UID]],
    domain: Optional[Union[str, UID]],
    device: Optional[Union[str, UID]],
    vm: Optional[Union[str, UID]],
) -> Address:
    """A convenience method for creating routes"""

    pub = PublicAddress(network=network, domain=domain)
    pri = PrivateAddress(device=device, vm=vm)

    return Address(pub_address=pub, pri_address=pri)
