from ...decorators import syft_decorator
from ...common.id import UID
from typing import final

@final
class PublicAddress(object):
    @syft_decorator(typechecking=True)
    def __init__(self, network: (str, UID), domain: (str, UID)):
        self.network = network
        self.domain = domain


@final
class PrivateAddress(object):
    @syft_decorator(typechecking=True)
    def __init__(self, device: (str, UID), vm: (str, UID)):
        self.device = device
        self.vm = vm


@final
class Address(object):
    @syft_decorator(typechecking=True)
    def __init__(self, pub_address: PublicAddress, pri_address: PrivateAddress):
        self.pub_address = pub_address
        self.pri_address = pri_address


@syft_decorator(typechecking=True)
def address(network: (str, UID), domain: (str, UID), device: (str, UID), vm: (str, UID)) -> Address:
    """A convenience method for creating routes"""

    pub = PublicAddress(network=network, domain=domain)
    pri = PrivateAddress(device=device, vm=vm)

    return Address(pub_address=pub, pri_address=pri)
