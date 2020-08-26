from syft.frameworks.torch.tensors.interpreters.replicated_shared import ReplicatedSharingTensor
from operator import xor


class FalconTensor(ReplicatedSharingTensor):
    def __init__(self, plain_text=None, players=None, ring_size=None, owner=None):
        super().__init__(plain_text=plain_text, players=players, ring_size=ring_size, owner=owner)

    def xor(self, value):
        return self.switch_public_private(value, self.__public_xor, self.__private_xor)

    def __public_xor(self, plain_text):
        return self + plain_text - (self * 2 * plain_text)

    def __private_xor(self, secret):
        return self._private_linear_operation(secret, xor)

    __xor__ = xor

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        type_name = type(self).__name__
        out = f"[" f"{type_name}]"
        if self.child is not None:
            for v in self.child.values():
                out += "\n\t-> " + str(v)
        return out
