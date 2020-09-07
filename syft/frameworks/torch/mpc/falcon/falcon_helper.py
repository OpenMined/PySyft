import torch
import syft

from syft.frameworks.torch.tensors.interpreters.replicated_shared import ReplicatedSharingTensor
from typing import Union


class FalconHelper:
    @classmethod
    def unfold(cls, image, kernel_size, padding):
        return cls.__switch_public_private(
            image, cls.__public_unfold, cls.__private_unfold, kernel_size, padding
        )

    @staticmethod
    def __public_unfold(image, kernel_size, padding):
        image = image.double()
        image = torch.nn.functional.unfold(image, kernel_size=kernel_size, padding=padding)
        image = image.long()
        return image

    @staticmethod
    def __private_unfold(image, kernel_size, padding):
        return image.unfold(kernel_size, padding)

    @staticmethod
    def xor(
        value: ReplicatedSharingTensor, other: Union[int, ReplicatedSharingTensor, torch.tensor]
    ) -> ReplicatedSharingTensor:
        assert value.ring_size == 2
        return value + other - 2 * value * other

    @classmethod
    def select_shares(
        cls,
        b: ReplicatedSharingTensor,
        x: ReplicatedSharingTensor,
        y: ReplicatedSharingTensor,
    ) -> ReplicatedSharingTensor:
        """
        return: x if b=0 | y if b=1
        """
        c_2, c_l = cls.__generate_random_bit(x.players, ring_sizes=[2, x.ring_size])
        b_xor_c = FalconHelper.xor(b, c_2).reconstruct()
        d = c_l * (1 - 2 * b_xor_c) + b_xor_c

        selected_val = (y - x) * d + x
        return selected_val

    @staticmethod
    def __generate_random_bit(players: list, ring_sizes: list) -> list:
        """
        generates a random bit and and shares it in arbitrary number of ring_sizes
        return: list [random bit shared in ring_size i]
        """
        bit = torch.randint(high=min(ring_sizes), size=[1])
        return [bit.share(*players, protocol="falcon", field=ring_size) for ring_size in ring_sizes]

    @staticmethod
    def determine_sign(
        x: ReplicatedSharingTensor, beta: ReplicatedSharingTensor
    ) -> ReplicatedSharingTensor:
        """
        return: x if beta = 0 | -x if beta = 1
        """
        return (1 - beta * 2) * x

    @staticmethod
    def private_compare_preprocess(players, p=7, k=10):
        beta = torch.randint(high=1, size=[1]).share(*players, protocol="falcon", field=2)
        beta_p = beta.inject_bit(p)
        m = []
        for _ in range(k):
            m_i = torch.tensor([0]).share(*players, protocol="falcon", field=p).rand_()
            if ((m_i ** (p - 1)).reonsturct() == torch.tensor([1])).all():
                m.append(m_i)
        return {"beta": [beta, beta_p], "m": m}

    @staticmethod
    def __switch_public_private(value, public_function, private_function, *args, **kwargs):
        if isinstance(value, (int, float, torch.Tensor, syft.FixedPrecisionTensor)):
            return public_function(value, *args, **kwargs)
        elif isinstance(value, syft.ReplicatedSharingTensor):
            return private_function(value, *args, **kwargs)
        else:
            raise ValueError(
                "expected int, float, torch tensor, or ReplicatedSharingTensor "
                "but got {}".format(type(value))
            )
