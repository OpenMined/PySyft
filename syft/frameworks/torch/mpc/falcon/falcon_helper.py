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
        value: ReplicatedSharingTensor, other: Union[int, ReplicatedSharingTensor, torch.LongTensor]
    ) -> ReplicatedSharingTensor:
        """
        Compute the XOR value between value and other.
        If value and other are both ints we should use the "^" operator.

        Args:
            value (ReplicatedSharingTensor): RST with ring size of 2 or integer value in {0, 1}
            other (int): integer with value in {0, 1}

        Returns:
            The XOR computation between value and other
        """

        assert (
            isinstance(value, ReplicatedSharingTensor) and value.ring_size == 2
        ), "First argument should be a RST with ring size 2"
        assert any(
            [
                isinstance(other, ReplicatedSharingTensor) and other.ring_size == 2,
                isinstance(other, int) and other in {0, 1},
                isinstance(other, torch.LongTensor) and ((other == 0) + (other == 1)).all(),
            ]
        ), "Second argument should be RST (with ring size of 2)/Integer/LongTensor values in {0, 1}"

        return value + other - 2 * value * other

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

    @staticmethod
    def select_share(
        b: ReplicatedSharingTensor,
        x: ReplicatedSharingTensor,
        y: ReplicatedSharingTensor,
    ) -> ReplicatedSharingTensor:
        """Select x or y depending on b

        Args:
            x (ReplicatedSharingTensor): RST that will be selected if b reconstructed is 0
            y (ReplicatedSharingTensor): RST that will be selected if b reconstructed is 1
            b (ReplicatedSharingTensor): RST of a bit

        Return:
            x if b == 0 else y
        """

        ring_size = x.ring_size
        players = x.players
        shape = x.shape

        c = torch.randint(high=2, size=shape)
        c_2 = c.share(*players, protocol="falcon", field=2)
        c_L = c.share(*players, protocol="falcon", field=ring_size)

        xor_b_c = FalconHelper.xor(b, c_2).reconstruct()
        d = c_L * (1 - 2 * xor_b_c) + xor_b_c

        selected_val = (y - x) * d + x
        return selected_val

    def determine_sign(
        x: ReplicatedSharingTensor, beta: ReplicatedSharingTensor
    ) -> ReplicatedSharingTensor:
        """determines the sign of x,  positive if beta is 0 or negative if beta is 1

        Args:
            x (ReplicatedSharingTensor): RST to perform the computation on
            beta (ReplicatedSharingTensor): the reconstructed value should be in {0, 1}

        Return:
            returns x if beta=0, or -x if beta=1
        """
        ring_size = x.ring_size
        players = x.players
        shape = x.shape

        result = (1 - beta * 2) * x

        return result
