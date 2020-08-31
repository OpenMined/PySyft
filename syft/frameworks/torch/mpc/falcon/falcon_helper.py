import torch
import syft


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
    def xor(value, other):
        return value + other - (value * 2 * other)

    @staticmethod
    def __switch_public_private(value, public_function, private_function, *args, **kwargs):
        if isinstance(value, (int, float, torch.Tensor, syft.FixedPrecisionTensor)):
            return public_function(value, *args, **kwargs)
        elif isinstance(value, syft.ReplicatedSharingTensor):
            return private_function(value, *args, **kwargs)
        else:
            raise ValueError(
                "expected int, float, torch tensor, or ReplicatedSharingTensor"
                "but got {}".format(type(value))
            )
