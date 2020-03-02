from syft.generic import ObjectConstructor
import torch as th


class TensorConstructor(ObjectConstructor):
    def underlying_framework_init(self, *args, **kwargs):
        return th.tensor(*args)
