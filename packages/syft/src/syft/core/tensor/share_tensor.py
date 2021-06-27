# stdlib
from functools import lru_cache

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
import numpy as np

# syft absolute
from syft.core.tensor.fixed_precision_tensor import FixedPrecisionTensor
from syft.core.tensor.passthrough import PassthroughTensor

# syft relative
from ...proto.core.tensor.share_tensor_pb2 import ShareTensor as ShareTensor_PB
from ..common.serde.deserialize import _deserialize as deserialize
from ..common.serde.serializable import bind_protobuf
from ..common.serde.serialize import _serialize as serialize
from ...core.common.serde.serializable import Serializable


@bind_protobuf
class ShareTensor(PassthroughTensor, Serializable):
    def __init__(
        self, rank, ring_size=2 ** 64, value=None, seed=None, seed_generators=None
    ):
        if seed_generators is None:
            self.seed_generators = [0, 1]
        else:
            self.seed_generators = seed_generators

        if seed is None:
            self.seed = 42
        else:
            self.seed = seed

        # TODO: This is not secure
        self.generators_przs = [
            np.random.default_rng(seed) for seed in self.seed_generators
        ]
        self.generator_ids = np.random.default_rng(self.seed)
        self.rank = rank
        self.ring_size = ring_size
        self.min_value, self.max_value = ShareTensor.compute_min_max_from_ring(
            self.ring_size
        )
        super().__init__(value)

    @staticmethod
    @lru_cache(32)
    def compute_min_max_from_ring(ring_size=2 ** 64):
        min_value = (-ring_size) // 2
        max_value = (ring_size - 1) // 2
        return min_value, max_value

    @staticmethod
    def generate_shares(secret, nr_shares, ring_size=2 ** 64):
        # syft relative
        from .fixed_precision_tensor import FixedPrecisionTensor

        if not isinstance(secret, FixedPrecisionTensor):
            secret = FixedPrecisionTensor(value=secret)

        shape = secret.shape
        min_value, max_value = ShareTensor.compute_min_max_from_ring(ring_size)

        generator_shares = np.random.default_rng()

        random_shares = []
        for i in range(nr_shares):
            random_value = generator_shares.integers(
                low=min_value, high=max_value, size=shape
            )
            fpt_value = FixedPrecisionTensor(value=random_value)
            random_shares.append(fpt_value)

        shares_fpt = []
        for i in range(nr_shares):
            if i == 0:
                share = value = random_shares[i]
            elif i < nr_shares - 1:
                share = random_shares[i] - random_shares[i - 1]
            else:
                share = secret - random_shares[i - 1]

            shares_fpt.append(share)

        # Add the ShareTensor class between them
        shares = []
        for rank, share_fpt in enumerate(shares_fpt):
            share_fpt.child = ShareTensor(rank=rank, value=share_fpt.child)
            shares.append(share_fpt)

        return shares

    @staticmethod
    def generate_przs(tensor, shape, rank):
        # syft absolute
        from syft.core.tensor.tensor import Tensor

        if tensor is None:
            tensor = Tensor(np.zeros(shape))

        fpt = tensor
        share = tensor.child
        if not isinstance(share, ShareTensor):
            fpt = FixedPrecisionTensor(value=share)
            fpt.child = ShareTensor(value=fpt.child, rank=rank)
            share = fpt.child

        share_1 = share.generators_przs[0].integers(
            low=share.min_value, high=share.max_value
        )
        share_2 = share.generators_przs[1].integers(
            low=share.min_value, high=share.max_value
        )
        share.child += share_1 - share_2

        return fpt

    def _object2proto(self) -> ShareTensor_PB:
        if isinstance(self.child, np.ndarray):
            return ShareTensor_PB(array=serialize(self.child), rank=self.rank)
        else:
            return ShareTensor_PB(tensor=serialize(self.child), rank=self.rank)

    @staticmethod
    def _proto2object(proto: ShareTensor_PB) -> "ShareTensor":
        if proto.HasField("tensor"):
            res = ShareTensor(rank=proto.rank, value=deserialize(proto.tensor))
        else:
            res = ShareTensor(rank=proto.rank, value=deserialize(proto.array))

        return res

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return ShareTensor_PB
