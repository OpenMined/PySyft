# stdlib
from functools import lru_cache
from typing import Any
from typing import Optional
from typing import Tuple

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
import numpy as np

# syft absolute
from syft.core.common.serde.deserialize import _deserialize as deserialize
from syft.core.common.serde.serializable import Serializable
from syft.core.common.serde.serializable import bind_protobuf
from syft.core.common.serde.serialize import _serialize as serialize
from syft.core.tensor.passthrough import PassthroughTensor
from syft.proto.core.tensor.share_tensor_pb2 import ShareTensor as ShareTensor_PB


@bind_protobuf
class ShareTensor(PassthroughTensor, Serializable):
    def __init__(
        self,
        rank: int,
        ring_size: int = 2 ** 64,
        value: Optional[Any] = None,
        seed_ids: Optional[int] = None,
    ) -> None:
        if seed_ids is None:
            self.seed_ids = 42
        else:
            self.seed_ids = seed_ids

        self.generator_ids = np.random.default_rng(self.seed_ids)
        self.rank = rank
        self.ring_size = ring_size
        self.min_value, self.max_value = ShareTensor.compute_min_max_from_ring(
            self.ring_size
        )
        super().__init__(value)

    @staticmethod
    @lru_cache(32)
    def compute_min_max_from_ring(ring_size: int = 2 ** 64) -> Tuple[int, int]:
        min_value = (-ring_size) // 2
        max_value = (ring_size - 1) // 2
        return min_value, max_value

    """ TODO: Remove this -- we would use generate_przs since the scenario we are testing is that
    the secret is remotly
    @staticmethod
    def generate_shares(secret, nr_shares, ring_size=2 ** 64):
        # syft relative
        from .fixed_precision_tensor import FixedPrecisionTensor

        if not isinstance(secret, (int, FixedPrecisionTensor)):
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
    """

    @staticmethod
    def generate_przs(
        value: Optional[Any],
        shape: Tuple[int],
        rank: int,
        nr_parties: int,
        seed_shares: int,
    ) -> "ShareTensor":
        # syft absolute
        from syft.core.tensor.tensor import Tensor

        if value is None:
            value = Tensor(np.zeros(shape, dtype=np.int64))

        # TODO: Sending the seed and having each party generate the shares is not safe
        # Since the parties would know some of the other parties shares (this might not impose a risk
        # when shares are not sent between parties -- like private addition/substraction, but it might
        # impose for multiplication
        # The secret holder should generate the shares and send them to the other parties
        generator_shares = np.random.default_rng(seed_shares)

        share = value.child
        if not isinstance(share, ShareTensor):
            share = ShareTensor(value=share, rank=rank)

        shares = [
            generator_shares.integers(low=share.min_value, high=share.max_value)
            for _ in range(nr_parties)
        ]
        share.child += shares[rank] - shares[(rank + 1) % nr_parties]
        return share

    def __add__(self, other: Any) -> "ShareTensor":
        if isinstance(other, ShareTensor):
            return ShareTensor(value=self.child + other.child, rank=self.rank)
        else:
            raise ValueError("Expected other to be ShareTensor")

    def __sub__(self, other: Any) -> "ShareTensor":
        if isinstance(other, ShareTensor):
            return ShareTensor(value=self.child - other.child, rank=self.rank)
        else:
            raise ValueError("Expected other to be ShareTensor")

    def __mul__(self, other: Any) -> "ShareTensor":
        if isinstance(other, ShareTensor):
            raise ValueError("Private Multiplication not yet implemented")
        else:
            return ShareTensor(value=self.child * other, rank=self.rank)

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
