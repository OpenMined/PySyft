# future
from __future__ import annotations

# stdlib
import functools
from functools import lru_cache
import operator
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
import numpy as np
import torch

# relative
from ....proto.core.tensor.share_tensor_pb2 import ShareTensor as ShareTensor_PB  # type: ignore
from ...common.serde.deserialize import _deserialize as deserialize
from ...common.serde.serializable import serializable
from ...common.serde.serialize import _serialize as serialize
from ..passthrough import PassthroughTensor  # type: ignore

METHODS_FORWARD_ALL_SHARES = {
    "repeat",
    "copy",
    "diagonal",
    "flatten",
    "transpose",
    "partition",
    "resize",
    "ravel",
    "compress",
    "reshape",
    "squeeze",
    "swapaxes",
    "__pos__",
}
INPLACE_OPS = {"resize", "put"}


@serializable()
class ShareTensor(PassthroughTensor):
    def __init__(
        self,
        rank: int,
        nr_parties: int,
        ring_size: int = 2 ** 32,
        value: Optional[Any] = None,
    ) -> None:
        self.rank = rank
        self.ring_size = ring_size
        self.nr_parties = nr_parties
        self.min_value, self.max_value = ShareTensor.compute_min_max_from_ring(
            self.ring_size
        )
        super().__init__(value)

    def __getitem__(self, item: Union[str, int, slice]) -> ShareTensor:
        return ShareTensor(
            rank=self.rank,
            nr_parties=self.nr_parties,
            ring_size=self.ring_size,
            value=self.child[item],
        )

    def copy_tensor(self) -> ShareTensor:
        return ShareTensor(
            rank=self.rank, nr_parties=self.nr_parties, ring_size=self.ring_size
        )

    @staticmethod
    @lru_cache(32)
    def compute_min_max_from_ring(ring_size: int = 2 ** 64) -> Tuple[int, int]:
        min_value = (-ring_size) // 2
        max_value = ring_size // 2 - 1
        return min_value, max_value

    """ TODO: Remove this -- we would use generate_przs since the scenario we are testing is that
    the secret is remotly
    @staticmethod
    def generate_shares(secret, nr_shares, ring_size=2 ** 64):
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
        shape: Tuple[int, ...],
        rank: int,
        nr_parties: int,
        seed_shares: int,
    ) -> "ShareTensor":

        # relative
        from ..tensor import Tensor

        if value is None:
            value = Tensor(np.zeros(shape, dtype=np.int32))  # TODO: change to np.int64

        # TODO: Sending the seed and having each party generate the shares is not safe
        # Since the parties would know some of the other parties shares (this might not impose a risk
        # when shares are not sent between parties -- like private addition/subtraction, but it might
        # impose for multiplication
        # The secret holder should generate the shares and send them to the other parties
        generator_shares = np.random.default_rng(seed_shares)

        share = value.child
        if not isinstance(share, ShareTensor):
            share = ShareTensor(value=share, rank=rank, nr_parties=nr_parties)

        shares = [
            generator_shares.integers(
                low=share.min_value, high=share.max_value, size=shape
            )
            for _ in range(nr_parties)
        ]
        share.child += shares[rank] - shares[(rank + 1) % nr_parties]

        return share

    @staticmethod
    def generate_przs_on_dp_tensor(
        value: Optional[Any],
        shape: Tuple[int],
        rank: int,
        nr_parties: int,
        seed_shares: int,
        share_wrapper: Any,
    ) -> PassthroughTensor:

        if value is not None:
            share = ShareTensor.generate_przs(
                value=value.child,
                shape=shape,
                rank=rank,
                nr_parties=nr_parties,
                seed_shares=seed_shares,
            )
        else:
            share = ShareTensor.generate_przs(
                value=value,
                shape=shape,
                rank=rank,
                nr_parties=nr_parties,
                seed_shares=seed_shares,
            )

        share_wrapper.child.child = share

        return share_wrapper

    @staticmethod
    def sanity_check(
        share: Union[int, float, torch.Tensor, np.ndarray, "ShareTensor"]
    ) -> None:
        """Check type for share

        Args:
            share (Union[int, float, ShareTensor, np.ndarray, torch.Tensor]): value to check

        Raises:
            ValueError: if type is not supported
        """
        if isinstance(share, float):
            raise ValueError("Type float not supported yet!")

        if isinstance(share, np.ndarray) and not np.issubdtype(share.dtype, np.integer):
            raise ValueError(f"NPArray should have type int, but found {share.dtype}")

        if isinstance(share, torch.Tensor) and torch.is_floating_point(share):
            raise ValueError("Torch tensor should have type int, but found float")

    def apply_function(
        self, y: Union[int, float, torch.Tensor, np.ndarray, "ShareTensor"], op_str: str
    ) -> "ShareTensor":
        """Apply a given operation.

        Args:
            y (Union[int, float, torch.Tensor, np.ndarray, "ShareTensor"]): tensor to apply the operator.
            op_str (str): Operator.

        Returns:
            ShareTensor: Result of the operation.
        """

        op = getattr(operator, op_str)
        if isinstance(y, ShareTensor):
            value = op(self.child, y.child)
        else:
            # TODO: Converting y to numpy because doing "numpy op torch tensor" raises exception
            value = op(self.child, np.array(y, np.int32))  # TODO: change to np.int64

        res = self.copy_tensor()
        res.child = value
        return res

    def add(
        self, y: Union[int, float, torch.Tensor, np.ndarray, "ShareTensor"]
    ) -> "ShareTensor":
        """Apply the "add" operation between "self" and "y".

        Args:
            y (Union[int, float, torch.Tensor, np.ndarray, "ShareTensor"]): self + y

        Returns:
            ShareTensor. Result of the operation.
        """

        ShareTensor.sanity_check(y)

        new_share = self.apply_function(y, "add")

        return new_share

    def sub(
        self, y: Union[int, float, torch.Tensor, np.ndarray, "ShareTensor"]
    ) -> "ShareTensor":
        """Apply the "sub" operation between "self" and "y".

        Args:
            y (Union[int, float, torch.Tensor, np.ndarray, "ShareTensor"]): self - y

        Returns:
            ShareTensor. Result of the operation.
        """

        ShareTensor.sanity_check(y)
        new_share = self.apply_function(y, "sub")
        return new_share

    def rsub(
        self, y: Union[int, float, torch.Tensor, np.ndarray, "ShareTensor"]
    ) -> "ShareTensor":
        """Apply the "rsub" operation between "self" and "y"

        Args:
            y (Union[int, float, torch.Tensor, np.ndarray, "ShareTensor"]): y - self

        Returns:
            ShareTensor. Result of the operation.
        """

        ShareTensor.sanity_check(y)
        new_self = self.mul(-1)
        new_share = new_self.apply_function(y, "add")
        return new_share

    def mul(
        self, y: Union[int, float, torch.Tensor, np.ndarray, "ShareTensor"]
    ) -> "ShareTensor":
        """Apply the "mul" operation between "self" and "y".

        Args:
            y (Union[int, float, torch.Tensor, np.ndarray, "ShareTensor"]): self * y

        Returns:
            ShareTensor. Result of the operation.
        """

        if isinstance(y, ShareTensor):
            raise ValueError("Private mul not supported yet")

        ShareTensor.sanity_check(y)
        new_share = self.apply_function(y, "mul")
        return new_share

    def matmul(
        self, y: Union[int, float, torch.Tensor, np.ndarray, "ShareTensor"]
    ) -> "ShareTensor":
        """Apply the "matmul" operation between "self" and "y".

        Args:
            y (Union[int, float, torch.Tensor, np.ndarray, "ShareTensor"]): self @ y.

        Returns:
            ShareTensor: Result of the operation.
        """
        if isinstance(y, ShareTensor):
            raise ValueError("Private matmul not supported yet")

        ShareTensor.sanity_check(y)
        new_share = self.apply_function(y, "matmul")
        return new_share

    def rmatmul(self, y: torch.Tensor) -> "ShareTensor":
        """Apply the "rmatmul" operation between "y" and "self".

        Args:
            y (torch.Tensor): y @ self

        Returns:
            ShareTensor. Result of the operation.
        """
        if isinstance(y, ShareTensor):
            raise ValueError("Private matmul not supported yet")

        ShareTensor.sanity_check(y)
        new_share = y.apply_function(self, "matmul")
        return new_share

    def __eq__(self, other: Any) -> bool:
        """Equal operator.

        Check if "self" is equal with another object given a set of
            attributes to compare.

        Args:
            other (Any): Value to compare.

        Returns:
            bool: True if equal False if not.

        """
        # relative
        from .... import Tensor

        if (
            isinstance(self.child, Tensor)
            and isinstance(other.child, Tensor)
            and (self.child != other.child).child.any()  # type: ignore
        ):
            return False

        if (
            isinstance(self.child, np.ndarray)
            and isinstance(other.child, np.ndarray)
            and (self.child != other.child).any()
        ):
            return False

        if self.rank != other.rank:
            return False

        if self.ring_size != other.ring_size:
            return False

        if self.nr_parties != other.nr_parties:
            return False

        return True

    # TRASK: commenting out because ShareTEnsor doesn't appear to have .session_uuid or .config
    # def div(
    #     self, y: Union[int, float, torch.Tensor, np.ndarray, "ShareTensor"]
    # ) -> "ShareTensor":
    #     """Apply the "div" operation between "self" and "y".
    #
    #     Args:
    #         y (Union[int, float, torch.Tensor, np.ndarray, "ShareTensor"]): Denominator.
    #
    #     Returns:
    #         ShareTensor: Result of the operation.
    #
    #     Raises:
    #         ValueError: If y is not an integer or LongTensor.
    #     """
    #     if not isinstance(y, (int, torch.LongTensor)):
    #         raise ValueError("Div works (for the moment) only with integers!")
    #
    #     res = ShareTensor(session_uuid=self.session_uuid, config=self.config)
    #     # res = self.apply_function(y, "floordiv")
    #     res.tensor = self.tensor // y
    #     return res

    @staticmethod
    def hook_method(__self: ShareTensor, method_name: str) -> Callable[..., Any]:
        """Hook a framework method.

        Args:
            method_name (str): method to hook

        Returns:
            A hooked method
        """

        def method_all_shares(
            _self: ShareTensor, *args: List[Any], **kwargs: Dict[Any, Any]
        ) -> Any:

            share = _self.child
            if method_name != "resize":
                method = getattr(share, method_name)
            else:
                # Should be modified to remove copy
                # https://stackoverflow.com/questions/23253144/numpy-the-array-doesnt-have-its-own-data
                share = share.copy()
                method = getattr(share, method_name)

            if method_name not in INPLACE_OPS:
                new_share = method(*args, **kwargs)
            else:
                method(*args, **kwargs)
                new_share = share

            res = ShareTensor(
                rank=_self.rank,
                nr_parties=_self.nr_parties,
                ring_size=_self.ring_size,
                value=new_share,
            )

            return res

        return functools.partial(method_all_shares, __self)

    def __getattribute__(self, attr_name: str) -> Any:

        if attr_name in METHODS_FORWARD_ALL_SHARES or attr_name in INPLACE_OPS:
            return ShareTensor.hook_method(self, attr_name)

        return object.__getattribute__(self, attr_name)

    def _object2proto(self) -> ShareTensor_PB:
        if isinstance(self.child, np.ndarray):
            return ShareTensor_PB(
                array=serialize(self.child), rank=self.rank, nr_parties=self.nr_parties
            )
        elif isinstance(self.child, torch.Tensor):
            return ShareTensor_PB(
                array=serialize(np.array(self.child)),
                rank=self.rank,
                nr_parties=self.nr_parties,
            )
        else:
            return ShareTensor_PB(
                tensor=serialize(self.child), rank=self.rank, nr_parties=self.nr_parties
            )

    @staticmethod
    def _proto2object(proto: ShareTensor_PB) -> "ShareTensor":
        if proto.HasField("tensor"):
            res = ShareTensor(
                rank=proto.rank,
                nr_parties=proto.nr_parties,
                value=deserialize(proto.tensor),
            )
        else:
            res = ShareTensor(
                rank=proto.rank,
                nr_parties=proto.nr_parties,
                value=deserialize(proto.array),
            )

        return res

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return ShareTensor_PB

    __add__ = add
    __radd__ = add
    __sub__ = sub
    __rsub__ = rsub
    __mul__ = mul
    __rmul__ = mul
    __matmul__ = matmul
    __rmatmul__ = rmatmul
