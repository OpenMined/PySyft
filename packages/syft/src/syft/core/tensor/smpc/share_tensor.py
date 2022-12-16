# future
from __future__ import annotations

# stdlib
from copy import deepcopy
import functools
from functools import lru_cache
import operator
import secrets
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from typing import Tuple
from typing import Union
from uuid import UUID

# third party
import gevent
import numpy as np
import torch

# relative
from . import utils
from .... import logger
from ....lib.numpy.array import capnp_deserialize
from ....lib.numpy.array import capnp_serialize
from ....proto.core.tensor.share_tensor_pb2 import ShareTensor as ShareTensor_PB
from ...common import UID
from ...common.serde.capnp import CapnpModule
from ...common.serde.capnp import chunk_bytes
from ...common.serde.capnp import combine_bytes
from ...common.serde.capnp import get_capnp_schema
from ...common.serde.capnp import serde_magic_header
from ...common.serde.deserialize import _deserialize as deserialize
from ...common.serde.serializable import serializable
from ...common.serde.serialize import _serialize as serialize
from ...node.common.action.greenlets_switch import przs_retrieve_object
from ...smpc.store.crypto_store import CryptoStore
from ..config import DEFAULT_RING_SIZE
from ..fixed_precision_tensor import FixedPrecisionTensor
from ..passthrough import PassthroughTensor  # type: ignore

if TYPE_CHECKING:
    # relative
    from ..tensor import Tensor


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
    "__neg__",
    "take",
    "choose",
    "cumsum",
    "trace",
}
INPLACE_OPS = {"resize", "put"}
RING_SIZE_TO_OP = {
    2: {
        "add": operator.xor,
        "sub": operator.xor,
        "mul": operator.and_,
        "lt": operator.lt,
        "gt": operator.gt,
        "ge": operator.ge,
        "le": operator.le,
        "eq": operator.eq,
        "ne": operator.ne,
    },
    2
    ** 32: {
        "add": operator.add,
        "sub": operator.sub,
        "mul": operator.mul,
        "matmul": operator.matmul,
        "lt": operator.lt,
        "gt": operator.gt,
        "ge": operator.ge,
        "le": operator.le,
        "eq": operator.eq,
        "ne": operator.ne,
    },
    2
    ** 64: {
        "add": operator.add,
        "sub": operator.sub,
        "mul": operator.mul,
        "matmul": operator.matmul,
        "lt": operator.lt,
        "gt": operator.gt,
        "ge": operator.ge,
        "le": operator.le,
        "eq": operator.eq,
        "ne": operator.ne,
    },
}

CACHE_CLIENTS: Dict[str, Any] = {}
GEVENT_LOGIN: bool = False  # prevent race conditions due to gevent monkey patching


def populate_store(*args: Any, **kwargs: Any) -> None:
    ShareTensor.crypto_store.populate_store(*args, **kwargs)  # type: ignore
    return None


@serializable(capnp_bytes=True)
class ShareTensor(PassthroughTensor):
    crypto_store = CryptoStore()

    __slots__ = (
        "rank",
        "ring_size",
        "clients",  # clients connections
        "min_value",
        "max_value",
        "parties_info",
        "nr_parties",
    )

    def __init__(
        self,
        rank: int,
        parties_info: List[Tuple],
        ring_size: int,
        clients: Optional[List[Any]] = None,
        value: Optional[Any] = None,
        init_clients: bool = False,
    ) -> None:
        # TODO: Ring size needs to be changed to 2^64 (or other specific sizes)
        self.rank = rank
        self.ring_size = ring_size
        self.nr_parties = len(parties_info)
        self.parties_info = parties_info
        self.clients = []
        if clients is not None:
            self.clients = clients
        elif init_clients:  # type: ignore
            self.clients = ShareTensor.login_clients(parties_info=parties_info)

        self.min_value, self.max_value = ShareTensor.compute_min_max_from_ring(
            self.ring_size
        )

        super().__init__(value)

    @staticmethod
    def login_clients(parties_info: List[Tuple]) -> Any:
        # relative
        from ....grid.client.client import login
        from ....grid.client.proxy_client import ProxyClient

        global GEVENT_LOGIN
        while GEVENT_LOGIN:
            gevent.sleep(0)
        GEVENT_LOGIN = True

        clients = []
        for party_info in parties_info:
            # if its localhost change it to a host that resolves outside the container
            external_host_info = party_info[0].as_container_host()
            client = CACHE_CLIENTS.get(str(external_host_info), None)

            if client is None:
                # # default cache to true, here to prevent multiple logins
                # # due to gevent monkey patching, context switch is done during
                # # during socket connection initialization.
                # CACHE_CLIENTS[str(external_host_info)] = True
                # TODO: refactor to use a guest account
                client = login(  # nosec
                    url=external_host_info,
                    email="info@openmined.org",
                    password="changethis",
                    port=external_host_info.port,
                    verbose=False,
                )
                CACHE_CLIENTS[str(external_host_info)] = client

            if client.id != party_info[1]:
                client = ProxyClient.create(client, party_info[1], party_info[2])
            clients.append(client)
        GEVENT_LOGIN = False
        return clients

    @staticmethod
    def get_id_rank_mapping(clients: List[Any]) -> Dict:
        client_uids = [client.id.no_dash for client in clients]
        client_uids.sort()
        ID_RANK_MAP = {}
        for idx, client_uid in enumerate(client_uids):
            ID_RANK_MAP[idx] = client_uid
            ID_RANK_MAP[client_uid] = idx
        return ID_RANK_MAP

    def __getitem__(self, item: Union[str, int, slice]) -> ShareTensor:
        return ShareTensor(
            rank=self.rank,
            parties_info=self.parties_info,
            ring_size=self.ring_size,
            value=self.child[item],
            clients=self.clients,
        )

    def copy_tensor(self) -> ShareTensor:
        return ShareTensor(
            value=self.child,
            rank=self.rank,
            parties_info=self.parties_info,
            ring_size=self.ring_size,
            clients=self.clients,
        )

    @staticmethod
    @lru_cache(32)
    def compute_min_max_from_ring(
        ring_size: int = DEFAULT_RING_SIZE,
    ) -> Tuple[int, int]:
        if ring_size == 2:
            min_value, max_value = 0, 1
        else:
            min_value = (-ring_size) // 2
            max_value = (ring_size) // 2 - 1
        return min_value, max_value

    @staticmethod
    @lru_cache(maxsize=None)
    def get_op(ring_size: int, op_str: str) -> Callable[..., Any]:
        """Returns method attribute based on ring_size and op_str.
        Args:
            ring_size (int): Ring size
            op_str (str): Operation string.
        Returns:
            op (Callable[...,Any]): The operation method for the op_str.
        Raises:
            ValueError : If invalid ring size or op_str is given as input.
        """
        ops = RING_SIZE_TO_OP.get(ring_size, None)

        if ops is None:
            raise ValueError(f"Do not have operations for ring size {ring_size}")

        op = ops.get(op_str, None)
        if op is None:
            raise ValueError(
                f"Operator {op_str} does not exist for ring size {ring_size}"
            )

        return op

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
        value: Any,
        shape: Tuple[int, ...],
        parties_info: List[Tuple],
        ring_size: Union[int, str] = DEFAULT_RING_SIZE,
        init_clients: bool = True,
    ) -> Tensor:
        # relative
        from ...node.common.action.beaver_action import BeaverAction
        from ...tensor.smpc import context
        from ..tensor import Tensor

        seed_id_locations = context.SMPC_CONTEXT.get("seed_id_locations", None)
        node = context.SMPC_CONTEXT.get("node", None)
        if seed_id_locations is None:
            raise ValueError(
                f"seed_id_locations:{seed_id_locations}  input should be a valid integer for PRZS generation"
            )
        if node is None:
            raise ValueError(
                f"Node:{seed_id_locations} input should be a valid Node Object"
            )
        ring_size = int(ring_size)
        nr_parties = len(parties_info)

        clients = ShareTensor.login_clients(parties_info=parties_info)

        id_rank_map = ShareTensor.get_id_rank_mapping(clients)

        rank = id_rank_map[node.id.no_dash]  # rank of the current party
        przs_client_id = id_rank_map[
            (rank + 1) % nr_parties
        ]  # get the client id of the next party
        self_generator_seed = secrets.randbits(64)
        generator = np.random.default_rng(seed_id_locations)
        przs_location = UID(UUID(bytes=generator.bytes(16)))

        for client in clients:
            if client.id.no_dash == przs_client_id:
                beaver_action = BeaverAction(
                    values=[str(self_generator_seed)],
                    locations=[przs_location],
                    address=client.address,
                )
                client.send_immediate_msg_without_reply(msg=beaver_action)
        other_generator_seed = przs_retrieve_object(node, przs_location).data

        if len(other_generator_seed) > 1:
            raise ValueError(
                f"PRZS should receive only one seed from peer client,got: {len(other_generator_seed)}"
            )
        other_generator_seed = int(other_generator_seed[0])

        self_generator = np.random.default_rng(self_generator_seed)
        other_generator = np.random.default_rng(other_generator_seed)

        # Try:
        # 1. First get numpy type if secret is numpy and obtain ring size from there
        # 2. If not get the type from the ring size

        numpy_type = None
        ring_size_final = None

        ring_size_from_type = utils.TYPE_TO_RING_SIZE.get(
            getattr(value, "dtype", None), None
        )
        if ring_size_from_type is None:
            logger.warning(f"Could not get ring size from {value}")
        else:
            ring_size_final = ring_size_from_type
            numpy_type = value.dtype

        if numpy_type is None:
            numpy_type = utils.RING_SIZE_TO_TYPE.get(ring_size, None)
            ring_size_final = ring_size

        if numpy_type is None:
            raise ValueError(f"Ring size {ring_size} not known how to be treated")

        if value is None:
            value = Tensor(np.zeros(shape, dtype=numpy_type))

        if isinstance(value.child, (ShareTensor, FixedPrecisionTensor)):
            value = value.child

        share = ShareTensor(
            value=value.child,
            rank=rank,
            parties_info=parties_info,
            init_clients=init_clients,
            ring_size=ring_size_final,  # type: ignore
        )

        self_generator_share = self_generator.integers(
            low=share.min_value,
            high=share.max_value,
            size=shape,
            endpoint=True,
            dtype=numpy_type,
        )
        other_generator_share = other_generator.integers(
            low=share.min_value,
            high=share.max_value,
            size=shape,
            endpoint=True,
            dtype=numpy_type,
        )
        op = ShareTensor.get_op(ring_size_final, "sub")
        # przs_share = op(shares[rank], shares[(rank + 1) % nr_parties])
        przs_share = op(self_generator_share, other_generator_share)
        share.child = op(share.child, przs_share)
        res = Tensor(share)

        return res

    @staticmethod
    def generate_przs_on_dp_tensor(
        value: Optional[Any],
        shape: Tuple[int],
        parties_info: List[Tuple],
        share_wrapper: Any,
        ring_size: Union[int, str] = DEFAULT_RING_SIZE,
    ) -> PassthroughTensor:
        # relative
        from ..autodp.gamma_tensor import GammaTensor

        ring_size = int(ring_size)
        if value and hasattr(value, "child"):
            if isinstance(value.child, GammaTensor):
                # We do this, since GammaTensor is a FrozenInstance, which prevents us from modifying child values.
                gt: GammaTensor = value.child
                new_gamma = GammaTensor(
                    child=FixedPrecisionTensor(value.child.child),
                    data_subjects=gt.data_subjects,
                    min_vals=gt.min_vals,
                    max_vals=gt.max_vals,
                    func_str=gt.func_str,
                    sources=gt.sources,
                )
                value.child = new_gamma

            else:
                value.child.child = FixedPrecisionTensor(value.child.child)  # type: ignore

        if value is not None:
            share = ShareTensor.generate_przs(
                value=value.child,
                shape=shape,
                parties_info=parties_info,
                ring_size=ring_size,
            )
        else:
            share = ShareTensor.generate_przs(
                value=value,
                shape=shape,
                parties_info=parties_info,
                ring_size=ring_size,
            )
        # relative
        from ..autodp.gamma_tensor import GammaTensor
        from ..autodp.phi_tensor import PhiTensor

        if isinstance(share_wrapper.child, (PhiTensor, GammaTensor)):
            share_wrapper.child.child.child = share.child
        else:
            share_wrapper.child.child = share.child

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

        if isinstance(share, np.ndarray) and (
            not np.issubdtype(share.dtype, np.integer)
            and share.dtype != np.dtype("bool")
        ):
            raise ValueError(
                f"NPArray should have type int or bool, but found {share.dtype}"
            )

        if isinstance(share, torch.Tensor) and torch.is_floating_point(share):
            raise ValueError("Torch tensor should have type int, but found float")

    def apply_function(
        self,
        y: Union[int, float, torch.Tensor, np.ndarray, "ShareTensor"],
        op_str: str,
    ) -> "ShareTensor":
        """Apply a given operation.

        Args:
            y (Union[int, float, torch.Tensor, np.ndarray, "ShareTensor"]): tensor to apply the operator.
            op_str (str): Operator.

        Returns:
            ShareTensor: Result of the operation.
        """
        ShareTensor.sanity_check(y)

        op = ShareTensor.get_op(self.ring_size, op_str)
        numpy_type = utils.RING_SIZE_TO_TYPE.get(self.ring_size, None)
        if numpy_type is None:
            raise ValueError(f"Do not know numpy type for ring size {self.ring_size}")

        if isinstance(y, ShareTensor):
            utils.get_ring_size(self.ring_size, y.ring_size)  # sanity check
            value = op(self.child, y.child)
        else:
            if op_str in {"add", "sub"}:
                # TODO: Converting y to numpy because doing "numpy op torch tensor" raises exception
                value = (
                    op(self.child, np.array(y, numpy_type))
                    if self.rank == 0
                    else deepcopy(self.child)
                )
            elif op_str in ["mul", "matmul", "lt"]:
                value = op(self.child, np.array(y, numpy_type))
            else:
                raise ValueError(f"{op_str} not supported")

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
        # relative
        from ...node.common.action.smpc_action_functions import private_mul

        if isinstance(y, ShareTensor):
            new_share = private_mul(self, y, "mul")
        else:
            new_share = self.apply_function(y, "mul")

        return new_share

    def truediv(self, y: Union[int, float, np.ndarray, "ShareTensor"]) -> "ShareTensor":
        """Apply the "division" operation between "self" and "y".

        Args:
            y (Union[int, float, np.ndarray, "ShareTensor"]): self / y

        Returns:
            ShareTensor. Result of the operation.
        """
        # relative
        from ...node.common.action.smpc_action_functions import public_divide

        if not isinstance(y, (int, np.integer)):
            raise ValueError("Current Division only works for integers")
        else:
            if self.ring_size != 2:
                new_share = public_divide(self, y)
            else:
                new_share = self.copy_tensor()

        return new_share

    def bit_decomposition(self, ring_size: Union[int, str], bitwise: bool) -> None:
        """Apply the "decomposition" operation on self

        Args:
            ring_size (int): Ring size to decompose the shares
            bitwise (bool): Flag for bitwise decomposition.

        Returns:
            ShareTensor. Result of the operation.
        """
        # relative
        from ...node.common.action.smpc_action_functions import _decomposition

        ring_size = int(ring_size)

        _decomposition(self, ring_size, bitwise)

    def matmul(
        self, y: Union[int, float, torch.Tensor, np.ndarray, "ShareTensor"]
    ) -> "ShareTensor":
        """Apply the "matmul" operation between "self" and "y".

        Args:
            y (Union[int, float, torch.Tensor, np.ndarray, "ShareTensor"]): self @ y

        Returns:
            ShareTensor. Result of the operation.
        """
        # relative
        from ...node.common.action.smpc_action_functions import private_mul

        if isinstance(y, ShareTensor):
            new_share = private_mul(self, y, "matmul")
        else:
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

        new_share = ShareTensor.apply_function(self, y, "matmul")
        return new_share

    def lt(self, y: Union[ShareTensor, np.ndarray]) -> "ShareTensor":
        """Apply the "lt" operation between "y" and "self".

        Args:
            y (Union[ShareTensor,np.ndarray]): self < y

        Returns:
            ShareTensor. Result of the operation.
        """
        # raise ValueError(
        #     "It should not reach this point since we generate SMPCAction for this"
        # )
        new_share = ShareTensor.apply_function(self, y, "lt")
        return new_share

    def gt(self, y: Union[ShareTensor, np.ndarray]) -> "ShareTensor":
        """Apply the "gt" operation between "y" and "self".

        Args:
            y (Union[ShareTensor,np.ndarray]): self > y

        Returns:
            ShareTensor. Result of the operation.
        """
        # raise ValueError(
        #     "It should not reach this point since we generate SMPCAction for this"
        # )
        new_share = ShareTensor.apply_function(self, y, "gt")
        return new_share

    def ge(self, y: Union[ShareTensor, np.ndarray]) -> "ShareTensor":
        """Apply the "ge" operation between "y" and "self".

        Args:
            y (Union[ShareTensor,np.ndarray]): self >= y

        Returns:
            ShareTensor. Result of the operation.
        """
        # raise ValueError(
        #     "It should not reach this point since we generate SMPCAction for this"
        # )
        new_share = ShareTensor.apply_function(self, y, "ge")
        return new_share

    def le(self, y: Union[ShareTensor, np.ndarray]) -> "ShareTensor":
        """Apply the "le" operation between "y" and "self".

        Args:
            y (Union[ShareTensor,np.ndarray]): self <= y

        Returns:
            ShareTensor. Result of the operation.
        """
        # raise ValueError(
        #     "It should not reach this point since we generate SMPCAction for this"
        # )
        new_share = ShareTensor.apply_function(self, y, "le")
        return new_share

    def ne(self, y: Union[ShareTensor, np.ndarray]) -> "ShareTensor":
        """Apply the "ne" operation between "y" and "self".

        Args:
            y (Union[ShareTensor,np.ndarray]): self != y

        Returns:
            ShareTensor. Result of the operation.
        """
        # raise ValueError(
        #     "It should not reach this point since we generate SMPCAction for this"
        # )
        new_share = ShareTensor.apply_function(self, y, "ne")
        return new_share

    def eq(self, other: Any) -> bool:
        """Equal operator.
        Check if "self" is equal with another object given a set of
            attributes to compare.
        Args:
            other (Any): Value to compare.
        Returns:
            bool: True if equal False if not.
        """
        # TODO: Rasswanth: Fix later after the comparison operation
        # relative
        # from .... import Tensor

        # if (
        #     isinstance(self.child, Tensor)
        #     and isinstance(other.child, Tensor)
        #     and (self.child != other.child).child.any()  # type: ignore
        # ):
        #     return False

        # if (
        #     isinstance(self.child, np.ndarray)
        #     and isinstance(other.child, np.ndarray)
        #     and (self.child != other.child).any()
        # ):
        #     return False

        # if self.rank != other.rank:
        #     return False

        # if self.ring_size != other.ring_size:
        #     return False

        # if self.nr_parties != other.nr_parties:
        #     return False

        # return True

        # ATTENTION: Why are we getting here now when we never did before?
        if not hasattr(other, "child"):
            return self.child == other

        return self.child == other.child

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

    def bit_extraction(self, pos: int = 0) -> ShareTensor:
        """Extracts the bit at the specified position.

        Args:
            pos (int): position to extract bit.

        Returns:
            ShareTensor : extracted bits at specific position.

        Raises:
            ValueError: If invalid position is provided.
        """
        ring_bits = utils.get_nr_bits(self.ring_size)
        if pos < 0 or pos > ring_bits - 1:
            raise ValueError(
                f"Invalid position for bit_extraction: {pos}, must be in range:[0,{ring_bits-1}]"
            )
        shape = self.shape
        numpy_type = utils.RING_SIZE_TO_TYPE[self.ring_size]
        # logical shift
        bit_mask = np.ones(shape, dtype=numpy_type) << pos
        value = self.child & bit_mask
        value = value.astype(np.bool_)
        share = self.copy_tensor()
        share.child = value
        return share

    def concatenate(self, other: ShareTensor, *args: Any, **kwargs: Any) -> ShareTensor:
        res = self.copy()
        res.child = np.concatenate((self.child, other.child), *args, **kwargs)
        return res

    @staticmethod
    def hook_method(__self: ShareTensor, method_name: str) -> Callable[..., Any]:
        """Hook a framework method.

        Args:
            method_name (str): method to hook

        Returns:
            A hooked method
        """

        def method_all_shares(_self: ShareTensor, *args: Any, **kwargs: Any) -> Any:

            share = _self.child

            method = getattr(share, method_name)
            if method_name not in INPLACE_OPS:
                new_share = method(*args, **kwargs)
            else:
                method(*args, **kwargs)
                new_share = share

            res = _self.copy_tensor()
            res.child = np.array(new_share)

            return res

        return functools.partial(method_all_shares, __self)

    def __getattribute__(self, attr_name: str) -> Any:
        if attr_name in METHODS_FORWARD_ALL_SHARES or attr_name in INPLACE_OPS:
            return ShareTensor.hook_method(self, attr_name)

        return object.__getattribute__(self, attr_name)

    # TODO: Add capnp serialization to ShareTensor
    def _object2proto(self) -> ShareTensor_PB:
        # This works only for unsigned types.
        length_rs = self.ring_size.bit_length()
        rs_bytes = self.ring_size.to_bytes((length_rs + 7) // 8, byteorder="big")

        proto_init_kwargs = {
            "rank": self.rank,
            "parties_info": [serialize(party) for party in self.parties_info],
            "ring_size": rs_bytes,
        }

        proto_init_kwargs["child"] = serialize(self.child, to_bytes=True)

        return ShareTensor_PB(**proto_init_kwargs)

    @staticmethod
    def _proto2object(proto: ShareTensor_PB) -> "ShareTensor":
        init_kwargs = {
            "rank": proto.rank,
            "parties_info": [deserialize(party) for party in proto.parties_info],
            "ring_size": int.from_bytes(proto.ring_size, "big"),
        }

        init_kwargs["value"] = deserialize(proto.child, from_bytes=True)

        # init_kwargs["init_clients"] = True
        res = ShareTensor(**init_kwargs)
        return res

    def _object2bytes(self) -> bytes:
        schema = get_capnp_schema(schema_file="share_tensor.capnp")

        st_struct: CapnpModule = schema.ShareTensor  # type: ignore
        st_msg = st_struct.new_message()
        # this is how we dispatch correct deserialization of bytes
        st_msg.magicHeader = serde_magic_header(type(self))

        # child of Share tensor could either be Python Scalar or np.ndarray
        if isinstance(self.child, np.ndarray) or np.isscalar(self.child):
            chunk_bytes(
                capnp_serialize(np.array(self.child), to_bytes=True), "child", st_msg
            )
            st_msg.isNumpy = True
        else:
            chunk_bytes(serialize(self.child, to_bytes=True), "child", st_msg)  # type: ignore
            st_msg.isNumpy = False

        st_msg.rank = self.rank
        st_msg.partiesInfo = serialize(self.parties_info, to_bytes=True)
        st_msg.ringSize = str(self.ring_size)

        return st_msg.to_bytes_packed()

    @staticmethod
    def _bytes2object(buf: bytes) -> ShareTensor:
        schema = get_capnp_schema(schema_file="share_tensor.capnp")
        st_struct: CapnpModule = schema.ShareTensor  # type: ignore
        # https://stackoverflow.com/questions/48458839/capnproto-maximum-filesize
        MAX_TRAVERSAL_LIMIT = 2**64 - 1

        st_msg = st_struct.from_bytes_packed(
            buf, traversal_limit_in_words=MAX_TRAVERSAL_LIMIT
        )

        if st_msg.isNumpy:
            child = capnp_deserialize(combine_bytes(st_msg.child), from_bytes=True)
        else:
            child = deserialize(combine_bytes(st_msg.child), from_bytes=True)

        return ShareTensor(
            value=child,
            rank=st_msg.rank,
            parties_info=deserialize(st_msg.partiesInfo, from_bytes=True),
            ring_size=int(st_msg.ringSize),
        )

    __add__ = add
    __radd__ = add
    __sub__ = sub
    __rsub__ = rsub
    __mul__ = mul
    __rmul__ = mul
    __matmul__ = matmul
    __rmatmul__ = rmatmul
    __truediv__ = truediv
    __lt__ = lt
    __gt__ = gt
    __ge__ = ge
    __le__ = le
    __eq__ = eq
    __ne__ = ne
