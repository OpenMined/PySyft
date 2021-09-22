# future
from __future__ import annotations

# stdlib
import functools
from functools import lru_cache
import itertools
import operator
import secrets
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import cast

# third party
import numpy as np
import numpy.typing as npt
import torch

# relative
from ..passthrough import PassthroughTensor  # type: ignore
from ..passthrough import SupportedChainType  # type: ignore
from ..util import implements  # type: ignore
from .share_tensor import ShareTensor
from .utils import ispointer

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
    "sum",
    "__pos__",
}
INPLACE_OPS = {
    "resize",
}


class MPCTensor(PassthroughTensor):
    def __init__(
        self,
        parties: List[Any],
        secret: Optional[Any] = None,
        shares: Optional[List[ShareTensor]] = None,
        shape: Optional[Tuple[int, ...]] = None,
        seed_shares: Optional[int] = None,
    ) -> None:

        if secret is None and shares is None:
            raise ValueError("Secret or shares should be populated!")

        if seed_shares is None:
            # Allow the user to specify if they want to use a specific seed when generating the shares
            # ^This is unsecure and should be used with cautioness
            seed_shares = secrets.randbits(32)

        self.seed_shares = seed_shares

        # TODO: We can get this from the the secret if the secret is local
        # TODO: https://app.clubhouse.io/openmined/story/1128/tech-debt-for-adp-smpc-demo?stories_sort_by\
        #  =priority&stories_group_by=WORKFLOW_STATE
        if shape is None:
            raise ValueError("Shape of the secret should be known")

        if secret is not None:
            if parties is None:
                raise ValueError(
                    "Parties should not be None if secret is not already secret shared"
                )
            shares = MPCTensor._get_shares_from_secret(
                secret=secret,
                parties=parties,
                shape=shape,
                seed_shares=seed_shares,
            )

        if shares is None:
            raise ValueError("Shares should not be None at this step")

        res = MPCTensor._mpc_from_shares(shares, parties)

        self.parties = parties

        self.mpc_shape = shape

        # we need to make sure that when we zip up clients from
        # multiple MPC tensors that they are in corresponding order
        # so we always sort all of them by the id of the domain
        # TODO: store children as lists of dictionaries because eventually
        # it's likely that we have multiple shares from the same client
        # (For example, if you wanted a domain to have 90% share ownership
        # you'd need to produce 10 shares and give 9 of them to the same domain)
        # TODO captured: https://app.clubhouse.io/openmined/story/1128/tech-debt-for-adp-smpc-\
        #  demo?stories_sort_by=priority&stories_group_by=WORKFLOW_STATE

        res.sort(key=lambda share: share.client.name + share.client.id.no_dash)

        super().__init__(res)

    def publish(self, sigma: float) -> MPCTensor:

        new_shares = list()
        for share in self.child:
            new_share = share.publish(sigma=sigma)
            new_shares.append(new_share)

        return MPCTensor(
            parties=self.parties,
            shares=new_shares,
            shape=self.mpc_shape,
            seed_shares=self.seed_shares,
        )

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.mpc_shape

    @staticmethod
    def _mpc_from_shares(
        shares: List[ShareTensor],
        parties: Optional[List[Any]] = None,
    ) -> List[ShareTensor]:
        if not isinstance(shares, list):
            raise ValueError("_mpc_from_shares expected a list of shares")

        if ispointer(shares[0]):
            # Remote shares
            return shares
        elif parties is None:
            raise ValueError(
                "Parties should not be None if shares are not already sent to parties"
            )
        else:
            return MPCTensor._mpc_from_local_shares(shares, parties)

    @staticmethod
    def _mpc_from_local_shares(
        shares: List[ShareTensor], parties: List[Any]
    ) -> List[ShareTensor]:
        # TODO: ShareTensor needs to have serde serializer/deserializer
        shares_ptr = [share.send(party) for share, party in zip(shares, parties)]
        return shares_ptr

    @staticmethod
    def _get_shares_from_secret(
        secret: Any, parties: List[Any], shape: Tuple[int, ...], seed_shares: int
    ) -> List[ShareTensor]:
        if ispointer(secret):
            if shape is None:
                raise ValueError("Shape must be specified when the secret is remote")
            return MPCTensor._get_shares_from_remote_secret(
                secret=secret, shape=shape, parties=parties, seed_shares=seed_shares
            )

        return MPCTensor._get_shares_from_local_secret(
            secret=secret, seed_shares=seed_shares, shape=shape, nr_parties=len(parties)
        )

    @staticmethod
    def _get_shares_from_remote_secret(
        secret: Any, shape: Tuple[int, ...], parties: List[Any], seed_shares: int
    ) -> List[ShareTensor]:
        shares = []
        nr_parties = len(parties)
        for i, party in enumerate(parties):
            if secret is not None and party == secret.client:
                value = secret
            else:
                value = None

            # relative
            from ..autodp.single_entity_phi import (
                TensorWrappedSingleEntityPhiTensorPointer,
            )

            if isinstance(secret, TensorWrappedSingleEntityPhiTensorPointer):

                share_wrapper = secret.to_local_object_without_private_data_child()
                share_wrapper_pointer = share_wrapper.send(party)

                remote_share = party.syft.core.tensor.smpc.share_tensor.ShareTensor.generate_przs_on_dp_tensor(
                    rank=i,
                    nr_parties=nr_parties,
                    value=value,
                    shape=shape,
                    seed_shares=seed_shares,
                    share_wrapper=share_wrapper_pointer,
                )

            else:
                remote_share = (
                    party.syft.core.tensor.smpc.share_tensor.ShareTensor.generate_przs(
                        rank=i,
                        nr_parties=nr_parties,
                        value=value,
                        shape=shape,
                        seed_shares=seed_shares,
                    )
                )

            shares.append(remote_share)

        return shares

    @staticmethod
    def _get_shares_from_local_secret(
        secret: Any, shape: Tuple[int, ...], nr_parties: int, seed_shares: int
    ) -> List[ShareTensor]:
        shares = []
        for i in range(nr_parties):
            if i == nr_parties - 1:
                value = secret
            else:
                value = None

            local_share = ShareTensor.generate_przs(
                rank=i,
                nr_parties=nr_parties,
                value=value,
                shape=shape,
                seed_shares=seed_shares,
            )

            shares.append(local_share)

        return shares

    def request(
        self,
        reason: str = "",
        block: bool = False,
        timeout_secs: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        for child in self.child:
            child.request(
                reason=reason, block=block, timeout_secs=timeout_secs, verbose=verbose
            )

    def reconstruct(self) -> np.ndarray:
        # TODO: It might be that the resulted shares (if we run any computation) might
        # not be available at this point. We need to have this fail well with a nice
        # description as to which node wasn't able to be reconstructued.
        # Captured: https://app.clubhouse.io/openmined/story/1128/tech-debt-for-adp-smpc-demo?\
        # stories_sort_by=priority&stories_group_by=WORKFLOW_STATE

        # for now we need to convert the values coming back to int32
        # sometimes they are floats coming from DP
        def convert_child_numpy_type(tensor: Any, np_type: type) -> Any:
            if isinstance(tensor, np.ndarray):
                return np.array(tensor, np_type)
            if hasattr(tensor, "child"):
                tensor.child = convert_child_numpy_type(
                    tensor=tensor.child, np_type=np_type
                )
            return tensor

        local_shares = []
        for share in self.child:
            res = share.get()
            res = convert_child_numpy_type(res, np.int32)
            local_shares.append(res)

        is_share_tensor = isinstance(local_shares[0], ShareTensor)

        if is_share_tensor:
            local_shares = [share.child for share in local_shares]

        result = local_shares[0]
        for share in local_shares[1:]:
            result = result + share

        if hasattr(result, "child") and isinstance(result.child, ShareTensor):
            return result.child.child

        return result

    get = reconstruct

    @staticmethod
    @lru_cache(maxsize=128)
    def __get_shape(
        op_str: str,
        x_shape: Tuple[int],
        y_shape: Tuple[int],
    ) -> Tuple[int]:
        """Get the shape of apply an operation on two values

        Args:
            op_str (str): the operation to be applied
            x_shape (Tuple[int]): the shape of op1
            y_shape (Tuple[int]): the shape of op2

        Returns:
            The shape of the result
        """
        op = getattr(operator, op_str)
        res = op(np.empty(x_shape), np.empty(y_shape)).shape
        cast(Tuple[int], res)
        return tuple(res)  # type: ignore

    @staticmethod
    def hook_method(__self: MPCTensor, method_name: str) -> Callable[..., Any]:
        """Hook a framework method.

        Args:
            method_name (str): method to hook

        Returns:
            A hooked method
        """

        def method_all_shares(
            _self: MPCTensor, *args: List[Any], **kwargs: Dict[Any, Any]
        ) -> Any:

            shares = []

            for share in _self.child:
                method = getattr(share, method_name)
                new_share = method(*args, **kwargs)
                shares.append(new_share)

                dummy_res = np.empty(_self.mpc_shape)
                if method_name not in INPLACE_OPS:
                    dummy_res = getattr(np.empty(_self.mpc_shape), method_name)(
                        *args, **kwargs
                    )
                else:
                    getattr(np.empty(_self.mpc_shape), method_name)(*args, **kwargs)

                new_shape = dummy_res.shape
            res = MPCTensor(parties=_self.parties, shares=shares, shape=new_shape)
            return res

        return functools.partial(method_all_shares, __self)

    def __getattribute__(self, attr_name: str) -> Any:

        if attr_name in METHODS_FORWARD_ALL_SHARES:
            return MPCTensor.hook_method(self, attr_name)

        return object.__getattribute__(self, attr_name)

    @staticmethod
    def reshare(mpc_tensor: MPCTensor, parties: Iterable[Any]) -> MPCTensor:
        """Reshare a given secret to a superset of parties.

        Args:
            mpc_tensor(MPCTensor): input MPCTensor to reshare.
            parties(List[Any]): Input parties List.

        Returns:
            res_mpc(MPCTensor): Reshared MPCTensor.

        Raises:
            ValueError: If the input MPCTensor and input parties are same.
        """
        mpc_parties = set(mpc_tensor.parties)
        parties = set(parties)
        shape = mpc_tensor.shape
        seed_shares = mpc_tensor.seed_shares
        client_map = {share.client: share for share in mpc_tensor.child}
        nr_parties = len(parties)
        if mpc_parties == parties:
            raise ValueError(
                "Input parties for resharing are same as the input parties."
            )

        shares = [client_map.get(party) for party in parties]
        for i, party in enumerate(parties):
            shares[
                i
            ] = party.syft.core.tensor.smpc.share_tensor.ShareTensor.generate_przs(
                rank=i,
                nr_parties=nr_parties,
                value=shares[i],
                shape=shape,
                seed_shares=seed_shares,
            )

        res_mpc = MPCTensor(shares=shares, shape=shape, parties=parties)  # type: ignore

        return res_mpc

    @staticmethod
    def sanity_checks(mpc_tensor: MPCTensor, other: Any) -> Tuple[MPCTensor, Any]:
        """Performs sanity checks to share data to whole superset of parites involved.

        Args:
            mpc_tensor(MPCTensor): input MPCTensor to perform sanity check on.
            other (Any): input operand.

        Returns:
            Tuple[MPCTensor,Any]: Rehared Tensor values.
        """
        if ispointer(other):
            parties = mpc_tensor.parties
            client = other.client
            public_shape = other.public_shape
            if public_shape is None:
                # TODO: Should be modified after Trask's Synthetic data PR.
                raise ValueError("The input tensor pointer should have public shape.")
            if client not in parties:
                parties.append(client)
                mpc_tensor = MPCTensor.reshare(mpc_tensor, parties)

            other = MPCTensor(secret=other, parties=parties, shape=public_shape)

        elif isinstance(other, MPCTensor):
            p1 = set(mpc_tensor.parties)  # parties in first MPCTensor
            p2 = set(other.parties)  # parties in second MPCTensor.
            if p1 != p2:
                parties_union = p1.union(p2)
                mpc_tensor = MPCTensor.reshare(mpc_tensor, parties_union)
                other = MPCTensor.reshare(other, parties_union)

        return mpc_tensor, other

    def __apply_private_op(self, other: MPCTensor, op_str: str) -> List[ShareTensor]:
        op = getattr(operator, op_str)
        if isinstance(other, MPCTensor):
            res_shares = [op(a, b) for a, b in zip(self.child, other.child)]
        else:
            raise ValueError("Add works only for the MPCTensor at the moment!")
        return res_shares

    def __apply_public_op(self, y: Any, op_str: str) -> List[ShareTensor]:
        op = getattr(operator, op_str)
        if op_str in {"mul", "matmul"}:
            res_shares = [op(share, y) for share in self.child]
        elif op_str in {"add", "sub"}:
            res_shares = self.child
            res_shares[0] = op(res_shares[0], y)
        else:
            raise ValueError(f"{op_str} not supported")

        return res_shares

    def __apply_op(
        self,
        y: Union[int, float, torch.Tensor, np.ndarray, MPCTensor],
        op_str: str,
    ) -> MPCTensor:
        """Apply an operation on "self" which is a MPCTensor "y".

         This function checks if "y" is private or public value.

        Args:
            y (Union[int, float, torch.Tensor, np.ndarray, MPCTensor]: tensor to apply the operation.
            op_str (str): the operation.

        Returns:
            MPCTensor. the operation "op_str" applied on "self" and "y"
        """
        x, y = MPCTensor.sanity_checks(self, y)

        if isinstance(y, MPCTensor):
            result = x.__apply_private_op(y, op_str)
        else:
            result = x.__apply_public_op(y, op_str)

        if isinstance(y, (float, int)):
            y_shape: Tuple[int, ...] = (1,)
        elif isinstance(y, MPCTensor):
            y_shape = y.mpc_shape
        else:
            y_shape = y.shape

        shape = MPCTensor.__get_shape(op_str, self.mpc_shape, y_shape)

        result = MPCTensor(shares=result, shape=shape, parties=x.parties)

        return result

    def add(
        self, y: Union[int, float, np.ndarray, torch.tensor, MPCTensor]
    ) -> MPCTensor:
        """Apply the "add" operation between "self" and "y".

        Args:
            y (Union[MPCTensor, torch.Tensor, float, int]): self + y.

        Returns:
            MPCTensor. Result of the operation.
        """
        res = self.__apply_op(y, "add")
        return res

    def sub(self, y: MPCTensor) -> MPCTensor:
        res = self.__apply_op(y, "sub")
        return res

    def rsub(self, y: MPCTensor) -> MPCTensor:
        new_self = self * (-1)
        res = new_self.__apply_op(y, "add")
        return res

    def mul(
        self, y: Union[int, float, np.ndarray, torch.tensor, MPCTensor]
    ) -> MPCTensor:
        if isinstance(y, MPCTensor):
            raise ValueError("Private multiplication not yet implemented!")
        else:
            res_shares = [
                operator.mul(a, b) for a, b in zip(self.child, itertools.repeat(y))
            ]

        y_shape = getattr(y, "shape", (1,))
        new_shape = MPCTensor.__get_shape("mul", self.mpc_shape, y_shape)
        res = MPCTensor(parties=self.parties, shares=res_shares, shape=new_shape)

        return res

    def matmul(
        self, y: Union[int, float, np.ndarray, torch.tensor, "MPCTensor"]
    ) -> MPCTensor:
        """Apply the "matmul" operation between "self" and "y"

        Args:
            y (Union[int, float, np.ndarray, torch.tensor, "MPCTensor"]): self @ y

        Returns:
            MPCTensor: Result of the opeartion.
        """
        if isinstance(y, ShareTensor):
            raise ValueError("Private matmul not supported yet")

        res = self.__apply_op(y, "matmul")

        return res

    def __str__(self) -> str:
        res = "MPCTensor"
        for share in self.child:
            res = f"{res}\n\t{share}"

        return res

    def __repr__(self) -> str:
        out = "MPCTensor"
        out += ".shape=" + str(self.shape) + "\n"
        for i, child in enumerate(self.child):
            out += f"\t .child[{i}] = " + child.__repr__() + "\n"
        out = out[:-1] + ""

        return out

    def put(
        self,
        indices: npt.ArrayLike,
        values: npt.ArrayLike,
        mode: Optional[str] = "raise",
    ) -> MPCTensor:
        """Performs Numpy put operation on the underlying ShareTensors.

        Args:
            indices (npt.ArrayLike): Target indices, interpreted as integers.
            values (npt.ArrayLike): Values to place at target indices.
            mode (Optional[str]): Specifies how out-of-bounds indices will behave.

        Returns:
            res (MPCTensor): Result of the operation.
        """
        shares = []
        shares.append(self.child[0].put(indices, values, mode))
        # since the value is public we assign directly to prevent overhead of random share creation.
        zero = np.zeros_like(values)
        for share in self.child[1::]:
            shares.append(share.put(indices, zero.copy(), mode))

        res = MPCTensor(shares=shares, parties=self.parties, shape=self.shape)
        return res

    __add__ = add
    __radd__ = add
    __sub__ = sub
    __rsub__ = rsub
    __mul__ = mul
    __rmul__ = mul
    __matmul__ = matmul


@implements(MPCTensor, np.add)
def add(x: np.ndarray, y: MPCTensor) -> SupportedChainType:
    return y.add(x)


@implements(MPCTensor, np.subtract)
def sub(x: np.ndarray, y: MPCTensor) -> SupportedChainType:
    return y.rsub(x)


@implements(MPCTensor, np.multiply)
def mul(x: np.ndarray, y: MPCTensor) -> SupportedChainType:
    return y.mul(x)
