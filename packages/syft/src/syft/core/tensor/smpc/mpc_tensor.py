# future
from __future__ import annotations

# stdlib
import functools
from functools import lru_cache
import itertools
import operator
import secrets
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import cast

# third party
import numpy as np
import torch

# syft absolute
from syft.core.tensor.passthrough import PassthroughTensor
from syft.core.tensor.passthrough import SupportedChainType
from syft.core.tensor.smpc.share_tensor import ShareTensor

# relative
from ..util import implements  # type: ignore
from .utils import ispointer

METHODS_FORWARD_ALL_SHARES = {
    "t",
    "squeeze",
    "unsqueeze",
    "view",
    "sum",
    "clone",
    "flatten",
    "reshape",
    "repeat",
    "narrow",
    "dim",
    "transpose",
}


class MPCTensor(PassthroughTensor):
    def __init__(
        self,
        parties: Optional[List[Any]] = None,
        secret: Optional[Any] = None,
        shares: Optional[List[ShareTensor]] = None,
        shape: Optional[Tuple[int]] = None,
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
    def shape(self) -> Optional[Tuple[int]]:
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
        secret: Any, parties: List[Any], shape: Tuple[int], seed_shares: int
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
        secret: Any, shape: Tuple[int], parties: List[Any], seed_shares: int
    ) -> List[ShareTensor]:
        shares = []
        for i, party in enumerate(parties):
            if secret is not None and party == secret.client:
                value = secret
            else:
                value = None

            # syft absolute
            from syft.core.tensor.autodp.single_entity_phi import (
                TensorWrappedSingleEntityPhiTensorPointer,
            )

            if isinstance(secret, TensorWrappedSingleEntityPhiTensorPointer):

                share_wrapper = secret.to_local_object_without_private_data_child()
                share_wrapper_pointer = share_wrapper.send(party)

                remote_share = party.syft.core.tensor.smpc.share_tensor.ShareTensor.generate_przs_on_dp_tensor(
                    rank=i,
                    nr_parties=len(parties),
                    value=value,
                    shape=shape,
                    seed_shares=seed_shares,
                    share_wrapper=share_wrapper_pointer,
                )

            else:
                remote_share = (
                    party.syft.core.tensor.smpc.share_tensor.ShareTensor.generate_przs(
                        rank=i,
                        nr_parties=len(parties),
                        value=value,
                        shape=shape,
                        seed_shares=seed_shares,
                    )
                )

            shares.append(remote_share)

        return shares

    @staticmethod
    def _get_shares_from_local_secret(
        secret: Any, shape: Tuple[int], nr_parties: int, seed_shares: int
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
            res = share.get_copy()
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

    def __getattribute__(self, attr_name: str) -> Any:
        if attr_name in METHODS_FORWARD_ALL_SHARES:

            def method_all_shares(
                _self: MPCTensor, *args: List[Any], **kwargs: Dict[Any, Any]
            ) -> Any:
                shares = []

                for share in _self.child:
                    method = getattr(share, attr_name)
                    new_share = method(*args, **kwargs)
                    shares.append(new_share)

                    dummy_res = getattr(np.empty(_self.mpc_shape), attr_name)(
                        *args, **kwargs
                    )
                    new_shape = dummy_res.shape
                res = MPCTensor(shares=shares, shape=new_shape)
                return res

            return functools.partial(method_all_shares, self)
        return object.__getattribute__(self, attr_name)

    def __apply_private_op(self, other: MPCTensor, op_str: str) -> List[ShareTensor]:
        op = getattr(operator, op_str)
        if isinstance(other, MPCTensor):
            res_shares = [op(a, b) for a, b in zip(self.child, other.child)]
        else:
            raise ValueError("Add works only for the MPCTensor at the moment!")
        return res_shares

    def __apply_public_op(self, y: Any, op_str: str) -> List[ShareTensor]:
        op = getattr(operator, op_str)
        if op_str in {"mul", "matmul", "add", "sub"}:
            res_shares = [op(share, y) for share in self.child]
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

        _self = self
        if ispointer(y):
            client = getattr(y, "client", None)
            if self.parties is None:
                raise Exception("")
            if client is not None and client not in self.parties:
                parties = self.parties + [client]
            else:
                parties = [party for party in self.parties]

            # TODO: Extract info for y shape from somewhere
            # We presume at the moment that it is the same shape
            # Captured: https://app.clubhouse.io/openmined/story/1128/tech-debt-for-adp-smpc-demo?\
            # stories_sort_by=priority&stories_group_by=WORKFLOW_STATE\

            y = MPCTensor(secret=y, shape=self.mpc_shape, parties=parties)

            seed_shares = secrets.randbits(32)

            shares = MPCTensor._get_shares_from_remote_secret(
                secret=None,
                shape=self.mpc_shape,
                parties=parties,
                seed_shares=seed_shares,
            )

            op = getattr(operator, op_str)

            new_shares = [
                op(share1, share2) for share1, share2 in zip(self.child, shares)
            ]

            new_shares.append(shares[-1])

            _self = MPCTensor(shares=new_shares, shape=self.mpc_shape, parties=parties)

        if isinstance(y, MPCTensor):
            result = _self.__apply_private_op(y, op_str)
        else:
            result = _self.__apply_public_op(y, op_str)

        if isinstance(y, (float, int)):
            y_shape = (1,)
        elif isinstance(y, MPCTensor):
            y_shape = y.mpc_shape
        else:
            y_shape = y.shape

        shape = MPCTensor.__get_shape(op_str, self.mpc_shape, y_shape)

        result = MPCTensor(shares=result, shape=shape, parties=_self.parties)

        return result

    def add(
        self, y: Union[int, float, np.ndarray, torch.tensor, MPCTensor]
    ) -> MPCTensor:
        """Apply the "add" operation between "self" and "y".

        Args:
            y (Union[MPCTensor, torch.Tensor, float, int]): self + y

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
        res = MPCTensor(shares=res_shares, shape=new_shape)

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

    __add__ = add
    __radd__ = add
    __sub__ = sub
    __rsub__ = rsub
    __mul__ = mul
    __rmul__ = mul


@implements(MPCTensor, np.add)
def add(x: np.ndarray, y: MPCTensor) -> SupportedChainType:
    return y.add(x)


@implements(MPCTensor, np.subtract)
def sub(x: np.ndarray, y: MPCTensor) -> SupportedChainType:
    return y.rsub(x)


@implements(MPCTensor, np.multiply)
def mul(x: np.ndarray, y: MPCTensor) -> SupportedChainType:
    return y.mul(x)
