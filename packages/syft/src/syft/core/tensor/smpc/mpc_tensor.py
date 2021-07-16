# stdlib
import itertools
import operator
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

# third party
import numpy as np

# syft absolute
from syft.core.tensor.passthrough import PassthroughTensor
from syft.core.tensor.smpc.share_tensor import ShareTensor

# syft relative
from .utils import ispointer


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
            seed_shares = 42

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

        if shape is None:
            raise ValueError("Shape of the secret should be known")

        self.mpc_shape = shape

        super().__init__(res)

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
        secret: Any, parties: List[Any], shape: Optional[Tuple[int]], seed_shares: int
    ) -> List[ShareTensor]:
        if ispointer(secret):
            if shape is None:
                raise ValueError("Shape must be specified when the secret is remote")
            return MPCTensor._get_shares_from_remote_secret(
                secret, shape, parties, seed_shares
            )

        return MPCTensor._get_shares_from_local_secret(secret, nr_parties=len(parties))

    @staticmethod
    def _get_shares_from_remote_secret(
        secret: Any, shape: Tuple[int], parties: List[Any], seed_shares: int
    ) -> List[ShareTensor]:
        shares = []
        for i, party in enumerate(parties):
            if party == secret.client:
                value = secret
            else:
                value = None

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
        secret: Any, nr_parties: int
    ) -> List[ShareTensor]:
        # TODO: ShareTensor needs to have serde serializer/deserializer
        shares = ShareTensor.generate_shares(secret=secret, nr_shares=nr_parties)
        return shares

    def reconstruct(self) -> Any:
        # TODO: It might be that the resulted shares (if we run any computation) might
        # not be available at this point

        local_shares = [share.get_copy() for share in self.child]
        is_share_tensor = isinstance(local_shares[0], ShareTensor)

        if is_share_tensor:
            local_shares = [share.child for share in local_shares]

        result = local_shares[0]
        for share in local_shares[1:]:
            result = result + share

        if not is_share_tensor:
            result = result.decode()
        return result

    @staticmethod
    def __get_shape(
        x_shape: Tuple[int],
        y_shape: Tuple[int],
        operator: Callable[[np.ndarray, np.ndarray], Any],
    ) -> Tuple[int]:
        res = operator(np.empty(x_shape), np.empty(y_shape)).shape
        return res

    def apply_private_op(self, other: "MPCTensor", op: str) -> "MPCTensor":
        operation = getattr(operator, op)
        if isinstance(other, MPCTensor):
            res_shares = [operation(a, b) for a, b in zip(self.child, other.child)]
        else:
            raise ValueError("Add works only for the MPCTensor at the moment!")

        new_shape = MPCTensor.__get_shape(self.mpc_shape, other.mpc_shape, operation)
        res = MPCTensor(shares=res_shares, shape=new_shape)
        return res

    def __add__(self, other: "MPCTensor") -> "MPCTensor":
        res = self.apply_private_op(other, "__add__")
        return res

    def __sub__(self, other: "MPCTensor") -> "MPCTensor":
        res = self.apply_private_op(other, "__sub__")
        return res

    def __mul__(self, other: Any) -> "MPCTensor":
        if isinstance(other, MPCTensor):
            raise ValueError("Private multiplication not yet implemented!")
        else:
            res_shares = [
                operator.mul(a, b) for a, b in zip(self.child, itertools.repeat(other))
            ]

        new_shape = MPCTensor.__get_shape(self.mpc_shape, other.shape, operator.mul)
        res = MPCTensor(shares=res_shares, shape=new_shape)

        return res
