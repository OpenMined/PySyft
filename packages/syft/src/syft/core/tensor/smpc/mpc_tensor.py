# stdlib
from functools import lru_cache
from functools import reduce
import itertools
import operator

# third party
import numpy as np

# syft absolute
from syft import logger
from syft.core.tensor.passthrough import PassthroughTensor
from syft.core.tensor.smpc.share_tensor import ShareTensor
from syft.core.tensor.tensor import Tensor

# syft relative
from .utils import ispointer
from .utils import parallel_execution


class MPCTensor(PassthroughTensor):
    def __init__(
        self, parties=None, secret=None, shares=None, shape=None, seed_shares=None
    ):
        if secret is None and shares is None:
            raise ValueError("Secret or shares should be populated!")

        if seed_shares is None:
            seed_shares = 42

        if secret is not None:
            shares = MPCTensor._get_shares_from_secret(
                secret=secret,
                parties=parties,
                shape=shape,
                seed_shares=seed_shares,
            )

        res = MPCTensor._mpc_from_shares(shares, parties)
        self.mpc_shape = shape

        super().__init__(res)

    @staticmethod
    def _mpc_from_shares(shares, parties):
        if not isinstance(shares, list):
            raise ValueError("_mpc_from_shares expected a list of shares")

        if ispointer(shares[0]):
            # Remote shares
            return shares
        else:
            MPCTensor._mpc_from_local_shares(shares, parties)

    @staticmethod
    def _mpc_from_local_shares(shares, parties):
        # TODO: ShareTensor needs to have serde serializer/deserializer
        shares_ptr = [share.send(party) for share, party in zip(shares, parties)]
        return shares_ptr

    @staticmethod
    def _get_shares_from_secret(secret, parties, shape, seed_shares):
        if ispointer(secret):
            if shape is None:
                raise ValueError("Shape must be specified when the secret is remote")
            return MPCTensor._get_shares_from_remote_secret(
                secret, shape, parties, seed_shares
            )

        return MPCTensor._get_shares_from_local_secret(secret, nr_parties=len(parties))

    @staticmethod
    def _get_shares_from_remote_secret(secret, shape, parties, seed_shares):
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
    def _get_shares_from_local_secret(secret, nr_parties):
        # TODO: ShareTensor needs to have serde serializer/deserializer
        shares = ShareTensor.generate_shares(secret=secret, nr_shares=nr_parties)
        return shares

    def reconstruct(self):
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
    def __get_shape(x_shape, y_shape, operator):
        res = operator(np.empty(x_shape), np.empty(y_shape)).shape
        return res

    def apply_private_op(self, other, op):
        operation = getattr(operator, op)
        if isinstance(other, MPCTensor):
            res_shares = [operation(a, b) for a, b in zip(self.child, other.child)]
        else:
            raise ValueError("Add works only for the MPCTensor at the moment!")

        new_shape = MPCTensor.__get_shape(self.mpc_shape, other.mpc_shape, operation)
        res = MPCTensor(shares=res_shares, shape=new_shape)
        return res

    def __add__(self, other):
        res = self.apply_private_op(other, "__add__")
        return res

    def __sub__(self, other):
        res = self.apply_private_op(other, "__sub__")
        return res

    def __mul__(self, other):
        if isinstance(other, MPCTensor):
            raise ValueError("Private multiplication not yet implemented!")
        else:
            res_shares = [
                operator.mul(a, b) for a, b in zip(self.child, itertools.repeat(other))
            ]

        new_shape = MPCTensor.__get_shape(self.mpc_shape, other.shape, operator.mul)
        res = MPCTensor(shares=res_shares, shape=new_shape)

        return res
