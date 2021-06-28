# stdlib
from functools import lru_cache
from functools import reduce
import operator

# third party
import numpy as np

# syft absolute
from syft.core.tensor.passthrough import PassthroughTensor
from syft.core.tensor.share_tensor import ShareTensor


def is_pointer(val):
    if "Pointer" in type(val).__name__:
        return True


class MPCTensor(PassthroughTensor):
    def __init__(
        self,
        parties=None,
        secret=None,
        shares=None,
        shape=None,
        seeds_przs_generators=None,
    ):
        if secret is None and shares is None:
            raise ValueError("Secret or shares should be populated!")

        if seeds_przs_generators is None:
            seeds_przs_generators = [42, 43]

        if secret is not None:
            shares = MPCTensor._get_shares_from_secret(
                secret=secret,
                parties=parties,
                shape=shape,
                seeds_przs_generators=seeds_przs_generators,
            )

        res = MPCTensor._mpc_from_shares(shares, parties)
        self.mpc_shape = shape

        super().__init__(res)

    @staticmethod
    def _mpc_from_shares(shares, parties):
        if not isinstance(shares, list):
            raise ValueError("_mpc_from_shares expected a list of shares")

        if is_pointer(shares[0]):
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
    def _get_shares_from_secret(secret, parties, shape, seeds_przs_generators):
        if is_pointer(secret):
            if shape is None:
                raise ValueError("Shape must be specified when the secret is remote")
            return MPCTensor._get_shares_from_remote_secret(
                secret, shape, parties, seeds_przs_generators
            )

        return MPCTensor._get_shares_from_local_secret(secret, nr_parties=len(parties))

    @staticmethod
    def _get_shares_from_remote_secret(secret, shape, parties, seeds_przs_generators):
        shares = []
        for i, party in enumerate(parties):
            next_party_idx = (i + 1) % len(parties)
            party_seeds_przs_generators = [
                seeds_przs_generators[i],
                seeds_przs_generators[next_party_idx],
            ]
            if party == secret.client:
                value = secret
            else:
                value = None

            remote_share = (
                party.syft.core.tensor.share_tensor.ShareTensor.generate_przs(
                    rank=i,
                    value=value,
                    shape=shape,
                    seeds_przs_generators=party_seeds_przs_generators,
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
        # share should be: FPT > ShareTensor
        local_shares = [share.get() for share in self.child]
        for local_share in local_shares:
            local_share.child = local_share.child.child

        result_fp = local_shares[0]
        for fpt in local_shares[1:]:
            result_fp = result_fp + fpt
        result = result_fp.decode()
        return result

    @staticmethod
    def __get_shape(x_shape, y_shape, operator):
        res = operator(np.empty(x_shape), np.empty(y_shape)).shape
        return res

    def __add__(self, other):
        if isinstance(other, MPCTensor):
            res_shares = [operator.add(a, b) for a, b in zip(self.child, other.child)]
        elif isinstance(other, np.ndarray):
            # Only party with rank 0 will add the value
            res_shares = [operator.add(self.child[0], other)]
            res_shares.extend(self.child[1:])
        else:
            raise ValueError(f"other should be a numpy like structure, but it is {type(other)}")
        new_shape = MPCTensor.__get_shape(self.mpc_shape, other.mpc_shape, operator.add)
        res = MPCTensor(shares = res_shares, shape=new_shape)

        return res
