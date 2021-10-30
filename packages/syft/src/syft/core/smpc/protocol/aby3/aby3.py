"""ABY3 Protocol.

ABY3 : A Mixed Protocol Framework for Machine Learning.
https://eprint.iacr.org/2018/403.pdf
"""
# stdlib
from functools import reduce
import secrets
from typing import Any
from typing import List
from uuid import UUID

# third party
import numpy as np

# relative
from .....ast.klass import get_run_class_method
from ....common import UID
from ....tensor.smpc.mpc_tensor import MPCTensor
from ....tensor.smpc.share_tensor import ShareTensor
from ....tensor.smpc.utils import get_nr_bits


class ABY3:
    """ABY3 Protocol Implementation."""

    def __eq__(self, other: Any) -> bool:
        """Check if "self" is equal with another object given a set of attributes to compare.

        Args:
            other (Any): Object to compare

        Returns:
            bool: True if equal False if not.
        """
        if type(self) != type(other):
            return False

        return True

    @staticmethod
    def bit_injection(x: MPCTensor, ring_size: int) -> MPCTensor:
        """Perform ABY3 bit injection for conversion of binary share to arithmetic share.

        Args:
            x (MPCTensor) : MPCTensor with shares of bit.
            ring_size (int) : Ring size of arithmetic share to convert.

        Returns:
            arith_share (MPCTensor): Arithmetic shares of bit in input ring size.

        Raises:
            ValueError: If input tensor is not binary shared.
            ValueError: If the exactly three parties are not involved in the computation.
        """
        # relative
        # relative
        from ....tensor import TensorPointer

        shape = x.shape
        parties = x.parties
        seed_id_locations = secrets.randbits(64)
        kwargs = {"seed_id_locations": seed_id_locations}
        path_and_name = x.child[0].path_and_name
        attr_path_and_name = f"{x.child[0].path_and_name}.bit_decomposition"

        if not isinstance(x.child[0], TensorPointer):
            decomposed_shares = [
                share.bit_decomposition(share, ring_size, False, **kwargs)
                for share in x.child
            ]
        else:
            decomposed_shares = []
            attr_path_and_name = f"{x.child[0].path_and_name}.bit_decomposition"
            op = get_run_class_method(attr_path_and_name, SMPC=True)
            for share in x.child:
                decomposed_shares.append(op(share, share, ring_size, False, **kwargs))

        decomposed_shares = ABY3.pregenerate_pointers(
            parties, 1, path_and_name, seed_id_locations
        )

        # List which contains the share of a single bit
        res_shares: List[MPCTensor] = []

        bit_shares = [share[0] for share in decomposed_shares]
        bit_shares = zip(*bit_shares)  # type: ignore
        for bit_sh in bit_shares:
            mpc = MPCTensor(
                shares=bit_sh, shape=shape, parties=parties, ring_size=ring_size
            )
            res_shares.append(mpc)

        # TODO: Should modify to xor at mpc tensor level
        arith_share = reduce(lambda a, b: a + b - (a * b * 2), res_shares)

        return arith_share

    @staticmethod
    def full_adder(a: List[MPCTensor], b: List[MPCTensor]) -> List[MPCTensor]:
        """Perform bit addition on MPCTensors using a full adder.

        Args:
            a (List[MPCTensor]): MPCTensor with shares of bit.
            b (List[MPCTensor]): MPCTensor with shares of bit.

        Returns:
            result (List[MPCTensor]): Result of the operation.

        TODO: Should modify ripple carry adder to parallel prefix adder.
        """
        ring_size = 2 ** 32
        ring_bits = get_nr_bits(ring_size)
        c = np.array([0], dtype=np.bool)  # carry bits of addition.
        result: List[MPCTensor] = []
        for idx in range(ring_bits):
            s = a[idx] + b[idx] + c
            c = a[idx] * b[idx] + c * (a[idx] + b[idx])
            result.append(s)
        return result

    @staticmethod
    def bit_decomposition(x: MPCTensor) -> List[MPCTensor]:
        """Perform ABY3 bit decomposition for conversion of arithmetic share to binary share.

        Args:
            x (MPCTensor): Arithmetic shares of secret.

        Returns:
            bin_share (List[MPCTensor]): Returns binary shares of each bit of the secret.

        TODO : Should be modified to use parallel prefix adder when multiprocessing
        functionality is integrated
        """
        # relative
        from ....tensor import TensorPointer

        nr_parties = len(x.parties)
        ring_size = 2 ** 32  # Should extract this info better
        ring_bits = get_nr_bits(ring_size)
        shape = x.shape
        parties = x.parties

        seed_id_locations = secrets.randbits(64)
        kwargs = {"seed_id_locations": seed_id_locations}
        path_and_name = x.child[0].path_and_name
        attr_path_and_name = f"{x.child[0].path_and_name}.bit_decomposition"

        if not isinstance(x.child[0], TensorPointer):
            decomposed_shares = [
                share.bit_decomposition(share, 2, True, **kwargs) for share in x.child
            ]
        else:
            decomposed_shares = []
            op = get_run_class_method(attr_path_and_name, SMPC=True)
            for share in x.child:
                decomposed_shares.append(op(share, share, 2, True, **kwargs))

        decomposed_shares = ABY3.pregenerate_pointers(
            parties, ring_bits, path_and_name, seed_id_locations
        )

        # List which contains the share of each share.
        # TODO: Shouldn't this be an empty list? and we append to it?
        res_shares: List[List[MPCTensor]] = [[] for _ in range(nr_parties)]

        for idx in range(ring_bits):
            bit_shares = [share[idx] for share in decomposed_shares]
            bit_shares = zip(*bit_shares)  # type: ignore
            for i, bit_sh in enumerate(bit_shares):
                mpc = MPCTensor(
                    shares=bit_sh, shape=shape, parties=parties, ring_size=2
                )
                res_shares[i].append(mpc)

        bin_share = reduce(ABY3.full_adder, res_shares)

        return bin_share

    @staticmethod
    def pregenerate_pointers(
        parties: List[Any], ring_bits: int, path_and_name: str, seed_id_locations: int
    ) -> List[List[List[ShareTensor]]]:
        generator = np.random.default_rng(seed_id_locations)
        # Skip the first ID ,as it is used for None return type in run class method.
        _ = UID(UUID(bytes=generator.bytes(16)))

        nr_parties = len(parties)
        resolved_pointer_type = []
        for party in parties:
            resolved_pointer_type.append(party.lib_ast.query(path_and_name))

        # TODO : Simplify it.
        share_pointers: List[List[List[Any]]] = [[] for _ in range(nr_parties)]

        for _ in range(ring_bits * nr_parties):
            for idx, party in enumerate(parties):
                id_at_location = UID(UUID(bytes=generator.bytes(16)))
                result = resolved_pointer_type[idx].pointer_type(client=party)
                result.id_at_location = id_at_location
                share_pointers[idx].append(result)

        share_pointers = [
            [
                share_lst[i : i + nr_parties]  # noqa
                for i in range(len(share_lst), nr_parties)
            ]
            for share_lst in share_pointers
        ]

        return share_pointers
