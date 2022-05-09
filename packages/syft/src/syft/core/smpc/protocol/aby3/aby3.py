"""ABY3 Protocol.

ABY3 : A Mixed Protocol Framework for Machine Learning.
https://eprint.iacr.org/2018/403.pdf
"""
# stdlib
from copy import deepcopy
from functools import reduce
import secrets
from typing import Any
from typing import Dict
from typing import List
from typing import Union
from uuid import UUID

# third party
import numpy as np
from tqdm import tqdm

# relative
from .....ast.klass import get_run_class_method
from ....common import UID
from ....tensor.config import DEFAULT_RING_SIZE
from ....tensor.smpc import context
from ....tensor.smpc import utils
from ....tensor.smpc.mpc_tensor import MPCTensor
from ....tensor.smpc.share_tensor import ShareTensor
from ....tensor.smpc.utils import get_nr_bits
from ...store.crypto_primitive_provider import CryptoPrimitiveProvider


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
        from ....tensor import TensorPointer

        shape = x.shape
        parties = x.parties
        seed_id_locations = secrets.randbits(64)
        kwargs = {"seed_id_locations": seed_id_locations}
        path_and_name = x.child[0].path_and_name
        attr_path_and_name = f"{x.child[0].path_and_name}.bit_decomposition"

        if not isinstance(x.child[0], TensorPointer):
            decomposed_shares = [
                share.bit_decomposition(share, str(ring_size), False, **kwargs)
                for share in x.child
            ]
        else:
            decomposed_shares = []
            op = get_run_class_method(attr_path_and_name, SMPC=True)
            for share in x.child:
                decomposed_shares.append(
                    op(share, share, str(ring_size), False, **kwargs)
                )

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
        parties = a[0].parties
        parties_info = a[0].parties_info

        shape_x = tuple(a[0].shape)  # type: ignore
        shape_y = tuple(b[0].shape)  # type: ignore
        ring_size = DEFAULT_RING_SIZE

        # For ring_size 2 we generate those before hand
        CryptoPrimitiveProvider.generate_primitives(
            "beaver_mul",
            nr_instances=128,
            parties=parties,
            g_kwargs={
                "a_shape": shape_x,
                "b_shape": shape_y,
                "parties_info": parties_info,
            },
            p_kwargs={"a_shape": shape_x, "b_shape": shape_y},
            ring_size=2,
        )

        ring_bits = get_nr_bits(ring_size)
        c = np.array([0], dtype=np.bool)  # carry bits of addition.
        result: List[MPCTensor] = []
        for idx in range(ring_bits):
            print("Bit Number :", idx)
            s_tmp = a[idx] + b[idx]
            s = s_tmp + c
            if idx != ring_bits - 1:
                c = a[idx] * b[idx] + c * s_tmp
                c.block
            result.append(s)
        return result

    @staticmethod
    def full_adder_spdz_compiler(
        a: List[ShareTensor], b: List[ShareTensor]
    ) -> List[ShareTensor]:
        # Specialized for 2 parties
        """Perform bit addition on ShareTensor using a full adder.

        Args:
            a (List[ShareTensor]): ShareTensor with shares of bit.
            b (List[ShareTensor]): ShareTensor with shares of bit.

        Returns:
            result (List[ShareTensor]): Result of the operation.

        TODO: Should modify ripple carry adder to parallel prefix adder.
        """
        # parties = a[0].parties
        # parties_info = a[0].parties_info

        shape_x = tuple(a[0].shape)  # type: ignore
        shape_y = tuple(b[0].shape)  # type: ignore
        ring_size = DEFAULT_RING_SIZE

        # # For ring_size 2 we generate those before hand
        # CryptoPrimitiveProvider.generate_primitives(
        #     "beaver_mul",
        #     nr_instances=64,
        #     parties=parties,
        #     g_kwargs={
        #         "a_shape": shape_x,
        #         "b_shape": shape_y,
        #         "parties_info": parties_info,
        #     },
        #     p_kwargs={"a_shape": shape_x, "b_shape": shape_y},
        #     ring_size=2,
        # )

        ring_bits = get_nr_bits(ring_size)

        carry = np.zeros(a[0].mpc_shape, dtype=np.bool)
        # one = MPCTensor(
        #     parties=parties,
        #     secret=Tensor(np.ones(a[0].mpc_shape, dtype=np.bool)),
        #     shape=a[0].mpc_shape,
        # )

        result: List[ShareTensor] = []

        def majority(
            a: Union[ShareTensor, np.ndarray],
            b: Union[ShareTensor, np.ndarray],
            c: Union[ShareTensor, np.ndarray],
        ) -> ShareTensor:

            return (a + c + np.array(1, dtype=bool)) * (b + c) + b

        for idx in tqdm(range(ring_bits), desc="Computing..."):
            s = a[idx] + b[idx] + carry
            if idx != ring_bits - 1:
                carry = majority(a[idx], b[idx], carry)
                # time.sleep(1)
                carry.block
            result.append(s)
        return result

    # @staticmethod
    # def pregenerate_pointers(
    #     parties: List[Any], ring_bits: int, path_and_name: str, seed_id_locations: int
    # ) -> List[List[List[ShareTensor]]]:
    #     generator = np.random.default_rng(seed_id_locations)
    #     # Skip the first ID ,as it is used for None return type in run class method.
    #     _ = UID(UUID(bytes=generator.bytes(16)))

    #     nr_parties = len(parties)
    #     resolved_pointer_type = [
    #         party.lib_ast.query(path_and_name) for party in parties
    #     ]

    #     """
    #     Consider bit decomposition.
    #     We create a share of share such that.
    #     Assume we are operating in ring_size 2**32(32 bits)
    #     Since we operate in n-out-of-n secret sharing, each party has a single share.
    #     Consider two parties such that a secret x is split as
    #     x = x1+x2
    #     To create share of shares,we use the intuition that the shares not held by party are made zero

    #                 Party1        Party2

    #     x1           x1             0

    #     x2           0               x2

    #     x_i_j denotes the shares held by jth party of ith share

    #     x_1_1 = x1
    #     x_1_2 = 0
    #     x_2_1 = 0
    #     x_2_2 = x2

    #     Party 1 = [x_1_1,x_2_1]
    #     Party 2  =[x_1_2,x_2_2]

    #     Now each party party has share of shares
    #     In bit decomposition, we split each bit and create share of shares

    #     Party 1 = [ [share of shares of first bit] ,[...second bit] ...[ nth bit]]

    #     Note: Count (share of shares for a particular bit) = number of parties
    #     """
    #     share_pointers: List[List[List[Any]]] = [[] for _ in range(nr_parties)]

    #     for _ in range(ring_bits * nr_parties):
    #         id_at_location = UID(UUID(bytes=generator.bytes(16)))
    #         for idx, party in enumerate(parties):
    #             result = resolved_pointer_type[idx].pointer_type(client=party)
    #             result.id_at_location = id_at_location
    #             share_pointers[idx].append(result)

    #     share_pointers = [
    #         [
    #             share_lst[i : i + nr_parties]  # noqa
    #             for i in range(0, len(share_lst), nr_parties)
    #         ]
    #         for share_lst in share_pointers
    #     ]

    #     return share_pointers

    @staticmethod
    def local_decomposition(x: ShareTensor, ring_size: int, bitwise: bool) -> None:
        """Performs local decomposition to generate shares of shares.

        Args:
            x (ShareTensor) : input ShareTensor.
            ring_size (str) : Ring size to generate decomposed shares in.
            bitwise (bool): Perform bit level decomposition on bits if set.

        Returns:
            List[List[ShareTensor]]: Decomposed shares in the given ring size.
        """
        rank = x.rank
        nr_parties = x.nr_parties
        numpy_type = utils.RING_SIZE_TO_TYPE[ring_size]
        shape = x.shape
        zero = np.zeros(shape, numpy_type)

        input_shares = []
        share_lst: List[List[ShareTensor]]

        if bitwise:
            ring_bits = utils.get_nr_bits(x.ring_size)  # for bit-wise decomposition
            input_shares = [x.bit_extraction(idx) for idx in range(ring_bits)]
        else:
            input_shares.append(x)

        for share in input_shares:
            bit_sh = []
            for i in range(nr_parties):
                sh = x.copy_tensor()
                sh.ring_size = ring_size
                if rank != i:
                    sh.child = deepcopy(zero)
                else:
                    sh.child = deepcopy(share.child.astype(numpy_type))
                bit_sh.append(sh)
            share_lst.append(bit_sh)

        return share_lst

    @staticmethod
    def bit_decomposition(x: ShareTensor) -> List[ShareTensor]:
        """Perform ABY3 bit decomposition for conversion of arithmetic share to binary share.

        Args:
            x (ShareTensor): Arithmetic shares of secret.

        Returns:
            bin_share (List[ShareTensor]): Returns binary shares of each bit of the secret.

        TODO : Should be modified to use parallel prefix adder when multiprocessing
        functionality is integrated
        """
        nr_parties = x.nr_parties
        res_shares = []
        decomposed_shares = ABY3.local_decomposition(x, 2, True)
        res_shares = list(zip(*decomposed_shares))  # Bit sharing for each party.

        # return res_shares
        if nr_parties == 2:
            # Specialized for two parties
            bin_share = ABY3.full_adder_spdz_compiler(res_shares[0], res_shares[1])
        else:
            bin_share = reduce(ABY3.full_adder, res_shares)

        return bin_share

    @staticmethod
    def lt(x: ShareTensor, y: Union[ShareTensor, np.ndarray]) -> ShareTensor:
        """Compute less than operator

        Args:
            x (ShareTensor): First input value
            y (ShareTensor): Second input value

        Returns:
            res (ShareTensor): Output of the comparision computation.
        """
        # Step 1: diff = x - y
        # Step 2: Local decomposition  -----| ____ Bit Decomposition
        # Step 3: sum carry adder      -----|
        # Step 4: res = sign(msb)   -------->      Bit Injection
        seed_id_locations = context.SMPC_CONTEXT.get("seed_id_locations", None)
        if seed_id_locations is None:
            raise ValueError(
                f"Input seed : {seed_id_locations} for comparison should not None"
            )
        generator = np.random.default_rng(seed_id_locations)
        # Skip the first ID ,as it is used for None return type in run class method.
        _ = UID(UUID(bytes=generator.bytes(16)))

        diff: ShareTensor = x - y
        res_shares = ABY3.bit_decomposition(diff)
        msb = res_shares[-1]
        res = ABY3.bit_injection(msb, DEFAULT_RING_SIZE)
        return res
