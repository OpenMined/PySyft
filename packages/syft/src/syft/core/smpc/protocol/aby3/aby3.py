"""ABY3 Protocol.

ABY3 : A Mixed Protocol Framework for Machine Learning.
https://eprint.iacr.org/2018/403.pdf
"""
# stdlib
# import itertools
from typing import Any
from typing import List

# relative
from ....tensor.smpc.mpc_tensor import MPCTensor

# from syft.core.tensor.smpc.mpc_tensor import ShareTensor
from ....tensor.smpc.utils import get_nr_bits

# from typing import Tuple
# from typing import Union

# import numpy as np


# from syft.core.tensor.smpc.utils import RING_SIZE_TO_TYPE

# from sympc.utils import parallel_execution

# gen = csprng.create_random_device_generator()
# NR_PARTIES = 3  # constant for aby3 protocols


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

    # @staticmethod
    # def bit_injection(x: MPCTensor, ring_size: int) -> MPCTensor:
    #     """Perform ABY3 bit injection for conversion of binary share to arithmetic share.

    #     Args:
    #         x (MPCTensor) : MPCTensor with shares of bit.
    #         ring_size (int) : Ring size of arithmetic share to convert.

    #     Returns:
    #         arith_share (MPCTensor): Arithmetic shares of bit in input ring size.

    #     Raises:
    #         ValueError: If input tensor is not binary shared.
    #         ValueError: If the exactly three parties are not involved in the computation.
    #     """
    #     input_ring = int(x.share_ptrs[0].get_ring_size().get_copy())  # input ring_size
    #     if input_ring != 2:
    #         raise ValueError("Bit injection works only for binary rings")

    #     args = [[share, str(ring_size)] for share in x.share_ptrs]

    #     decomposed_shares = parallel_execution(
    #         ABY3.local_decomposition, session.parties
    #     )(args)

    #     # Using zip for grouping on pointers is compute intensive.
    #     x1_share = []
    #     x2_share = []
    #     x3_share = []

    #     for share in list(
    #         map(lambda x: x[0].resolve_pointer_type(), decomposed_shares)
    #     ):
    #         x1_share.append(share[0].resolve_pointer_type())
    #         x2_share.append(share[1].resolve_pointer_type())
    #         x3_share.append(share[2].resolve_pointer_type())

    #     x1 = MPCTensor(shares=x1_share, session=session, shape=x.shape)
    #     x2 = MPCTensor(shares=x2_share, session=session, shape=x.shape)
    #     x3 = MPCTensor(shares=x3_share, session=session, shape=x.shape)

    #     arith_share = x1 ^ x2 ^ x3

    #     return arith_share

    @staticmethod
    def full_adder(a: List[MPCTensor], b: List[MPCTensor]) -> List[MPCTensor]:
        """Perform bit addition on MPCTensors using a full adder.

        Args:
            a (List[MPCTensor]): MPCTensor with shares of bit.
            b (List[MPCTensor]): MPCTensor with shares of bit.

        Returns:
            result (List[MPCTensor]): Result of the operation.

        TODO: Should modify ripple carry adder to parallel prefix adder,currently unused.
        """
        ring_size = 2 ** 32
        ring_bits = get_nr_bits(ring_size)
        c = 0  # carry bits of addition.
        result: List[MPCTensor] = []
        for idx in range(ring_bits):
            s = a[idx] + b[idx] + c
            c = a[idx] * b[idx] + c * (a[idx] + b[idx])
            result.append(s)
        return result

    # @staticmethod
    # def bit_decomposition(x: MPCTensor, session: Session) -> List[MPCTensor]:
    #     """Perform ABY3 bit decomposition for conversion of arithmetic share to binary share.

    #     Args:
    #         x (MPCTensor): Arithmetic shares of secret.
    #         session (Session): session the share belongs to.

    #     Returns:
    #         bin_share (List[MPCTensor]): Returns binary shares of each bit of the secret.

    #     TODO : Should be modified to use parallel prefix adder when multiprocessing
    #     functionality is integrated,currently unused.
    #     """
    #     x1: List[MPCTensor] = []  # bit shares of shares
    #     x2: List[MPCTensor] = []
    #     x3: List[MPCTensor] = []

    #     args = [[share, "2", True] for share in x.share_ptrs]

    #     decomposed_shares = parallel_execution(
    #         ABY3.local_decomposition, session.parties
    #     )(args)

    #     # Initially we have have List[p1,p2,p3] where p1,p2,p3 are list returned from parties.
    #     # Each of p1,p2,p3 is List[ [x1,x2,x3] ,...] in bit length of the session ring size.
    #     # Each element of the list is a share of the shares for each bit.
    #     x_sh = itertools.starmap(zip, zip(*decomposed_shares))

    #     for x1_sh, x2_sh, x3_sh in x_sh:
    #         x1_sh = [ptr.resolve_pointer_type() for ptr in x1_sh]
    #         x2_sh = [ptr.resolve_pointer_type() for ptr in x2_sh]
    #         x3_sh = [ptr.resolve_pointer_type() for ptr in x3_sh]

    #         x1_m = MPCTensor(shares=x1_sh, session=session, shape=x.shape)
    #         x2_m = MPCTensor(shares=x2_sh, session=session, shape=x.shape)
    #         x3_m = MPCTensor(shares=x3_sh, session=session, shape=x.shape)

    #         x1.append(x1_m)
    #         x2.append(x2_m)
    #         x3.append(x3_m)

    #     x1_2 = ABY3.full_adder(x1, x2, session)
    #     bin_share = ABY3.full_adder(x1_2, x3, session)

    #     return bin_share

    # @staticmethod
    # def bit_decomposition_ttp(x: MPCTensor, session: Session) -> List[MPCTensor]:
    #     """Perform ABY3 bit decomposition using orchestrator as ttp.

    #     Args:
    #         x (MPCTensor): Arithmetic shares of secret.
    #         session (Session): session the share belongs to.

    #     Returns:
    #         b_sh (List[MPCTensor]): Returns binary shares of each bit of the secret.

    #     TODO: We should modify to use parallel prefix adder, which requires multiprocessing.
    #     """
    #     # Decoding is not done as they are shares of PRRS.
    #     tensor = x.reconstruct(decode=False)
    #     b_sh: List[MPCTensor] = []  # binary shares of bits
    #     ring_size = session.ring_size
    #     shares_sum = ReplicatedSharedTensor.shares_sum
    #     ring_bits = get_nr_bits(ring_size)

    #     for idx in range(ring_bits):
    #         bit_mask = torch.ones(tensor.shape, dtype=tensor.dtype) << idx
    #         secret = (tensor & bit_mask).type(torch.bool)
    #         r1 = torch.empty(size=tensor.shape, dtype=torch.bool).random_(generator=gen)
    #         r2 = torch.empty(size=tensor.shape, dtype=torch.bool).random_(generator=gen)
    #         r3 = shares_sum([secret, r1, r2], ring_size=2)
    #         shares = [r1, r2, r3]
    #         config = Config(encoder_base=1, encoder_precision=0)
    #         sh_ptr = ReplicatedSharedTensor.distribute_shares(
    #             shares=shares, session=session, ring_size=2, config=config
    #         )
    #         b_mpc = MPCTensor(shares=sh_ptr, session=session, shape=tensor.shape)
    #         b_sh.append(b_mpc)

    #     return b_sh
