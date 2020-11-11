import torch

from syft.frameworks.torch.tensors.interpreters.replicated_shared import ReplicatedSharingTensor
from syft.frameworks.torch.mpc.falcon.falcon_helper import FalconHelper


class ABY3Helper:
    def bit_inject(rst_binary):
        assert rst_binary.ring_size == 2

        return_wrapper = rst_binary.is_wrapper
        if return_wrapper:
            rst_binary = rst_binary.child

        players = rst_binary.players
        shares_p0, shares_p1, shares_p2 = rst_binary.retrieve_pointers()

        """
            x shared in ring size 2
            x_b = (x_b_0, x_b_1, x_b_2)

            Share the shares in ring size L
            Resharing the binary shares in arithmetic ones should have no communication cost
            but because we use an orchestrator there is added the communication overhead

            Step 1: x1_shares = ((x_b_0, 0), (0, 0), (0, x_b_0))
            Step 2: x2_shares = ((0, x_b_1), (x_b_1, 0), (0, 0))
            Step 3: x3_shares = ((0, 0), (0, x_b_2), (x_b_2, 0))

            For emulating a xor on arithmetic shares:
                * [[v1 xor v2]] = [[v1]] + [[v2]] - 2 * [[v1]] * [[v2]]

            (x0_shares xor x1_shares xor x2_shares) == x_b (ring size L)
        """

        # Step 1
        rst1 = FalconHelper.binary_to_arithmetic((shares_p0[0], shares_p2[1]), players)

        # Step 2
        rst2 = FalconHelper.binary_to_arithmetic((shares_p1[0], shares_p0[1]), players)

        # Step 3
        rst3 = FalconHelper.binary_to_arithmetic((shares_p2[0], shares_p1[1]), players)

        tmp = FalconHelper.xor(rst1, rst2)
        res = FalconHelper.xor(tmp, rst3)

        if return_wrapper:
            return res.wrap()

        return res
