# stdlib
from typing import Any
from typing import List

# third party
import numpy as np

# relative
from .. import Tensor
from .. import TensorPointer
from .... import lib
from ....ast.klass import pointerize_args_and_kwargs
from ...node.common.action.function_or_constructor_action import (
    RunFunctionOrConstructorAction,
)
from .mpc_tensor import MPCTensor


def stack(mpc_tensors: List[MPCTensor]) -> MPCTensor:
    shares = list(zip(*[tensor.child for tensor in mpc_tensors]))

    attr_path_and_name = f"syft.core.tensor.smpc.static.stack_helper"
    # We should do downcast and pointerize but we know that mpc_tensor already
    # contain tensors
    ref_shape = mpc_tensors[0].shape

    if any(mpc_tensor.shape != ref_shape for mpc_tensor in mpc_tensors):
        raise ValueError("All MPCTensors should have the same shape!")

    res_shares = []
    for shares_party in shares:
        client = shares_party[0].client

        result = TensorPointer(
            client=client,
            public_shape=ref_shape,
            public_dtype=shares_party[0].public_dtype,
        )

        # QUESTION can the id_at_location be None?
        result_id_at_location = getattr(result, "id_at_location", None)

        args = client.syft.lib.python.List(shares_party)
        cmd = RunFunctionOrConstructorAction(
            path=attr_path_and_name,
            args=shares_party,
            kwargs={},
            id_at_location=result_id_at_location,
            address=shares_party[0].client.address,
            is_static=True,
        )

        client.send_immediate_msg_without_reply(msg=cmd)
        res_shares.append(result)

    shape = (len(mpc_tensors),) + mpc_tensors[0].shape
    ring_size = mpc_tensors[0].ring_size
    result = MPCTensor(
        parties=mpc_tensors[0].parties,
        shares=res_shares,
        shape=shape,
        ring_size=ring_size,
    )
    return result


def stack_helper(*tensors: list[Tensor]) -> Tensor:
    res = np.stack(tensors)
    return res


STATIC_FUNCTIONS = {"stack": stack}
