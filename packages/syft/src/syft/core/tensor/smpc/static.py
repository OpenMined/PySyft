"""Module to keep static function that should usually stay in the library module.

Examples: torch.stack, torch.argmax
"""
# stdlib
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

# third party
import numpy as np
import torch

# relative
from ..tensor import Tensor
from ..util import parallel_execution
from .mpc_tensor import MPCTensor
from .share_tensor import ShareTensor


def helper_argmax(
    x: MPCTensor,
    axis: Optional[int] = None,
    keepdims: bool = False,
    one_hot: bool = False,
) -> MPCTensor:
    """Compute argmax using pairwise comparisons. Makes the number of rounds fixed, here it is 2.

    This is inspired from CrypTen.

    Args:
        x (MPCTensor): the MPCTensor on which to compute helper_argmax on
        axis (Optional[int]): compute argmax over a specific axis(es)
        keepdims (bool): when one_hot is true, keep all the dimensions of the tensor
        one_hot (bool): return the argmax as a one hot vector

    Returns:
        Given the args, it returns a one hot encoding (as an MPCTensor) or the index
        of the maximum value

    Raises:
        ValueError: In case more max values are found and we need to return the index
    """
    # for each share in MPCTensor
    #   do the algorithm portrayed in paper (helper_argmax_pairwise)
    #   results in creating two matrices and subtraction them

    prep_x = x.flatten() if axis is None else x
    parties = x.parties
    args = [[share_ptr_tensor, axis] for share_ptr_tensor in prep_x.child]

    if not isinstance(prep_x.mpc_shape, tuple) and len(prep_x.mpc_shape) != 0:
        raise ValueError(
            "Expected shape to be tuple and '> 0', but got {len(prep_x.mpc_shape)}!"
        )

    dummy_val = Tensor(ShareTensor.get_dummy_value(prep_x.mpc_shape))
    compare_shape = tuple(helper_argmax_pairwise(dummy_val, axis).shape)
    shares = parallel_execution(helper_argmax_pairwise, parties)(args)

    x_pairwise = MPCTensor(shares=shares, parties=x.parties, shape=compare_shape)

    # with the MPCTensor tensor we check what entries are positive
    # then we check what columns of M matrix have m-1 non-zero entries after comparison
    # (by summing over cols)
    pairwise_comparisons = x_pairwise >= 0
    # re-compute row_length
    _axis = -1 if axis is None else axis
    row_length = prep_x.mpc_shape[_axis] if prep_x.mpc_shape[_axis] > 1 else 2

    result = pairwise_comparisons.sum(0)
    compare_shape = compare_shape[1:]  # Remove the leading dimension because of sum(0)
    result = result >= (row_length - 1)

    return result


def argmax(
    x: MPCTensor,
    axis: Optional[int] = None,
    keepdims: bool = False,
) -> MPCTensor:
    """Compute argmax using pairwise comparisons. Makes the number of rounds fixed, here it is 2.

    This is inspired from CrypTen.

    Args:
        x (MPCTensor): the MPCTensor that argmax will be computed on
        axis (Optional[int]): compute argmax over a specific axis(es) (numpy) / dimensions (torch)
        keepdims (bool): when one_hot is true and dim is set, keep all the dimensions of the tensor

    Returns:
        The index of the maximum value as an MPCTensor
    """
    if keepdims and axis is None:
        raise ValueError("keepdims=True requires 'axis' to be populated")

    argmax = helper_argmax(x, axis=axis, keepdims=keepdims, one_hot=False)
    shape = argmax.mpc_shape

    # TODO: We should make shape always to be passed in at the MPCTensor
    if shape is None:
        raise ValueError("Shape is not and it should be populated")

    if axis is None:
        check = argmax * Tensor(np.arange(np.prod(shape), dtype=np.int64))
        argmax = check.sum()
        """ TODO: Select only one
        For the moment we do not consider that we have more max values
        nr_max_values = argmax >= row_length
        if nr_max_values.reconstruct():
            # In case we have 2 max values, rather then returning an invalid index
            # we raise an exception
            raise ValueError("There are multiple argmax values")
        """
    else:
        argmax_shape = [1] * len(shape)
        argmax_shape[axis] = shape[axis]
        argmax = argmax * Tensor(np.arange(shape[axis], dtype=np.int64)).reshape(
            argmax_shape
        )
        argmax_mpc = argmax.sum(axis=axis, keepdims=keepdims)
    return argmax_mpc


def max(
    x: MPCTensor,
    axis: Optional[int] = None,
    keepdims: bool = False,
) -> Union[MPCTensor, Tuple[MPCTensor, MPCTensor]]:
    """Compute the maximum value for an MPCTensor.

    Args:
        x (MPCTensor): MPCTensor to be computed the maximum value on.
        axis (Optional[int]): The dimension (torch)/ axis (numpy) over which to compute the maximum.
        keepdims (bool): when one_hot is true and dim is set, keep all the dimensions of the tensor

    Returns:
        A tuple representing (max MPCTensor, indices_max MPCTensor)
    """
    if keepdims and axis is None:
        raise ValueError("keepdims=True requires 'axis' to be populated")

    if x.mpc_shape is None:
        raise ValueError(
            "Received an MPCTensor that has None for the attribute 'mpc_shape'!"
        )
    argmax_mpc = helper_argmax(x, axis=axis, keepdims=keepdims, one_hot=True)
    argmax = argmax_mpc.reshape(x.mpc_shape)
    max_mpc = argmax * x
    if axis is None:
        res = max_mpc.sum()
    else:
        shape = argmax_mpc.shape
        if shape is None:
            raise ValueError("Shape for MPCTensor should not be None!")
        size = [1] * len(shape)
        size[axis] = shape[axis]
        argmax_mpc = argmax_mpc * Tensor(np.arange(shape[axis])).reshape(size)
        # Torch has dim/numpy has axis
        # Torch has keepdim/numpy has keepdims
        argmax_mpc = argmax_mpc.sum(axis=axis, keepdims=keepdims)
        max_mpc = max_mpc.sum(axis=axis, keepdims=keepdims)
        res = max_mpc, argmax_mpc

    return res


def helper_argmax_pairwise(share: Tensor, axis: Optional[int] = None) -> Tensor:
    """Helper function that would compute the difference between all the elements in a tensor.

    Credits goes to the CrypTen team.

    Args:
        share (ShareTensor): Share tensor
        axis (Optional[int]): dimension/axis to compute over

    Returns:
        A ShareTensor that represents the difference between each "row" in the ShareTensor.
    """
    axis = -1 if axis is None else axis
    share_shape = share.shape
    row_length = share_shape[axis] if share_shape[axis] > 1 else 2

    is_nparray = False
    data = share.child.child

    # Need to convert to torch tensor for having a common path in the code
    # numpy array has broadcast_to | -> convert those to torch
    # torch has expand             |
    if isinstance(share.child.child, np.ndarray):
        is_nparray = True
        data = torch.tensor(share.child.child)

    # Copy each row (length - 1) times to compare to each other row
    a = data.expand(row_length - 1, *share.shape)

    # Generate cyclic permutations for each row
    b = torch.stack([data.roll(i + 1, dims=axis) for i in range(row_length - 1)])
    res = a - b

    if is_nparray:
        res = res.numpy()

    public_shape = res.shape
    public_dtype = str(res.dtype)
    share.child.child = res

    return Tensor(
        share.child,
        public_shape=public_shape,
        public_dtype=public_dtype,
    )


STATIC_FUNCS: Dict[str, Callable] = {
    "argmax": argmax,
    "max": max,
}
