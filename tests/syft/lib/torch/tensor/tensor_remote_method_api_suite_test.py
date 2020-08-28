"""In this test suite, we code generate a test for the full PyTorch Tensor
method API. We do this by getting the "allowlist" of what we allow  to be
called from the syft package. We then permute this list over all tensor
types and ensure that they can be executed in a remote environment.
"""

from syft.lib.torch import allowlist
from syft.core.pointer.pointer import Pointer
from syft.lib.torch.tensor_util import TORCH_STR_DTYPE
from syft.lib.python.primitive import isprimitive

import syft as sy
import torch as th
import pytest
from itertools import product
from typing import List
from typing import Any
from typing import Dict
from typing import Type
from typing import Callable


tensor_type = type(th.tensor([1, 2, 3]))

# Currently, we do not have constructors with torch.Tensor
# for dtype in ["complex*", "q*"] (complex and quantized types)
TYPES_EXCEPTIONS_PREFIX = ("complex", "q")

TEST_TYPES = [
    e for e in TORCH_STR_DTYPE.keys() if not e.startswith(TYPES_EXCEPTIONS_PREFIX)
]


BASIC_OPS = ["unfold"]
BASIC_OPS_RETURN_TYPE = {"unfold": "torch.Tensor"}
# BASIC_OPS = list()
# BASIC_OPS_RETURN_TYPE = {}
# for method, return_type in allowlist.items():
#     if "torch.Tensor." in method:
#         method_name = method.split(".")[-1]
#         BASIC_OPS.append(method_name)
#         BASIC_OPS_RETURN_TYPE[method_name] = return_type

BASIC_SELF_TENSORS: List[Any] = list()
BASIC_SELF_TENSORS.append([-1, 0, 1, 2, 3, 4])  # with a 0
# BASIC_INPUT_TENSORS.append([-1, 1, 2, 3, 4]) # without a 0
# BASIC_INPUT_TENSORS.append([-0.1, 0.1, 0.2, 0.3, 0.4]) # with a decimal
BASIC_SELF_TENSORS.append([[-0.1, 0.1], [0.2, 0.3]])  # square

BASIC_METHOD_ARGS = list()
BASIC_METHOD_ARGS.append("self")
BASIC_METHOD_ARGS.append("None")
BASIC_METHOD_ARGS.append("[0]")
BASIC_METHOD_ARGS.append("[1]")
BASIC_METHOD_ARGS.append("0")
BASIC_METHOD_ARGS.append("1")

# BASIC_METHOD_ARGS.append("True") #TODO: add ints as remote argument types (can't call .serialize() on bool)
# BASIC_METHOD_ARGS.append("False") #TODO: add ints as remote argument types (can't call .serialize() on bool)

TEST_DATA = list(product(TEST_TYPES, BASIC_OPS, BASIC_SELF_TENSORS, BASIC_METHOD_ARGS))

# Step 1: Create remote worker
alice = sy.VirtualMachine(name="alice")
alice_client = alice.get_client()


def is_expected_runtime_error(msg: str) -> bool:
    expected_msgs = {
        "not implemented for",
        "two bool tensors is not supported.",
        "ZeroDivisionError",
        "not supported on",
        "Can only calculate the mean of floating types",
        "expected a tensor with 2 or more dimensions of floating types",
        "only supports floating-point dtypes",
        "invalid argument 1: A should be 2 dimensional at",
        "invalid argument 1: expected a matrix at",
        "Expected object of scalar type Long but got scalar type",
        "expected total dims >= 2, but got total dims = 1",
        "Integer division of tensors using div or / is no longer supported",
        "result type Float can't be cast to the desired output type",
        "inconsistent tensor size, expected tensor",
        "size mismatch",
        "1D tensors expected, got 2D",
        "requested resize to",
        "ger: Expected 1-D ",
        "At least one of 'min' or 'max' must not be None",
        "Boolean value of Tensor with more than one value is ambiguous",
        "shape '[1]' is invalid for input of size",  # BASIC_METHOD_ARGS.append("[1]")
        "Negation, the `-` operator, on a bool tensor is not supported",
    }

    return any(expected_msg in msg for expected_msg in expected_msgs)


def is_expected_type_error(msg: str) -> bool:
    expected_msgs = {
        "received an invalid combination of arguments - got (), but expected",
        "missing 1 required positional arguments:",
        "takes no arguments",
        "is only implemented on",
        "missing 1 required positional argument",
        "unfold() missing 3 required positional argument: 'dimension', 'size', 'step'",
        "takes 0 positional arguments but",
        "takes from 1 to 0 positional arguments but",
        "argument after * must be an iterable, not int",
        "must be Number, not Tensor",
        "flatten(): argument 'start_dim' (position 1) must be int, not Tensor",
        "diagonal(): argument 'offset' (position 1) must be int, not Tensor",
        "eig(): argument 'eigenvectors' (position 1) must be bool, not Tensor",
        "(position 1) must be int, not Tensor",
        "received an invalid combination of arguments",
        "pinverse(): argument 'rcond' (position 1) must be float, not Tensor",
        "must be bool, not Tensor",
        "nonzero() takes from 1 to 0 positional arguments but",

    }

    return any(expected_msg in msg for expected_msg in expected_msgs)


def is_expected_value_error(msg: str) -> bool:
    expected_msgs = {"only one element tensors can be converted to Python scalars"}

    return any(expected_msg in msg for expected_msg in expected_msgs)


def is_expected_index_error(msg: str) -> bool:
    expected_msgs = {"Dimension out of range"}

    return any(expected_msg in msg for expected_msg in expected_msgs)


def full_name_with_qualname(klass: type) -> str:
    return f"{klass.__module__}.{klass.__qualname__}"


@pytest.mark.parametrize("tensor_type, op_name, self_tensor, _args", TEST_DATA)
def test_all_allowlisted_tensor_methods_work_remotely_on_all_types(
    tensor_type: str, op_name: str, self_tensor: List, _args: str
) -> None:

    # Step 2: Decide which type we're testing
    t_type = TORCH_STR_DTYPE[tensor_type]

    # Step 3: Create the object we're going to call a method on
    # NOTE: we need a second copy because some methods mutate tensor before we send
    self_tensor, self_tensor_copy = (
        th.tensor(self_tensor, dtype=t_type),
        th.tensor(self_tensor, dtype=t_type),
    )

    # Copy the UID over so that its easier to see they are supposed to be the same obj
    self_tensor_copy.id = self_tensor.id  # type: ignore

    # TODO: This needs to be enforced to something more specific
    args: Any

    # Step 4: Create the arguments we're going to pass the method on
    if _args == "None":
        args = None
    elif _args == "self":
        args = [th.tensor(self_tensor, dtype=t_type)]
    elif _args == "0":
        args = 0
    elif _args == "1":
        args = 1
    elif _args == "[0]":
        args = [th.tensor([0], dtype=t_type)]
    elif _args == "[1]":
        args = [th.tensor([1], dtype=t_type)]
    elif _args == "True":
        args = [True]
    elif _args == "False":
        args = [False]
    else:
        args = _args

    expected_exception: Dict[Type, Callable[[str], bool]] = {
        RuntimeError: is_expected_runtime_error,
        TypeError: is_expected_type_error,
        ValueError: is_expected_value_error,
        IndexError: is_expected_index_error,
    }

    # Step 4: Get the method we're going to call
    target_op_method = getattr(self_tensor, op_name)

    # Step 5: Test to see whether this method and arguments combination is valid
    # in normal PyTorch. If it this is an invalid combination, abort the test
    try:
        # if this is a valid method for this type in torch
        valid_torch_command = True
        if args is not None:
            target_result = target_op_method(*args)
        else:
            target_result = target_op_method()

        if target_result == NotImplemented:
            valid_torch_command = False

    except (RuntimeError, TypeError, ValueError, IndexError) as e:
        msg = repr(e)
        if type(e) in expected_exception and expected_exception[type(e)](msg):
            valid_torch_command = False
        else:
            raise e

    # Step 6: If the command is valid, continue testing
    if valid_torch_command:

        # Step 7: Send our target tensor to alice.
        # NOTE: send the copy we haven't mutated
        xp = self_tensor_copy.send(alice_client)
        if args is not None:
            argsp = [
                arg.send(alice_client) if hasattr(arg, "send") else arg for arg in args
            ]
        else:
            argsp = None  # type:ignore

        # Step 8: get the method on the pointer to alice we want to test
        op_method = getattr(xp, op_name, None)

        # Step 9: make sure the method exists
        assert op_method

        # Step 10: Execute the method remotely
        if argsp is not None:
            result = op_method(*argsp)
        else:
            result = op_method()  # type:ignore

        # Step 11: Ensure the method returned a pointer
        assert isinstance(result, Pointer)

        # Step 12: Get the result
        local_result = result.get()

        # Step 13: If there are NaN values, set them to 0 (this is normal for division by 0)

        try:
            # TODO: We should detect tensor vs primitive in a more reliable way
            # set all NaN to 0
            if isprimitive(value=target_result):
                # check that it matches functionally
                assert local_result == target_result
                # unbox the real value for type comparison below
                local_result = local_result.data
            else:
                # type(target_result) == torch.Tensor

                # Set all NaN to 0
                # If we have two tensors like
                # local = [Nan, 0, 1] and remote = [0, Nan, 1]
                # those are not equal
                nan_mask = local_result.isnan()

                # Use the same mask for local and target
                local_result[nan_mask] = 0
                target_result[nan_mask] = 0

                # Step 14: Ensure we got the same result locally (using normal pytorch) as we did remotely
                # using Syft pointers to communicate with remote torch objects
                assert (local_result == target_result).all()

            # make sure the return types match
            assert type(local_result) == type(target_result)

            # make sure the return type matches the specified allowlist return type
            assert (
                full_name_with_qualname(type(local_result))
                == BASIC_OPS_RETURN_TYPE[op_name]
            )

        except RuntimeError as e:
            msg = repr(e)
            # some types can't set Nans to 0 or do the final check
            if "not implemented for" not in msg:
                raise e

    # TODO: put thought into garbage collection and then
    #  uncoment this.
    # del xp
    #
    # assert len(alice.store.) == 0
