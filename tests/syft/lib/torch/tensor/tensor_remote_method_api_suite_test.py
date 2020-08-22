"""In this test suite, we code generate a test for the full PyTorch Tensor
method API. We do this by getting the "allowlist" of what we allow  to be
called from the syft package. We then permute this list over all tensor
types and ensure that they can be executed in a remote environment.
"""

from syft.lib.torch import allowlist
from syft.core.pointer.pointer import Pointer
from syft.lib.torch.tensor_util import TORCH_STR_DTYPE

import syft as sy
import torch as th
import pytest
from itertools import product
from typing import List
from typing import Any

tensor_type = type(th.tensor([1, 2, 3]))

# Currently, we do not have constructors with torch.Tensor
# for dtype in ["complex*", "q*"] (complex and quantized types)
TYPES_EXCEPTIONS_PREFIX = ("complex", "q")

TEST_TYPES = [
    e for e in TORCH_STR_DTYPE.keys() if not e.startswith(TYPES_EXCEPTIONS_PREFIX)
]

BASIC_OPS = list()
for method in allowlist.keys():
    if "torch.Tensor." in method:
        BASIC_OPS.append(method.split(".")[-1])

BASIC_SELF_TENSORS: List[Any] = list()
BASIC_SELF_TENSORS.append([-1, 0, 1, 2, 3, 4])  # with a 0
# BASIC_INPUT_TENSORS.append([-1, 1, 2, 3, 4]) # without a 0
# BASIC_INPUT_TENSORS.append([-0.1, 0.1, 0.2, 0.3, 0.4]) # with a decimal
BASIC_SELF_TENSORS.append([[-0.1, 0.1], [0.2, 0.3]])  # square

BASIC_METHOD_ARGS = list()
BASIC_METHOD_ARGS.append("self")
BASIC_METHOD_ARGS.append("None")
# BASIC_METHOD_ARGS.append("0") #TODO: add ints as remote argument types (can't call .serialize() on int)
# BASIC_METHOD_ARGS.append("1") #TODO: add ints as remote argument types (can't call .serialize() on int)
# BASIC_METHOD_ARGS.append("True") #TODO: add ints as remote argument types (can't call .serialize() on bool)
# BASIC_METHOD_ARGS.append("False") #TODO: add ints as remote argument types (can't call .serialize() on bool)


TEST_DATA = list(product(TEST_TYPES, BASIC_OPS, BASIC_SELF_TENSORS, BASIC_METHOD_ARGS))

# Step 1: Create remote worker
alice = sy.VirtualMachine(name="alice")
alice_client = alice.get_client()


@pytest.mark.parametrize("tensor_type, op_name, self, _args", TEST_DATA)
def test_all_allowlisted_tensor_methods_work_remotely_on_all_types(
    tensor_type: str, op_name: str, self: List, _args: str
) -> None:

    # Step 2: Decide which type we're testing
    t_type = TORCH_STR_DTYPE[tensor_type]

    # Step 3: Create the object we're going to call a method on
    self = th.tensor(self, dtype=t_type)

    # Step 4: Create the arguments we're going to pass the method on
    if _args == "None":
        args = None
    elif _args == "self":
        args = [th.tensor(self, dtype=t_type)]
    elif _args == "0":
        args = [0]
    elif _args == "True":
        args = [True]
    elif _args == "False":
        args = [False]

    # Step 4: Get the method we're going to call
    target_op_method = getattr(self, op_name)

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

    except RuntimeError as e:

        msg = repr(e)

        if """not implemented for""" in msg:
            valid_torch_command = False
        elif """two bool tensors is not supported.""" in msg:
            valid_torch_command = False
        elif msg == """RuntimeError('ZeroDivisionError')""":
            valid_torch_command = False
        elif """not supported on""" in msg:
            valid_torch_command = False
        elif """is not supported""" in msg:
            valid_torch_command = False
        elif "At least one of" in msg and " must not be None" in msg:
            valid_torch_command = False
        elif "1D tensors expected, got 2D, 2D tensors at" in msg:
            valid_torch_command = False
        elif "RuntimeError('ger: Expected 1-D argument self, but got 2-D')" in msg:
            valid_torch_command = False
        elif "RuntimeError('Can only calculate the mean of floating types." in msg:
            valid_torch_command = False
        elif "expected a tensor with 2 or more dimensions of floating types" in msg:
            valid_torch_command = False
        elif "only supports floating-point dtypes" in msg:
            valid_torch_command = False
        elif "invalid argument 1: A should be 2 dimensional at" in msg:
            valid_torch_command = False
        elif "invalid argument 1: expected a matrix at" in msg:
            valid_torch_command = False
        elif "Expected object of scalar type Long but got scalar type" in msg:
            valid_torch_command = False
        elif "RuntimeError('expected total dims >= 2, but got total dims = 1')" == msg:
            valid_torch_command = False
        elif "Integer division of tensors using div or / is no longer supported" in msg:
            valid_torch_command = False
        else:
            print(msg)
            raise e

    except TypeError as e:
        msg = repr(e)

        if "received an invalid combination of arguments - got (), but expected" in msg:
            valid_torch_command = False
        elif "missing 1 required positional arguments:" in msg:
            valid_torch_command = False
        elif "takes no arguments" in msg:
            valid_torch_command = False
        elif "is only implemented on" in msg:
            valid_torch_command = False
        elif "missing 1 required positional argument" in msg:
            valid_torch_command = False
        elif "takes 0 positional arguments but" in msg:
            valid_torch_command = False
        elif "argument after * must be an iterable, not int" in msg:
            valid_torch_command = False
        elif "must be Number, not Tensor" in msg:
            valid_torch_command = False
        elif (
            """TypeError("flatten(): argument 'start_dim' (position 1) must be int, not Tensor")"""
            == msg
        ):
            valid_torch_command = False
        elif (
            """TypeError("diagonal(): argument 'offset' (position 1) must be int, not Tensor")"""
            == msg
        ):
            valid_torch_command = False
        elif (
            """TypeError("eig(): argument 'eigenvectors' (position 1) must be bool, not Tensor")"""
            == msg
        ):
            valid_torch_command = False
        elif """(position 1) must be int, not Tensor")""" in msg:
            valid_torch_command = False
        elif "received an invalid combination of arguments" in msg:
            valid_torch_command = False
        elif (
            """TypeError("pinverse(): argument 'rcond' (position 1) must be float, not Tensor")"""
            == msg
        ):
            valid_torch_command = False
        elif """must be bool, not Tensor""" in msg:
            valid_torch_command = False
        else:
            print(msg)
            raise e

    except ValueError as e:
        msg = repr(e)
        if (
            msg
            == "ValueError('only one element tensors can be converted to Python scalars')"
        ):
            valid_torch_command = False
        else:
            print(msg)
            raise e

    except IndexError as e:
        msg = repr(e)
        if "Dimension out of range" in msg:
            valid_torch_command = False
        else:
            print(msg)
            raise e

    # Step 6: If the command is valid, continue testing
    if valid_torch_command:

        # Step 7: Send our target tensor to alice.
        xp = self.send(alice_client)  # type:ignore
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
            # set all NaN to 0
            local_result[local_result != local_result] = 0
            target_result[target_result != target_result] = 0

            # Step 14: Ensure we got the same result locally (using normal pytorch) as we did remotely
            # using Syft pointers to communicate with remote torch objects
            assert (local_result == target_result).all()

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
