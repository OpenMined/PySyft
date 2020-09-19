"""In this test suite, we code generate a test for the full PyTorch Tensor
method API. We do this by getting the "allowlist" of what we allow  to be
called from the syft package. We then permute this list over all tensor
types and ensure that they can be executed in a remote environment.
"""

# stdlib
from itertools import product
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Type
from typing import Union

# third party
from packaging import version
import pytest
import torch as th

# syft absolute
import syft as sy
from syft.core.pointer.pointer import Pointer
from syft.lib.torch import allowlist
from syft.lib.torch.tensor_util import TORCH_STR_DTYPE

# from syft.lib.python.primitive import isprimitive


TORCH_VERSION = version.parse(th.__version__)


tensor_type = type(th.tensor([1, 2, 3]))

# Currently, we do not have constructors with torch.Tensor
# for dtype in ["complex*", "q*"] (complex and quantized types)
TYPES_EXCEPTIONS_PREFIX = ("complex", "q")

TEST_TYPES = [
    e for e in TORCH_STR_DTYPE.keys() if not e.startswith(TYPES_EXCEPTIONS_PREFIX)
]

# only float types
TEST_TYPES = [ttype for ttype in TEST_TYPES if "float" in ttype]


def get_return_type(support_dict: Union[str, Dict[str, str]]) -> str:
    if isinstance(support_dict, str):
        return support_dict
    else:
        return support_dict["return_type"]


def version_supported(support_dict: Union[str, Dict[str, str]]) -> bool:
    if isinstance(support_dict, str):
        return True
    else:
        return TORCH_VERSION >= version.parse(support_dict["min_version"])


BASIC_OPS = list()
BASIC_OPS_RETURN_TYPE = {}
for method, return_type_name_or_dict in allowlist.items():
    if method.startswith("torch.Tensor."):
        if version_supported(support_dict=return_type_name_or_dict):
            return_type = get_return_type(support_dict=return_type_name_or_dict)
            method_name = method.split(".")[-1]
            BASIC_OPS.append(method_name)
            method_name = method_name.replace("torch.Tensor.", "torch.nn.Parameter.")
            BASIC_OPS_RETURN_TYPE[method_name] = return_type
        else:
            print(f"Skipping torch.{method} not supported in {TORCH_VERSION}")

BASIC_SELF_TENSORS: List[Any] = list()
BASIC_SELF_TENSORS.append([-1, 0, 1.0, 2, 3, 4])  # with a 0
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

# BASIC_METHOD_ARGS.append("True") #TODO: can't call .serialize() on bool
# BASIC_METHOD_ARGS.append("False") #TODO: can't call .serialize() on bool

TEST_DATA = list(product(TEST_TYPES, BASIC_OPS, BASIC_SELF_TENSORS, BASIC_METHOD_ARGS))

# Step 1: Create remote worker
alice = sy.VirtualMachine(name="alice")
alice_client = alice.get_client()


def is_expected_runtime_error(msg: str) -> bool:
    expected_msgs = {
        "not implemented for",
        "not supported on",
        "expected a tensor with 2 or more dimensions of floating types",
        "Expected object of scalar type Long but got scalar type",
        "expected total dims >= 2, but got total dims = 1",
        "invalid argument 1: A should be 2 dimensional at",
        "invalid argument 1: expected a matrix at",
        "arguments don't support automatic differentiation, but one of the arguments requires grad",
        "cannot resize variables that require grad",
        "inconsistent tensor size, expected tensor",
        "size mismatch",
        "1D tensors expected, got 2D",
        "requested resize to",
        "ger: Expected 1-D ",
        "At least one of 'min' or 'max' must not be None",
        "Boolean value of Tensor with more than one value is ambiguous",
        "shape '[1]' is invalid for input of size",
        "a leaf Variable that requires grad has been used in an in-place operation",
        "a leaf Variable that requires grad is being used in an in-place operation",
        "the derivative for 'other' is not implemented",
        "is only supported for integer type tensors",
        "INTERNAL ASSERT FAILED",
        "vector and vector expected, got",  # torch==1.4.0 "ger"
        "cbitor is only supported for integer type tensors",  # torch==1.4.0 "__or__"
        "cbitand is only supported for integer type tensors",  # torch==1.4.0 "__and__"
    }

    return any(expected_msg in msg for expected_msg in expected_msgs)


def is_expected_type_error(msg: str) -> bool:
    expected_msgs = {
        "received an invalid combination of arguments - got (), but expected",
        "takes no arguments",
        "takes 0 positional arguments but 1 was given",
        "takes from 1 to 0 positional arguments but",
        "argument after * must be an iterable",
        "missing 1 required positional argument",
        "missing 2 required positional argument",
        "(operator.invert) is only implemented on integer and Boolean-type tensors",
        "diagonal(): argument 'offset' (position 1) must be int",
        "argument 'min' (position 1) must be Number",
        "argument 'max' (position 1) must be Number",
        "argument 'diagonal' (position 1) must be int",
        "argument 'dim' (position 1) must be int",
        "argument 'dim0' (position 1) must be int",
        "argument 'start_dim' (position 1) must be int",
        "argument 'k' (position 1) must be int",
        "argument 'rcond' (position 1) must be float",
        "argument 'p' (position 2) must be Number",
        "argument 'sorted' must be bool",
        "argument 'return_inverse' must be bool",
        "received an invalid combination of arguments",
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
def test_all_allowlisted_parameter_methods_work_remotely_on_all_types(
    tensor_type: str, op_name: str, self_tensor: List, _args: str
) -> None:

    # Step 2: Decide which type we're testing
    t_type = TORCH_STR_DTYPE[tensor_type]

    # Step 3: Create the object we're going to call a method on
    # NOTE: we need a second copy because some methods mutate tensor before we send
    self_tensor, self_tensor_copy = (
        th.nn.Parameter(th.tensor(self_tensor, dtype=t_type)),
        th.nn.Parameter(th.tensor(self_tensor, dtype=t_type)),
    )

    # Copy the UID over so that its easier to see they are supposed to be the same obj
    self_tensor_copy.id = self_tensor.id  # type: ignore

    # TODO: This needs to be enforced to something more specific
    args: Any

    # Step 4: Create the arguments we're going to pass the method on
    if _args == "None":
        args = None
    elif _args == "self":
        args = [th.nn.Parameter(self_tensor)]
    elif _args == "0":
        args = 0
    elif _args == "1":
        args = 1
    elif _args == "[0]":
        args = [th.nn.Parameter(th.tensor([0], dtype=t_type))]
    elif _args == "[1]":
        args = [th.nn.Parameter(th.tensor([1], dtype=t_type))]
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
            print(msg)
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
            # if isprimitive(value=target_result):
            #     # check that it matches functionally
            #     assert local_result == target_result
            #     # unbox the real value for type comparison below
            #     local_result = local_result.data
            # else:

            # type(target_result) == torch.Tensor

            # Set all NaN to 0
            # If we have two tensors like
            # local = [Nan, 0, 1] and remote = [0, Nan, 1]
            # those are not equal
            # Tensor.isnan was added in torch 1.6
            # so we need to do torch.isnan(tensor)
            nan_mask = th.isnan(local_result)

            # Use the same mask for local and target
            local_result[nan_mask] = 0
            target_result[nan_mask] = 0

            # Step 14: Ensure we got the same result locally (using normal pytorch)
            # as we did remotely using Syft pointers to communicate with remote torch objects
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
            if (
                "not implemented for" not in msg
                and "a leaf Variable that requires grad" not in msg  # TODO why?
            ):
                raise e

    # TODO: put thought into garbage collection and then
    #  uncoment this.
    # del xp
    #
    # assert len(alice.store.) == 0
