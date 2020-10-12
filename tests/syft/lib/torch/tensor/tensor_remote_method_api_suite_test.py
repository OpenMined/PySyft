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
from syft.lib.python.primitive_factory import PrimitiveFactory
from syft.lib.python.primitive_factory import isprimitive
from syft.lib.python.primitive_interface import PyPrimitive
from syft.lib.torch import allowlist
from syft.lib.torch.tensor_util import TORCH_STR_DTYPE
from syft.lib.util import full_name_with_qualname

TORCH_VERSION = version.parse(th.__version__)


tensor_type = type(th.tensor([1, 2, 3]))

# Currently, we do not have constructors with torch.Tensor
# for dtype in ["complex*", "q*"] (complex and quantized types)
TYPES_EXCEPTIONS_PREFIX = ("complex", "q")

TEST_TYPES = [
    e for e in TORCH_STR_DTYPE.keys() if not e.startswith(TYPES_EXCEPTIONS_PREFIX)
]


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


SKIP_METHODS = ["__getitem__", "max"]

BASIC_OPS = list()
BASIC_OPS_RETURN_TYPE = {}
for method, return_type_name_or_dict in allowlist.items():
    if "torch.Tensor." in method:
        if version_supported(support_dict=return_type_name_or_dict):
            return_type = get_return_type(support_dict=return_type_name_or_dict)
            method_name = method.split(".")[-1]
            # some ops cant be tested with the current suite
            if method_name in SKIP_METHODS:
                continue
            BASIC_OPS.append(method_name)
            BASIC_OPS_RETURN_TYPE[method_name] = return_type
        else:
            print(f"Skipping torch.{method} not supported in {TORCH_VERSION}")

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
BASIC_METHOD_ARGS.append("True")
BASIC_METHOD_ARGS.append("False")

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
        # py3.6 torch==1.6.0 "square" space typo "Longcan't" is correct
        "result type Longcan't be cast to the desired output type Bool",
        "result type Long can't be cast to the desired output type Bool",
        "inconsistent tensor size, expected tensor",
        "size mismatch",
        "1D tensors expected, got 2D",
        "requested resize to",
        "ger: Expected 1-D ",
        "At least one of 'min' or 'max' must not be None",
        "Boolean value of Tensor with more than one value is ambiguous",
        "shape '[1]' is invalid for input of size",  # BASIC_METHOD_ARGS.append("[1]")
        "Negation, the `-` operator, on a bool tensor is not supported",
        "True division requires a floating output type, but got",
        "vector and vector expected, got",  # torch==1.4.0 "ger"
        "bitor is only supported for integer type tensors",  # torch==1.4.0 "__or__"
        "bitand is only supported for integer type tensors",  # torch==1.4.0 "__and__"
        "only Tensors of floating point dtype can require gradients",
        "std and var only support floating-point dtypes",
        "Subtraction, the `-` operator, with a bool tensor is not supported",
        "a leaf Variable that requires grad is being used in an in-place operation",
        "expects norm to be integer or float",
        "Mismatch in shape: grad_output",
        "element 0 of tensors does not require",
        "grad can be implicitly created only for scalar outputs",
    }

    return any(expected_msg in msg for expected_msg in expected_msgs)


def is_expected_type_error(msg: str) -> bool:
    expected_msgs = {
        "received an invalid combination of arguments - got (), but expected",
        "missing 1 required positional argument",
        "takes no arguments",
        "is only implemented on",
        "takes 0 positional arguments but",
        "takes from 1 to 0 positional arguments but",
        "argument after * must be an iterable, not int",
        "must be Number, not Tensor",
        "diagonal(): argument 'offset' (position 1) must be int",
        "eig(): argument 'eigenvectors' (position 1) must be bool",
        "(position 1) must be Tensor, not",
        "(position 1) must be int, not Tensor",
        "received an invalid combination of arguments",
        "must be bool, not Tensor",
        "nonzero() takes from 1 to 0 positional arguments but",
        "transpose_() missing 2 required positional argument",  # "transpose_"
        "(position 1) must be bool",
        "argument 'return_inverse' must be bool",
        "__bool__ should return bool, returned Bool",
        "argument 'sorted' must be bool, not",
        "got NotImplementedType instead",
        "argument 'min' (position 1) must be Number",
        "argument 'max' (position 1) must be Number",
        "argument 'diagonal' (position 1) must be int",
        "argument 'dim' (position 1) must be int",
        "argument 'dim0' (position 1) must be int",
        "argument 'start_dim' (position 1) must be int",
        "argument 'k' (position 1) must be int",
        "argument 'rcond' (position 1) must be float",
        "argument 'p' (position 2) must be Number",
        "object is not iterable",
    }

    return any(expected_msg in msg for expected_msg in expected_msgs)


def is_expected_value_error(msg: str) -> bool:
    expected_msgs = {"only one element tensors can be converted to Python scalars"}

    return any(expected_msg in msg for expected_msg in expected_msgs)


def is_expected_index_error(msg: str) -> bool:
    expected_msgs = {"Dimension out of range"}

    return any(expected_msg in msg for expected_msg in expected_msgs)


@pytest.mark.slow
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

    # we dont have .id's by default anymore
    # self_tensor_copy.id = self_tensor.id  # type: ignore

    args: List[Any] = []

    # Step 4: Create the arguments we're going to pass the method on
    if _args == "None":
        args = []
    elif _args == "self":
        args = [th.tensor(self_tensor, dtype=t_type)]
    elif _args == "0":
        args = [PrimitiveFactory.generate_primitive(value=0)]
    elif _args == "1":
        args = [PrimitiveFactory.generate_primitive(value=1)]
    elif _args == "[0]":
        args = [th.tensor([0], dtype=t_type)]
    elif _args == "[1]":
        args = [th.tensor([1], dtype=t_type)]
    elif _args == "True":
        args = [PrimitiveFactory.generate_primitive(value=True)]
    elif _args == "False":
        args = [PrimitiveFactory.generate_primitive(value=False)]
    else:
        args = [_args]

    try:
        _ = len(args)
    except Exception as e:
        err = f"Args must be iterable so it can be spread with * into the method. {e}"
        raise Exception(err)

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
        is_property = False
        # if this is a valid method for this type in torch
        valid_torch_command = True
        if type(target_op_method).__name__ in ["builtin_function_or_method", "method"]:
            # this prevents things like syft.lib.python.bool.Bool from getting treated
            # as an Int locally but then failing on the upcast to builtins.bool on
            # the remote side
            upcasted_args = []
            for arg in args:
                upcast_attr = getattr(arg, "upcast", None)
                if upcast_attr is not None:
                    upcasted_args.append(upcast_attr())
                else:
                    upcasted_args.append(arg)
            target_result = target_op_method(*upcasted_args)

            if target_result == NotImplemented:
                valid_torch_command = False
        else:
            # we have a property and already have its value
            is_property = True
            target_result = target_op_method

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
        argsp: List[Any] = []
        if len(args) > 0 and not is_property:
            argsp = [
                arg.send(alice_client) if hasattr(arg, "send") else arg for arg in args
            ]

        # Step 8: get the method on the pointer to alice we want to test
        op_method = getattr(xp, op_name, None)

        # Step 9: make sure the method exists
        assert op_method is not None

        # Step 10: Execute the method remotely
        if is_property:
            # we already have the result
            result = op_method
        else:
            result = op_method(*argsp)

        # Step 11: Ensure the method returned a pointer
        assert isinstance(result, Pointer)

        # Step 12: Get the result
        local_result = result.get()

        # Step 13: If there are NaN values, set them to 0 (this is normal for division by 0)

        try:
            # only do single value comparisons, do lists, tuples etc below in the else
            if not hasattr(target_result, "__len__") and (
                isprimitive(value=target_result)
                or issubclass(type(local_result), PyPrimitive)
            ):
                # check that it matches functionally
                assert local_result == target_result

                # convert target_result for type comparison below
                target_result = PrimitiveFactory.generate_primitive(value=target_result)
            else:
                if issubclass(type(target_result), tuple) or issubclass(
                    type(local_result), tuple
                ):
                    target_result = list(target_result)
                    local_result = list(local_result)

                # Set all NaN to 0
                # If we have two tensors like
                # local = [Nan, 0, 1] and remote = [0, Nan, 1]
                # those are not equal
                # Tensor.isnan was added in torch 1.6
                # so we need to do torch.isnan(tensor)

                # to handle tuple return types for now we just make sure everything
                # is in a list of at least 1 size and then iterate over it
                if type(local_result) is not list:
                    local_result = [local_result]
                    target_result = [target_result]

                for i, local_item in enumerate(local_result):
                    target_item = target_result[i]

                    nan_mask = th.isnan(local_item)

                    # Use the same mask for local and target
                    local_item[nan_mask] = 0
                    target_item[nan_mask] = 0

                    # Step 14: Ensure we got the same result locally (using normal pytorch) as we did remotely
                    # using Syft pointers to communicate with remote torch objects
                    assert (local_item == target_item).all()

            # make sure the return types match
            assert type(local_result) == type(target_result)

            # TODO: Fix this workaround for types that sometimes return Tensor tuples
            # we are getting back more than 1 return type so we need to fake it until
            # we add Union to the return types
            if type(local_result) is list:
                local_result = local_result[0]

            # make sure the return type matches the specified allowlist return type
            local_type = full_name_with_qualname(klass=type(local_result))
            expected_type = BASIC_OPS_RETURN_TYPE[op_name]
            python_types = "syft.lib.python"
            if local_type.startswith(python_types) and expected_type.startswith(
                python_types
            ):
                # python types seem to resolve as both int.Int and .Int causing issues
                # in the match
                assert local_type.split(".")[-1] == expected_type.split(".")[-1]

            else:
                assert local_type == expected_type

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
