"""In this test suite, we load the allowlist.json and run through all the tests based
on what expected inputs and return types are provided by the json file.
"""

# stdlib
from itertools import product
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List as ListType
from typing import Union

# third party
from packaging import version
import pytest
from pytest import approx
import torch as th

# syft absolute
import syft as sy
from syft.core.pointer.pointer import Pointer
from syft.lib.python import List
from syft.lib.python import String
from syft.lib.python.primitive_factory import PrimitiveFactory
from syft.lib.python.primitive_factory import isprimitive
from syft.lib.python.primitive_interface import PyPrimitive
from syft.lib.torch import allowlist
from syft.lib.torch.return_types import types_fields
from syft.lib.torch.tensor_util import TORCH_STR_DTYPE
from syft.lib.util import full_name_with_qualname

TORCH_VERSION = version.parse(th.__version__.split("+")[0])
py_ver = sys.version_info
PYTHON_VERSION = version.parse(f"{py_ver.major}.{py_ver.minor}")
OS_NAME = platform.system().lower()


# Allow List Test Generation
#
# The purpose of this code is to gather all of the explicitly defined classes,
# attributes, methods and properties that are can be executed remotely via a pointer.
# The vast majority of types and inputs are tensors or collections of numeric values
# so for many cases simply enumerating all possible supported numeric data types
# and using them as both Tensors and inputs covers a lot of the functionality.
# However there are instances where specific methods or properties either do not
# accept input, have specific input requirements or kwargs, or give a variety of output
# types. Additionally some methods exist or behave differently depending on the version
# of the library that we are testing. To support all of this added complexity we start
# with the simplest version which is the "attribute path" to "return type" mapping aka:
#
# allowlist["torch.Tensor.__add__"] = "torch.Tensor"
#
# We then check to see if a explicit configuration is set for the attribute path in
# allowlist_test.json and if so we use the configuration from there.
# To keep the custom definitions DRY we use a concept called "profiles" which are just
# a key lookup in the main JSON tree. Many configuration options also support the
# concept of a key lookup for example you will see lots of "all" which is used to map
# to a whole pre-defined set of supported data types, or inputs. By utilising this
# approach we can prevent the need to maintain a large list of expected exceptions
# which increases the likelihood tests are skipped by accident and makes it very hard
# to track exactly which data types and inputs are supported for a given attribute.
#
# Once we understand what can be tested we construct a product or combination of all
# possible options using the following required variables for each test run:
#
# tensor_type, op_name, self_tensor, _args, is_property, return_type
#
# tensor_type:  this is the data type we convert our test tensor to e.g. uint8
# op_name:      this is the method or property we are testing e.g. __add__
# self_tensor:  this is the tensor we are operating when doing the test
# _args:        this is the input used on the method where "self" represents the
#               self_tensor and None represents no input
# is_property:  this helps understand if the attribute can be called or not
# return_type:  this helps verify that the return type matches the expected return type

# this handles instances where the allow list provides more meta information
def get_return_type(support_dict: Union[str, Dict[str, str]]) -> str:
    if isinstance(support_dict, str):
        return support_dict
    else:
        return support_dict["return_type"]


# this allows us to skip tests based on a few loose matching rules for example:
#
# "skip": [{ "data_types": ["bool"], "inputs": [0, 1] }]
#
# This rule set will match on any test which is for the data type bool and has the
# inputs 0 or 1. Since the primary variation of test combinations relates to their
# tensors, inputs and the data types associated with them
def check_skip(
    combination: ListType,
    skip_rule: Dict[str, Any],
    lib_version: Union[version.Version, version.LegacyVersion],
) -> bool:
    combination_dict = {
        "data_types": combination[0],
        "tensors": combination[2],
        "inputs": combination[3],
    }

    # we will assume the skip rule applies and try to prove it doesn't by finding a
    # condition which we cannot satisfy
    applies = True
    for key in skip_rule.keys():
        # handle the version checks later
        if key.endswith("_version"):
            continue
        if key in combination_dict:
            # first check it is not equal since then the rule would apply
            if combination_dict[key] != skip_rule[key]:
                # second check if the rule is a list of rules in which case we
                # need to check for the presence of our current combination value
                if isinstance(skip_rule[key], Iterable):
                    # we cant use x not in y because bool matches for 0 and 1 in lists
                    presence = any(
                        combination_dict[key] == item
                        and type(combination_dict[key]) == type(item)
                        for item in skip_rule[key]
                    )

                    # if we cant find a matching item of value and type then we dont
                    # match and we wont skip this test, if its True we match so far and
                    # can keep checking more rules
                    applies = presence
                    if not applies:
                        return applies

                else:
                    # we dont match and the rule is not a list of rules so we wont skip
                    # and can break
                    applies = False
                    return applies

    # skip rule versioning example:
    # "skip": [{ "data_types": ["float16"], "lte_version": "1.5.1" }]
    # this says skip float16 if the library is less than or equal to 1.5.1
    # say we are running 1.5.0 that means
    # torch==1.5.0 <= 1.5.1 so this rule applies and we could possibly need to skip

    # if there is a lte_version for the skip rule and our lib is NOT lte, less than or
    # equal to the version then we cant skip because the rule only prohibits lte
    if "lte_version" in skip_rule and (
        not lib_version <= version.parse(skip_rule["lte_version"])
    ):
        applies = False

    # if there is a gte_version for the skip rule and our lib is NOT gte, greater than
    # or equal to the version then  we cant skip because the rule only prohibits gte
    if "gte_version" in skip_rule and (
        not lib_version >= version.parse(skip_rule["gte_version"])
    ):
        applies = False

    # if we were unable to fail the skip rules we should have applies = True and will
    # skip this particular test combination
    return applies


BASIC_OPS = list()
BASIC_OPS_RETURN_TYPE = {}
# here we are loading up the true allowlist which means that we are only testing what
# can be used by the end user
for method, return_type_name_or_dict in allowlist.items():
    if method.startswith("torch.Tensor."):
        return_type = get_return_type(support_dict=return_type_name_or_dict)
        if return_type == "unknown":
            return_type = "torch.Tensor"
        method_name = method.split(".")[-1]
        BASIC_OPS.append(method_name)
        BASIC_OPS_RETURN_TYPE[method_name] = return_type

# load our custom configurations
try:
    with open(__file__.replace(".py", ".json"), "r") as f:
        TEST_JSON = json.loads(f.read())
except Exception as e:
    print(f"Exception {e} triggered")
    raise e

# we need a file to keep all the errors in that makes it easy to debug failures
TARGET_PLATFORM = f"{PYTHON_VERSION}_{TORCH_VERSION}_{OS_NAME}"
ERROR_FILE_PATH = os.path.abspath(
    Path(__file__) / "../../../.." / f"allowlist_test_errors_{TARGET_PLATFORM}.jsonl"
)

# we need a file to keep track of all the methods that are skipped or succeed
SUPPORT_FILE_PATH = os.path.abspath(
    Path(__file__) / "../../../.." / f"allowlist_test_support_{TARGET_PLATFORM}.jsonl"
)


# clear the file before running the tests
if os.path.exists(ERROR_FILE_PATH):
    # this one we can delete since we dont start writing until we are into the tests
    try:
        os.unlink(ERROR_FILE_PATH)
    except Exception as e:
        print(f"Exception {e} triggered")


# we are running many works in parallel and theres a race condition with deleting this
# file and then writing to it during the collection phase so we are going to just
# spread out the workers and only delete if the file isn't brand new
time.sleep(random.random() * 2)

if os.path.exists(SUPPORT_FILE_PATH):
    # we need to write during gathering so we need to delete this carefully
    try:
        file_stat = os.stat(SUPPORT_FILE_PATH)
        diff = time.time() - file_stat.st_mtime
        if diff > 0.1:
            # only delete on the first run
            for retry in range(5):
                try:
                    os.unlink(SUPPORT_FILE_PATH)
                    break
                except BaseException:
                    time.sleep(1)
    except Exception:
        print(f"Failed while trying to os.stat file {SUPPORT_FILE_PATH}")


# write test debug info to make it easy to debug long running tests with large output
def write_error_debug(debug_data: Dict[str, Any]) -> None:
    # save a file in the root project dir
    try:
        with open(ERROR_FILE_PATH, "a+") as f:
            f.write(f"{json.dumps(debug_data, default=str)}\n")
    except Exception as e:
        print(f"Exception {e} triggered")


# write the result of support for the test for creating a comprehensive report
def write_support_result(test_details: Dict[str, Any]) -> None:
    # save a file in the root project dir
    try:
        with open(SUPPORT_FILE_PATH, "a+") as f:
            f.write(f"{json.dumps(test_details, default=str)}\n")
    except Exception as e:
        print(f"Exception {e} triggered")


TEST_DATA = []
# we iterate over the allowlist and only override where explicitly defined
for op in BASIC_OPS:
    skip = []
    not_available = []
    deterministic = True
    if op not in TEST_JSON["tests"]["torch.Tensor"]:
        # there is no custom configuration so we will test all supported combinations
        dtypes = ["common"]
        test_tensors = ["tensor1", "tensor2"]
        test_inputs = ["all"]
        is_property = False
        return_type = BASIC_OPS_RETURN_TYPE[op]
        # continue  # dont try testing without testing rules
    else:
        meta = {}
        # if there is a profile use the settings from that first
        if "profile" in TEST_JSON["tests"]["torch.Tensor"][op]:
            profile = TEST_JSON["tests"]["torch.Tensor"][op]["profile"]
            meta.update(TEST_JSON["profiles"][profile])

        # now update with what ever extra settings are overridden on the method
        meta.update(TEST_JSON["tests"]["torch.Tensor"][op])
        dtypes = meta["data_types"]
        test_tensors = meta["tensors"]
        test_inputs = meta["inputs"]
        is_property = meta["property"]
        return_type = meta["return_type"]

        # grab skip rules
        if "skip" in meta:
            skip += meta["skip"]

        # grab not_available rules
        if "not_available" in meta:
            not_available += meta["not_available"]

        if "deterministic" in meta:
            deterministic = meta["deterministic"]

        # this is the minimum version of the library required to run this test
        # which should match the values in the actual allowlist.py
        if "min_version" in meta and TORCH_VERSION < version.parse(meta["min_version"]):
            # skip attributes which are not supported in the TORCH_VERSION
            continue

    # gather all the data types we want to test
    data_types = []
    for dtype in dtypes:
        if issubclass(type(dtype), str) and dtype in TEST_JSON["data_types"]:
            data_types += TEST_JSON["data_types"][dtype]
        else:
            data_types.append(dtype)

    # collect the tensors we want to test
    tensors = []
    for tensor in test_tensors:
        if issubclass(type(tensor), str) and tensor in TEST_JSON["tensors"]:
            tensors += TEST_JSON["tensors"][tensor]
        else:
            tensors.append(tensor)

    # get the list of inputs
    inputs = []
    for input in test_inputs:
        if issubclass(type(input), str) and input in TEST_JSON["inputs"]:
            inputs += TEST_JSON["inputs"][input]
        elif issubclass(type(input), dict):
            resolved_args = {}
            for k, v in input.items():  # type: ignore
                if issubclass(type(v), str) and v in TEST_JSON["inputs"]:
                    resolved_args[k] = TEST_JSON["inputs"][v]
                else:
                    resolved_args[k] = v
            inputs.append(resolved_args)
        else:
            inputs.append(input)

    combinations = list(
        product(
            data_types,
            [op],
            tensors,
            inputs,
            [is_property],
            [return_type],
            [deterministic],
        )
    )

    skipped_combinations = set()
    not_available_combinations = set()
    for combination in combinations:
        # skips are temporary
        for skip_rule in skip:
            if check_skip(
                combination=list(combination),
                skip_rule=skip_rule,
                lib_version=TORCH_VERSION,
            ):
                # we use str(combination) so that we can hash the entire combination
                skipped_combinations.add(str(combination))

        # not available are features we cant or wont test because they arent supported
        for na_rule in not_available:
            if check_skip(
                combination=list(combination),
                skip_rule=na_rule,
                lib_version=TORCH_VERSION,
            ):
                # we use str(combination) so that we can hash the entire combination
                not_available_combinations.add(str(combination))

    for combination in combinations:
        # we need to record the support for this combination, the key will be unique
        # and easy to match multiple entries of the same combination
        support_data = {}
        support_data["tensor_type"] = combination[0]
        support_data["op_name"] = combination[1]

        # we use str so that we can hash the entire combination with nested lists
        if (
            str(combination) not in skipped_combinations
            and str(combination) not in not_available_combinations
        ):
            TEST_DATA.append(combination)
        elif str(combination) in not_available_combinations:
            support_data["status"] = "not_available"
            write_support_result(support_data)
        elif str(combination) in skipped_combinations:
            support_data["status"] = "skip"
            write_support_result(support_data)


# if the environment variables below are set bigger than 1 we will split the TEST_DATA
# into parts so that these can be parallelized by different test runners or containers
# TEST_CHUNK = int(os.getenv("TEST_CHUNK", 1))
# TEST_CHUNKS = int(os.getenv("TEST_CHUNKS", 1))

# # chunk the tests
# if TEST_CHUNKS > 1:
#     chunk_size = math.ceil(len(TEST_DATA) / TEST_CHUNKS)
#     start_offset = (TEST_CHUNK - 1) * chunk_size
#     end_offset = start_offset + chunk_size
#     TEST_DATA = TEST_DATA[start_offset:end_offset]


@pytest.mark.torch
@pytest.mark.parametrize(
    "tensor_type, op_name, self_tensor, _args, is_property, return_type, deterministic",
    TEST_DATA,
)
def test_all_allowlisted_tensor_methods(
    tensor_type: str,
    op_name: str,
    self_tensor: ListType,
    _args: Union[str, ListType, Dict, bool, None],
    is_property: bool,
    return_type: str,
    deterministic: bool,
    client: sy.VirtualMachineClient,
) -> None:

    support_data = {}
    support_data["tensor_type"] = tensor_type
    support_data["op_name"] = op_name

    # this is used in bulk error reporting
    debug_data: Dict[str, Any] = {}
    debug_data["input"] = {
        "tensor_type": tensor_type,
        "op_name": op_name,
        "self_tensor": self_tensor,
        "_args": _args,
        "is_property": is_property,
        "return_type": return_type,
        "deterministic": deterministic,
    }

    try:
        # Step 2: Decide which type we're testing
        t_type = TORCH_STR_DTYPE[tensor_type]

        # Step 3: Create the object we're going to call a method on
        # NOTE: we need a second copy because some methods mutate tensor before we send
        requires_grad = False
        if op_name in ["backward", "retain_grad", "grad"]:
            requires_grad = True
        self_tensor, self_tensor_copy = (
            th.tensor(self_tensor, dtype=t_type, requires_grad=requires_grad),
            th.tensor(self_tensor, dtype=t_type, requires_grad=requires_grad),
        )

        # we dont have .id's by default anymore
        # self_tensor_copy.id = self_tensor.id  # type: ignore

        args: Union[ListType[Any], Dict[str, Any]]

        # Step 4: Create the arguments we're going to pass the method on
        if _args is None:
            args = []
        elif _args == "self":
            args = [th.tensor(self_tensor, dtype=t_type, requires_grad=requires_grad)]
        elif isinstance(_args, list):
            args = [th.tensor(_args, dtype=t_type)]
        elif isinstance(_args, dict):
            args = {}
            tuple_args = False
            if len(_args) > 0 and "ALL_TUPLE_ARGS_" in list(_args.keys())[0]:
                tuple_args = True
            for k, v in _args.items():
                arg_type = t_type
                real_k = k
                if isinstance(v, list):
                    if "_DTYPE_" in real_k:
                        parts = real_k.split("_DTYPE_")
                        real_k = parts[0]
                        v_dtype_attr = parts[1]
                        v_dtype = getattr(th, v_dtype_attr, None)
                        if v_dtype is not None:
                            arg_type = v_dtype

                    if real_k.startswith("LIST_"):
                        # use a normal list
                        real_k = real_k.replace("LIST_", "")
                        if real_k.startswith("TENSOR_"):
                            real_k = real_k.replace("TENSOR_", "")
                            args[real_k] = [th.tensor(t, dtype=arg_type) for t in v]
                        else:
                            args[real_k] = v
                    elif real_k.startswith("0d_"):
                        # make a 0d tensor
                        real_k = real_k.replace("0d_", "")
                        args[real_k] = th.tensor(v[0], dtype=arg_type)
                    else:
                        args[real_k] = th.tensor(v, dtype=arg_type)
                elif v == "self":
                    args[real_k] = [
                        th.tensor(
                            self_tensor, dtype=arg_type, requires_grad=requires_grad
                        )
                    ]
                else:
                    args[real_k] = v
            if tuple_args:
                args = list(args.values())
        else:
            args = [PrimitiveFactory.generate_primitive(value=_args, recurse=True)]

        try:
            _ = len(args)
        except Exception as e:
            err = (
                f"Args must be iterable so it can be spread with * into the method. {e}"
            )
            raise Exception(err)

        # Step 4: Get the method we're going to call
        # if op_name=="grad", we need to do more operations first
        if op_name == "grad":
            self_tensor.sum().backward()  # type: ignore
        target_op_method = getattr(self_tensor, op_name)

        # Step 5: Test to see whether this method and arguments combination is valid
        # in normal PyTorch. If it this is an invalid combination, abort the test
        try:
            if not is_property:
                upcasted_args: Union[dict, list]
                # this prevents things like syft.lib.python.bool.Bool from getting
                # treated as an Int locally but then failing on the upcast to
                # builtins.bool on the remote side
                if isinstance(args, dict):
                    upcasted_args = {}
                    for k, arg in args.items():
                        upcast_attr = getattr(arg, "upcast", None)
                        if upcast_attr is not None:
                            upcasted_args[k] = upcast_attr()
                        else:
                            upcasted_args[k] = arg
                else:
                    upcasted_args = []
                    for arg in args:
                        upcast_attr = getattr(arg, "upcast", None)
                        if upcast_attr is not None:
                            upcasted_args.append(upcast_attr())
                        else:
                            upcasted_args.append(arg)

                if issubclass(type(upcasted_args), dict):
                    target_result = target_op_method(**upcasted_args)
                else:
                    target_result = target_op_method(*upcasted_args)
            else:
                # we have a property and already have its value
                target_result = target_op_method

            debug_data["target_result"] = target_result
            debug_data["target_result_type"] = type(target_result)

        except Exception as e:
            error = (
                "Exception in allowlist suite. If this is an expected exception, update"
            )
            error += " the .json file to prevent this test combination."
            print(error)
            raise e

        # Step 6: Send our target tensor to alice.
        # NOTE: send the copy we haven't mutated
        xp = self_tensor_copy.send(client)

        # if op_name=="grad", we need to do more operations first
        if op_name == "grad":
            xp.sum().backward()

        argsp: ListType[Any] = []
        if len(args) > 0 and not is_property:
            if isinstance(args, dict):
                argsp = [
                    arg.send(client) if hasattr(arg, "send") else arg
                    for arg in args.values()
                ]
            else:
                argsp = [
                    arg.send(client) if hasattr(arg, "send") else arg for arg in args
                ]

        # Step 7: get the method on the pointer to alice we want to test
        op_method = getattr(xp, op_name, None)

        # Step 8: make sure the method exists
        assert op_method is not None

        # Step 9: Execute the method remotely
        if is_property:
            # we already have the result
            result = op_method
        else:
            result = op_method(*argsp)

        # Step 10: Ensure the method returned a pointer
        assert isinstance(result, Pointer)

        # Step 11: Get the result
        result_pointer_type = type(result)
        local_result = result.get()

        debug_data["local_result"] = local_result
        debug_data["local_result_type"] = type(local_result)

        # Step 12: If there are NaN values, set them to 0 (normal for division by 0)
        try:
            target_fqn = full_name_with_qualname(klass=type(target_result))
            if target_fqn.startswith("torch.return_types."):
                fields = types_fields[type(local_result)]
                for field in fields:
                    assert compare_tensors(
                        left=getattr(local_result, field, None),
                        right=getattr(target_result, field, None),
                    )
            elif (
                issubclass(type(local_result), (str, String))
                or not hasattr(target_result, "__len__")
                and (
                    isprimitive(value=target_result)
                    or issubclass(type(local_result), PyPrimitive)
                )
            ):
                # only do single value comparisons, do lists, tuples etc below in the else
                # check that it matches functionally
                if deterministic:
                    if issubclass(type(target_result), float):
                        assert local_result + target_result == approx(2 * target_result)
                    else:
                        assert local_result == target_result

                # convert target_result for type comparison below
                target_result = PrimitiveFactory.generate_primitive(value=target_result)
            else:
                if issubclass(type(target_result), tuple) or issubclass(
                    type(local_result), tuple
                ):
                    target_result = list(target_result)
                    local_result = list(local_result)

                # to handle tuple return types for now we just make sure everything
                # is in a list of at least 1 size and then iterate over it
                delist = False
                if type(local_result) is not list:
                    delist = True
                    local_result = [local_result]
                    target_result = [target_result]

                if deterministic:
                    for i, local_item in enumerate(local_result):
                        target_item = target_result[i]
                        if issubclass(type(target_item), th.Tensor) and issubclass(
                            type(local_item), th.Tensor
                        ):
                            assert compare_tensors(left=target_item, right=local_item)
                        else:
                            if not hasattr(local_item, "__len__") and not hasattr(
                                target_item, "__len__"
                            ):
                                assert local_item + target_item == approx(
                                    2 * local_item
                                )
                            else:
                                for left, right in zip(local_item, target_item):
                                    assert left + right == approx(2 * left)

                if delist:
                    # debox the tensors if they were not lists originally
                    local_result = local_result[0]
                    target_result = target_result[0]

            # make sure the return types match
            if isprimitive(value=target_result):
                target_result = PrimitiveFactory.generate_primitive(value=target_result)

            if isprimitive(value=local_result):
                local_result = PrimitiveFactory.generate_primitive(value=local_result)
            assert type(local_result) == type(target_result)

            # TODO: Fix this workaround for types that sometimes return Tensor tuples
            # we are getting back more than 1 return type so we need to fake it until
            # we add Union to the return types
            if (
                hasattr(local_result, "__len__")
                and not issubclass(type(local_result), (th.Tensor, str, String))
                and not target_fqn.startswith("torch.return_types.")
            ):
                if issubclass(type(local_result), (list, List)) and issubclass(
                    type(target_result), (list, List)
                ):
                    assert len(local_result) == len(target_result)
                    for left, right in zip(local_result, target_result):
                        assert type(left) == type(right)
                        if deterministic:
                            if issubclass(type(left), th.Tensor) and issubclass(
                                type(right), th.Tensor
                            ):
                                assert compare_tensors(left=left, right=right)
                            elif not hasattr(local_item, "__len__") and not hasattr(
                                target_item, "__len__"
                            ):
                                assert local_item + target_item == approx(
                                    2 * local_item
                                )
                            else:
                                for left, right in zip(local_item, target_item):
                                    assert left + right == approx(2 * left)
                else:
                    # TODO: Fix this when we find one
                    raise Exception("Unsupported Union return type")

            # make sure the return type matches the specified allowlist return type
            if not issubclass(type(local_result), th.Tensor):
                if target_fqn.startswith("torch.return_types."):
                    local_type = full_name_with_qualname(klass=type(local_result))
                else:
                    local_type = full_name_with_qualname(
                        klass=type(
                            PrimitiveFactory.generate_primitive(value=local_result)
                        )
                    )
                full_result_pointer_type = full_name_with_qualname(
                    klass=result_pointer_type
                )
                python_types = "syft.lib.python"
                if local_type.startswith(python_types) and return_type.startswith(
                    python_types
                ):
                    # python types seem to resolve as both int.Int and .Int causing issues
                    # in the match
                    assert local_type.split(".")[-1] == return_type.split(".")[-1]
                elif full_result_pointer_type.endswith("UnionPointer"):
                    union_part = local_type.rsplit(".", 1)[-1]
                    # check the returned value is part of the original expected Union
                    # Bool in syft.proxy.syft.lib.misc.union.BoolFloatIntUnionPointer
                    assert union_part in full_result_pointer_type
                    # check the result pointer type matches the expected test Union
                    assert full_result_pointer_type == return_type
                else:
                    assert local_type == return_type

            # Test Passes
            support_data["status"] = "pass"
            write_support_result(support_data)

        except Exception as e:
            error = "Exception in allowlist suite during final comparison."
            print(error)
            raise e

    except Exception as e:
        print(f"Test Exception: {e}")
        support_data["status"] = "fail"
        write_support_result(support_data)
        debug_data["exception"] = str(e)
        debug_data["exception_type"] = type(e)
        write_error_debug(debug_data)
        raise e


def compare_tensors(left: th.Tensor, right: th.Tensor) -> bool:
    try:
        # if they don't match we can try to remove NaN's
        if not (left == right).all():
            # Set all NaN to 0
            # If we have two tensors like
            # local = [Nan, 0, 1] and remote = [0, Nan, 1]
            # those are not equal
            # Tensor.isnan was added in torch 1.6
            # so we need to do torch.isnan(tensor)

            # th.isnan fails on some versions of torch or methods for
            # example the op T / t() and the data_type float16 on
            # RuntimeError: "ne_cpu" not implemented for 'Half'
            nan_mask = th.isnan(left)

            # check if any of the elements are actually NaN because
            # assigning to some masked positions fails on some versions
            # of torch and some methods so its best to avoid unless
            # actually needed
            if any(nan_mask.view(-1)):
                # Use the same mask for local and target
                left[nan_mask] = 0
                right[nan_mask] = 0

            # Step 13: Ensure we got the same result locally (using
            # normal pytorch) as we did remotely using Syft pointers to
            # communicate with remote torch objects
            assert (left == right).all()
        return True
    except Exception as e:
        # if the issue is with equality of the data_type we can try
        # comparing them with numpy
        if "eq_cpu" in str(e):
            assert (left.numpy() == right.numpy()).all()
            return True
        else:
            # otherwise lets just raise the exception
            raise e
