from collections import OrderedDict
import pytest
import numpy
import torch
from functools import partial
import traceback
import io

import syft
from syft.serde import msgpack
from syft.workers.virtual import VirtualWorker

# Make dict of type codes
CODE = OrderedDict()
for cls, simplifier in msgpack.serde.get_simplifiers():
    CODE[cls] = simplifier[0]
FORCED_CODE = OrderedDict()
for cls, simplifier in msgpack.serde.forced_full_simplifiers.items():
    FORCED_CODE[cls] = simplifier[0]

########################################################################
# Functions that return list of serde samples in the following format:
# [
#   {
#    "value": original_value,
#    "simplified": simplified_value,
#    "cmp_detailed": custom_detailed_values_comparison_function, # optional
#    "cmp_simplified": custom_simplified_values_comparison_function, # optional
#    "framework": None or torch, # optional, affects tensor serialization strategy
#    "forced": (bool), # optional, enables forced full simplification
#   },
#   ...
# ]
########################################################################

########################################################################
# Native types.
########################################################################

# None


def make_none(**kwargs):
    return [{"value": None}]


# Dict.
def make_dict(**kwargs):
    return [
        {
            "value": {1: "hello", 2: "world"},
            "simplified": (
                CODE[dict],
                (
                    (1, (CODE[str], (b"hello",))),  # [not simplified tuple]  # key  # value
                    (2, (CODE[str], (b"world",))),
                ),
            ),
        },
        {
            "value": {"hello": "world"},
            "simplified": (
                CODE[dict],
                (
                    (  # [not simplified tuple]
                        (CODE[str], (b"hello",)),  # key
                        (CODE[str], (b"world",)),  # value
                    ),
                ),
            ),
        },
        {"value": {}, "simplified": (CODE[dict], tuple())},
    ]


# List.
def make_list(**kwargs):
    return [
        {
            "value": ["hello", "world"],
            "simplified": (
                CODE[list],
                ((CODE[str], (b"hello",)), (CODE[str], (b"world",))),  # item
            ),
        },
        {"value": ["hello"], "simplified": (CODE[list], ((CODE[str], (b"hello",)),))},  # item
        {"value": [], "simplified": (CODE[list], tuple())},
        # Tests that forced full simplify should return just simplified object if it doesn't have full simplifier
        {
            "forced": True,
            "value": ["hello"],
            "simplified": (CODE[list], ((CODE[str], (b"hello",)),)),  # item
        },
    ]


# Tuple.
def make_tuple(**kwargs):
    return [
        {
            "value": ("hello", "world"),
            "simplified": (CODE[tuple], ((CODE[str], (b"hello",)), (CODE[str], (b"world",)))),
        },
        {"value": ("hello",), "simplified": (CODE[tuple], ((CODE[str], (b"hello",)),))},
        {"value": tuple(), "simplified": (CODE[tuple], tuple())},
    ]


# Set.
def make_set(**kwargs):
    def compare_simplified(actual, expected):
        """When set is simplified and converted to tuple, elements order in tuple is random
        We compare tuples as sets because the set order is undefined"""
        assert actual[0] == expected[0]
        assert set(actual[1]) == set(expected[1])
        return True

    return [
        {
            "value": {"hello", "world"},
            "simplified": (CODE[set], ((CODE[str], (b"world",)), (CODE[str], (b"hello",)))),
            "cmp_simplified": compare_simplified,
        },
        {"value": {"hello"}, "simplified": (CODE[set], ((CODE[str], (b"hello",)),))},
        {"value": set([]), "simplified": (CODE[set], tuple())},
    ]


# Slice.
def make_slice(**kwargs):
    return [
        {"value": slice(10, 20, 30), "simplified": (CODE[slice], (10, 20, 30))},
        {"value": slice(10, 20), "simplified": (CODE[slice], (10, 20, None))},
        {"value": slice(10), "simplified": (CODE[slice], (None, 10, None))},
    ]


# Range.
def make_range(**kwargs):
    return [
        {"value": range(1, 3, 4), "simplified": (CODE[range], (1, 3, 4))},
        {"value": range(1, 3), "simplified": (CODE[range], (1, 3, 1))},
    ]


# String.
def make_str(**kwargs):
    return [
        {"value": "a string", "simplified": (CODE[str], (b"a string",))},
        {"value": "", "simplified": (CODE[str], (b"",))},
    ]


# Int.
def make_int(**kwargs):
    return [
        {"value": 5, "simplified": 5},
        # Tests that forced full simplify should return just simplified object if it doesn't have full simplifier
        {"forced": True, "value": 5, "simplified": 5},
    ]


# Float.
def make_float(**kwargs):
    return [{"value": 5.1, "simplified": 5.1}]


# Ellipsis.
def make_ellipsis(**kwargs):
    return [{"value": ..., "simplified": (CODE[type(Ellipsis)], (b"",))}]


########################################################################
# Numpy.
########################################################################

# numpy.ndarray
def make_numpy_ndarray(**kwargs):
    np_array = numpy.random.random((2, 2))

    def compare(detailed, original):
        """Compare numpy arrays"""
        assert numpy.array_equal(detailed, original)
        return True

    return [
        {
            "value": np_array,
            "simplified": (
                CODE[type(np_array)],
                (
                    np_array.tobytes(),  # (bytes) serialized bin
                    (CODE[tuple], (2, 2)),  # (tuple) shape
                    (CODE[str], (b"float64",)),  # (str) dtype.name
                ),
            ),
            "cmp_detailed": compare,
        }
    ]


# numpy.float32, numpy.float64, numpy.int32, numpy.int64
def make_numpy_number(dtype, **kwargs):
    num = numpy.array([2.2], dtype=dtype)[0]
    return [
        {
            "value": num,
            "simplified": (
                CODE[dtype],
                (
                    num.tobytes(),  # (bytes)
                    (CODE[str], (num.dtype.name.encode("utf-8"),)),  # (str) dtype.name
                ),
            ),
        }
    ]


########################################################################
# PyTorch.
########################################################################

# Utility functions.


def compare_modules(detailed, original):
    """Compare ScriptModule instances"""
    input = torch.randn(10, 3)
    # NOTE: after serde TopLevelTracedModule or jit.ScriptFunction become
    # ScriptModule (that's what torch.jit.load returns in detail function)
    assert isinstance(detailed, torch.jit.ScriptModule)
    # Code changes after torch.jit.load(): function becomes `forward` method
    if type(original) != torch.jit.ScriptFunction:
        assert detailed.code == original.code
    # model outputs match
    assert detailed(input).equal(original(input))
    return True


def save_to_buffer(tensor) -> bin:
    """Serializes a pytorch tensor to binary"""
    binary_stream = io.BytesIO()
    torch.save(tensor, binary_stream)
    return binary_stream.getvalue()


# torch.device
def make_torch_device(**kwargs):
    torch_device = torch.device("cpu")
    return [
        {
            "value": torch_device,
            "simplified": (CODE[type(torch_device)], ((CODE[str], (b"cpu",)),)),  # (str) device
        }
    ]


# torch.dtype
def make_torch_dtype(**kwargs):
    torch_dtype = torch.int32
    return [
        {"value": torch_dtype, "simplified": (CODE[type(torch_dtype)], "int32")}  # (str) device
    ]


# torch.jit.ScriptModule
def make_torch_scriptmodule(**kwargs):
    class ScriptModule(torch.jit.ScriptModule):
        def __init__(self):
            super(ScriptModule, self).__init__()

        @torch.jit.script_method
        def forward(self, x):  # pragma: no cover
            return x + 2

    sm = ScriptModule()
    return [
        {
            "value": sm,
            "simplified": (
                CODE[torch.jit.ScriptModule],
                (sm.save_to_buffer(),),  # (bytes) serialized torchscript
            ),
            "cmp_detailed": compare_modules,
        }
    ]


# torch.jit.ScriptFunction
def make_torch_scriptfunction(**kwargs):
    @torch.jit.script
    def func(x):  # pragma: no cover
        return x + 2

    return [
        {
            "value": func,
            "simplified": (
                CODE[torch.jit.ScriptFunction],
                (func.save_to_buffer(),),  # (bytes) serialized torchscript
            ),
            "cmp_detailed": compare_modules,
        }
    ]


# torch.memory_format
def make_torch_memoryformat(**kwargs):
    memory_format = torch.preserve_format

    return [{"value": memory_format, "simplified": (CODE[torch.memory_format], 3)}]


# torch.jit.TopLevelTracedModule
# NOTE: if the model is created inside the function, it will be serialized differently depending on the context
class TopLevelTraceModel(torch.nn.Module):
    def __init__(self):
        super(TopLevelTraceModel, self).__init__()
        self.w1 = torch.nn.Parameter(torch.randn(3, 1), requires_grad=True)
        self.b1 = torch.nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, x):
        x = x @ self.w1 + self.b1
        return x


topLevelTraceModel = TopLevelTraceModel()


def make_torch_topleveltracedmodule(**kwargs):
    tm = torch.jit.trace(topLevelTraceModel, torch.randn(10, 3))

    return [
        {
            "value": tm,
            "simplified": (
                CODE[torch.jit.TopLevelTracedModule],
                (tm.save_to_buffer(),),  # (bytes) serialized torchscript
            ),
            "cmp_detailed": compare_modules,
        }
    ]


# torch.nn.parameter.Parameter
def make_torch_parameter(**kwargs):
    param = torch.nn.Parameter(torch.randn(3, 3), requires_grad=True)

    def compare(detailed, original):
        assert type(detailed) == torch.nn.Parameter
        assert detailed.data.equal(original.data)
        assert detailed.id == original.id
        assert detailed.requires_grad == original.requires_grad
        return True

    return [
        {
            "value": param,
            "simplified": (
                CODE[torch.nn.Parameter],
                (
                    param.id,  # (int) id
                    msgpack.serde._simplify(
                        kwargs["workers"]["serde_worker"], param.data
                    ),  # (Tensor) data
                    param.requires_grad,  # (bool) requires_grad
                    None,
                ),
            ),
            "cmp_detailed": compare,
        }
    ]


# torch.Tensor
def make_torch_tensor(**kwargs):
    tensor = torch.randn(3, 3)
    tensor.tag("tag1")
    tensor.describe("desc")

    def compare(detailed, original):
        assert type(detailed) == torch.Tensor
        assert detailed.data.equal(original.data)
        assert detailed.id == original.id
        assert detailed.requires_grad == original.requires_grad
        assert detailed.tags == original.tags
        assert detailed.description == original.description
        return True

    return [
        # Default pytorch tensor serialization strategy
        {
            "value": tensor,
            "simplified": (
                CODE[torch.Tensor],
                (
                    tensor.id,  # (int) id
                    save_to_buffer(tensor),  # (bytes) serialized tensor
                    None,  # (AbstractTensor) chain
                    None,  # (AbstractTensor) grad_chain
                    (CODE[set], ((CODE[str], (b"tag1",)),)),  # (set of str) tags
                    (CODE[str], (b"desc",)),  # (str) description
                    (CODE[str], (b"torch",)),  # (str) framework
                    None,  # (int) origin
                    None,  # (int) id_at_origin
                ),
            ),
            "cmp_detailed": compare,
        },
        # "All" tensor serialization strategy
        {
            "framework": None,
            "value": tensor,
            "simplified": (
                CODE[torch.Tensor],
                (
                    tensor.id,  # (int) id
                    (
                        CODE[tuple],
                        (  # serialized tensor
                            (CODE[tuple], (3, 3)),  # tensor.shape
                            (CODE[str], (b"float32",)),  # tensor.dtype
                            (
                                CODE[list],
                                tuple(tensor.flatten().tolist()),
                            ),  # tensor contents as flat list
                        ),
                    ),
                    None,  # (AbstractTensor) chain
                    None,  # (AbstractTensor) grad_chain
                    (CODE[set], ((CODE[str], (b"tag1",)),)),  # (set of str) tags
                    (CODE[str], (b"desc",)),  # (str) description
                    (CODE[str], (b"all",)),  # (str) framework
                    None,  # (int) origin
                    None,  # (int) id_at_origin
                ),
            ),
            "cmp_detailed": compare,
        },
    ]


# torch.Size
def make_torch_size(**kwargs):
    return [
        {
            "value": torch.randn(3, 3).size(),
            "simplified": (CODE[torch.Size], (3, 3)),  # (int) *shape
        }
    ]


########################################################################
# PySyft.
########################################################################

# Utility functions


def compare_actions(detailed, original):
    """Compare 2 Actions"""
    assert len(detailed) == len(original)
    for original_op, detailed_op in zip(original, detailed):
        for original_arg, detailed_arg in zip(original_op.args, detailed_op.args):
            assert original_arg == detailed_arg
        for original_return, detailed_return in zip(original_op.return_ids, detailed_op.return_ids):
            assert original_return == detailed_return
        assert original_op.name == detailed_op.name
        assert original_op.kwargs == detailed_op.kwargs
    return True


def compare_placeholders_list(detailed, original):
    """Compare 2 lists of placeholders"""
    assert len(detailed) == len(original)
    for original_ph, detailed_ph in zip(original, detailed):
        assert detailed_ph.id == original_ph.id
        assert detailed_ph.tags == original_ph.tags
        assert detailed_ph.description == original_ph.description
        assert detailed_ph.expected_shape == original_ph.expected_shape
    return True


def compare_placeholders_dict(detailed, original):
    """Compare 2 dicts of placeholders"""
    assert len(detailed) == len(original)
    for key, detailed_ph in detailed.items():
        original_ph = original[key]
        assert detailed_ph.id == original_ph.id
        assert detailed_ph.tags == original_ph.tags
        assert detailed_ph.description == original_ph.description
        assert detailed_ph.expected_shape == original_ph.expected_shape
    return True


def compare_roles(detailed, original):
    """Compare 2 Roles"""
    assert detailed.id == original.id
    compare_actions(detailed.actions, original.actions)
    compare_placeholders_list(detailed.state.state_placeholders, original.state.state_placeholders)
    compare_placeholders_dict(detailed.placeholders, original.placeholders)
    assert detailed.input_placeholder_ids == original.input_placeholder_ids
    assert detailed.output_placeholder_ids == original.output_placeholder_ids
    return True


# AdditiveSharingTensor
def make_additivesharingtensor(**kwargs):
    workers = kwargs["workers"]
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]
    tensor = torch.tensor([[3.1, 4.3]]).fix_prec().share(alice, bob, crypto_provider=james)
    ast = tensor.child.child

    def compare(detailed, original):
        assert (
            type(detailed)
            == syft.frameworks.torch.tensors.interpreters.additive_shared.AdditiveSharingTensor
        )
        assert detailed.id == original.id
        assert detailed.field == original.field
        assert detailed.child.keys() == original.child.keys()
        return True

    return [
        {
            "value": ast,
            "simplified": (
                CODE[
                    syft.frameworks.torch.tensors.interpreters.additive_shared.AdditiveSharingTensor
                ],
                (
                    ast.id,  # (int or str) id
                    (CODE[str], (str(ast.field).encode("utf-8"),))
                    if ast.field == 2 ** 64
                    else ast.field,  # (int or str) field
                    ast.dtype.encode("utf-8"),
                    (CODE[str], (ast.crypto_provider.id.encode("utf-8"),)),  # (str) worker_id
                    msgpack.serde._simplify(
                        kwargs["workers"]["serde_worker"], ast.child
                    ),  # (dict of AbstractTensor) simplified chain
                    ast.get_garbage_collect_data(),
                ),
            ),
            "cmp_detailed": compare,
        }
    ]


# FixedPrecisionTensor
def make_fixedprecisiontensor(**kwargs):
    workers = kwargs["workers"]
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]
    t = torch.tensor([[3.1, 4.3]])
    fpt_tensor = t.fix_prec(base=12, precision_fractional=5).share(
        alice, bob, crypto_provider=james
    )
    fpt = fpt_tensor.child
    fpt.tag("tag1")
    fpt.describe("desc")
    # AdditiveSharingTensor.simplify sets garbage_collect_data=False on child tensors during simplify
    # This changes tensors' internal state in chain and is required to pass the test
    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], fpt)

    def compare(detailed, original):
        assert (
            type(detailed)
            == syft.frameworks.torch.tensors.interpreters.precision.FixedPrecisionTensor
        )
        assert detailed.id == original.id
        assert detailed.field == original.field
        assert detailed.base == original.base
        assert detailed.precision_fractional == original.precision_fractional
        assert detailed.kappa == original.kappa
        assert detailed.tags == original.tags
        assert detailed.description == original.description
        return True

    return [
        {
            "value": fpt,
            "simplified": (
                CODE[syft.frameworks.torch.tensors.interpreters.precision.FixedPrecisionTensor],
                (
                    fpt.id,  # (int or str) id
                    (CODE[str], (str(fpt.field).encode("utf-8"),))
                    if fpt.field == 2 ** 64
                    else fpt.field,  # (int or str) field
                    fpt.dtype,  # (str) dtype
                    12,  # (int) base
                    5,  # (int) precision_fractional
                    fpt.kappa,  # (int) kappa
                    (CODE[set], ((CODE[str], (b"tag1",)),)),  # (set of str) tags
                    (CODE[str], (b"desc",)),  # (str) description
                    msgpack.serde._simplify(
                        kwargs["workers"]["serde_worker"], fpt.child
                    ),  # (AbstractTensor) chain
                ),
            ),
            "cmp_detailed": compare,
        }
    ]


# LoggingTensor
def make_loggingtensor(**kwargs):
    t = torch.randn(3, 3)
    lt = syft.frameworks.torch.tensors.decorators.logging.LoggingTensor().on(t).child

    def compare(detailed, original):
        assert type(detailed) == syft.frameworks.torch.tensors.decorators.logging.LoggingTensor
        assert detailed.id == original.id
        assert detailed.child.equal(original.child)
        return True

    return [
        {
            "value": lt,
            "simplified": (
                CODE[syft.frameworks.torch.tensors.decorators.logging.LoggingTensor],
                (
                    lt.id,  # (int or str) id
                    msgpack.serde._simplify(
                        kwargs["workers"]["serde_worker"], lt.child
                    ),  # (AbstractTensor) chain
                ),
            ),
            "cmp_detailed": compare,
        }
    ]


# syft.execution.placeholder_id.PlaceholderId
def make_placeholder_id(**kwargs):
    p = syft.execution.placeholder.PlaceHolder()
    obj_id = p.id

    def compare(detailed, original):
        assert type(detailed) == syft.execution.placeholder_id.PlaceholderId
        assert detailed.value == original.value
        return True

    return [
        {
            "value": obj_id,
            "simplified": (CODE[syft.execution.placeholder_id.PlaceholderId], (obj_id.value,)),
            "cmp_detailed": compare,
        }
    ]


# syft.generic.pointers.multi_pointer.MultiPointerTensor
def make_multipointertensor(**kwargs):
    workers = kwargs["workers"]
    alice, bob = workers["alice"], workers["bob"]
    t = torch.randn(3, 3)
    mpt = t.send(alice, bob).child

    def compare(detailed, original):
        assert type(detailed) == syft.generic.pointers.multi_pointer.MultiPointerTensor
        assert detailed.id == original.id
        assert detailed.child.keys() == original.child.keys()
        return True

    return [
        {
            "value": mpt,
            "simplified": (
                CODE[syft.generic.pointers.multi_pointer.MultiPointerTensor],
                (
                    mpt.id,  # (int or str) id
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], mpt.child),  # (dict)
                ),
            ),
            "cmp_detailed": compare,
        }
    ]


# syft.frameworks.torch.fl.dataset
def make_basedataset(**kwargs):
    workers = kwargs["workers"]
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]
    dataset = syft.BaseDataset(torch.tensor([1, 2, 3, 4]), torch.tensor([5, 6, 7, 8]))
    dataset.tag("#tag1").describe("desc")

    def compare(detailed, original):
        assert type(detailed) == syft.BaseDataset
        assert (detailed.data == original.data).all()
        assert (detailed.targets == original.targets).all()
        assert detailed.id == original.id
        assert detailed.tags == original.tags
        assert detailed.description == original.description
        return True

    return [
        {
            "value": dataset,
            "simplified": (
                CODE[syft.frameworks.torch.fl.dataset.BaseDataset],
                (
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], dataset.data),
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], dataset.targets),
                    dataset.id,
                    msgpack.serde._simplify(
                        kwargs["workers"]["serde_worker"], dataset.tags
                    ),  # (set of str) tags
                    msgpack.serde._simplify(
                        kwargs["workers"]["serde_worker"], dataset.description
                    ),  # (str) description
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], dataset.child),
                ),
            ),
            "cmp_detailed": compare,
        }
    ]


# syft.generic.pointers.pointer_dataset.PointerDataset
def make_pointerdataset(**kwargs):
    alice, me = kwargs["workers"]["alice"], kwargs["workers"]["me"]
    data = torch.tensor([1, 2, 3, 4])
    targets = torch.tensor([5, 6, 7, 8])
    dataset = syft.BaseDataset(data, targets).tag("#test")
    dataset.send(alice)
    ptr = me.request_search(["#test"], location=alice)[0]

    def compare(detailed, original):
        assert type(detailed) == syft.generic.pointers.pointer_dataset.PointerDataset
        assert detailed.id == original.id
        assert detailed.id_at_location == original.id_at_location
        assert detailed.location == original.location
        assert detailed.tags == original.tags
        assert detailed.description == original.description
        assert detailed.garbage_collect_data == original.garbage_collect_data
        return True

    return [
        {
            "value": ptr,
            "simplified": (
                CODE[syft.generic.pointers.pointer_dataset.PointerDataset],
                (
                    ptr.id,  # (int) id
                    ptr.id_at_location,  # (int) id_at_location
                    (CODE[str], (b"alice",)),  # (str) worker_id
                    (CODE[set], ((CODE[str], (b"#test",)),)),  # (set or None) tags
                    None,  # description
                    False,  # (bool) garbage_collect_data
                ),
            ),
            "cmp_detailed": compare,
        }
    ]


# syft.execution.plan.Plan
def make_plan(**kwargs):
    # Function to plan
    @syft.func2plan([torch.Size((3,))])
    def plan(x):
        x = x + x
        y = torch.abs(x)
        return x

    # Model to plan
    class Net(syft.Plan):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = torch.nn.Linear(3, 3)
            self.fc2 = torch.nn.Linear(3, 2)

        def forward(self, x):
            x = torch.nn.functional.relu(self.fc1(x))
            x = self.fc2(x)
            return torch.nn.functional.log_softmax(x, dim=0)

    with kwargs["workers"]["serde_worker"].registration_enabled():
        model_plan = Net()
        model_plan.build(torch.tensor([1.0, 2.0, 3.0]))

    def compare(detailed, original):
        assert type(detailed) == syft.execution.plan.Plan
        assert detailed.id == original.id
        compare_roles(detailed.role, original.role)
        assert detailed.include_state == original.include_state
        assert detailed.is_built == original.is_built
        assert detailed.name == original.name
        assert detailed.tags == original.tags
        assert detailed.description == original.description
        with kwargs["workers"]["serde_worker"].registration_enabled():
            t = torch.tensor([1.1, -2, 3])
            res1 = detailed(t)
            res2 = original(t)
        assert res1.equal(res2)
        return True

    return [
        {
            "value": plan,
            "simplified": (
                CODE[syft.execution.plan.Plan],
                (
                    plan.id,  # (int or str) id
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], plan.role),
                    plan.include_state,
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], plan.name),
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], plan.tags),
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], plan.description),
                    msgpack.serde._simplify(
                        kwargs["workers"]["serde_worker"], plan.torchscript
                    ),  # Torchscript
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], plan.input_types),
                ),
            ),
            "cmp_detailed": compare,
        },
        {
            "value": model_plan,
            "simplified": (
                CODE[syft.execution.plan.Plan],
                (
                    model_plan.id,  # (int or str) id
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], model_plan.role),
                    model_plan.include_state,
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], model_plan.name),
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], model_plan.tags),
                    msgpack.serde._simplify(
                        kwargs["workers"]["serde_worker"], model_plan.description
                    ),
                    msgpack.serde._simplify(
                        kwargs["workers"]["serde_worker"], model_plan.torchscript
                    ),  # Torchscript
                    msgpack.serde._simplify(
                        kwargs["workers"]["serde_worker"], model_plan.input_types
                    ),
                ),
            ),
            "cmp_detailed": compare,
        },
    ]


# Role
def make_role(**kwargs):
    @syft.func2plan(args_shape=[(1,)], state=(torch.tensor([1.0]),))
    def plan_abs(x, state):
        (bias,) = state.read()
        x = x.abs()
        return x + bias

    plan_abs.build(torch.tensor([3.0]))
    role = plan_abs.role

    def compare(detailed, original):
        assert type(detailed) == syft.execution.role.Role
        compare_roles(detailed, original)
        return True

    return [
        {
            "value": role,
            "simplified": (
                CODE[syft.execution.role.Role],
                (
                    role.id,
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], role.actions),
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], role.state),
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], role.placeholders),
                    role.input_placeholder_ids,
                    role.output_placeholder_ids,
                ),
            ),
            "cmp_detailed": compare,
        }
    ]


def make_type(**kwargs):
    serialized_type = type("test")

    def compare(detailed, original):
        assert type(detailed) == type(original)
        assert detailed == original
        return True

    return [
        {
            "value": serialized_type,
            "simplified": (msgpack.serde._simplify(syft.hook.local_worker, serialized_type)),
            "cmp_detailed": compare,
        }
    ]


def make_nested_type_wrapper(**kwargs):
    reference_serialized_input = (
        (type(torch.tensor([1.0, -2.0])), type(torch.tensor([1, 2]))),
        {
            "k1": [type(5), (type(True), type(False))],
            "k2": {
                "kk1": [type(torch.tensor([5, 7])), type(torch.tensor([5, 7]))],
                "kk2": [type(True), (type(torch.tensor([9, 10])),)],
            },
            "k3": type(torch.tensor([8])),
        },
        type(torch.tensor([11, 12])),
        (type(1), (type(2), (type(3), (type(4), [type(5), type(6)])))),
    )

    wrapper = syft.execution.plan.NestedTypeWrapper()
    wrapper.nested_input_types = reference_serialized_input

    def compare(detailed, original):
        assert detailed.nested_input_types == original.nested_input_types
        return True

    return [
        {
            "value": wrapper,
            "simplified": syft.serde.msgpack.serde._simplify(syft.hook.local_worker, wrapper),
            "cmp_detailed": compare,
        }
    ]


# State
def make_state(**kwargs):
    me = kwargs["workers"]["me"]

    t1, t2 = torch.randn(3, 3), torch.randn(3, 3)
    p1, p2 = syft.PlaceHolder(), syft.PlaceHolder()
    p1.tag("state1"), p2.tag("state2")
    p1.instantiate(t1), p2.instantiate(t2)
    state = syft.execution.state.State(state_placeholders=[p1, p2])

    def compare(detailed, original):
        assert type(detailed) == syft.execution.state.State
        compare_placeholders_list(detailed.state_placeholders, original.state_placeholders)
        for i in range(len(original.tensors())):
            assert detailed.tensors()[i].equal(original.tensors()[i])
        return True

    return [
        {
            "value": state,
            "simplified": (
                CODE[syft.execution.state.State],
                (
                    (
                        CODE[list],
                        (  # (list) state_placeholders
                            msgpack.serde._simplify(kwargs["workers"]["serde_worker"], p1),
                            msgpack.serde._simplify(kwargs["workers"]["serde_worker"], p2),
                        ),
                    ),
                    (
                        CODE[list],
                        (  # (list) tensors
                            msgpack.serde._simplify(kwargs["workers"]["serde_worker"], t1),
                            msgpack.serde._simplify(kwargs["workers"]["serde_worker"], t2),
                        ),
                    ),
                ),
            ),
            "cmp_detailed": compare,
        }
    ]


# Protocol
def make_protocol(**kwargs):
    alice = kwargs["workers"]["alice"]
    bob = kwargs["workers"]["bob"]

    @syft.func2protocol(roles=["alice", "bob"], args_shape={"alice": ((1,),), "bob": ((1,),)})
    def protocol(alice, bob):
        tensor1 = alice.load(torch.tensor([1]))
        tensor2 = bob.load(torch.tensor([1]))

        t1plus = tensor1 + 1
        t2plus = tensor2 + 1

        return t1plus, t2plus

    protocol.build()

    # plan.owner = worker
    protocol.tag("aaa")
    protocol.describe("desc")

    def compare(detailed, original):
        assert type(detailed) == syft.execution.protocol.Protocol
        assert detailed.id == original.id
        assert detailed.name == original.name
        assert detailed.roles.keys() == original.roles.keys()
        for k, v in detailed.roles.items():
            assert compare_roles(original.roles[k], v)
        assert detailed.tags == original.tags
        assert detailed.description == original.description
        return True

    return [
        {
            "value": protocol,
            "simplified": (
                CODE[syft.execution.protocol.Protocol],
                (
                    protocol.id,  # (int or str) id
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], protocol.name),
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], protocol.roles),
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], protocol.tags),
                    msgpack.serde._simplify(
                        kwargs["workers"]["serde_worker"], protocol.description
                    ),
                ),
            ),
            "cmp_detailed": compare,
        }
    ]


# syft.generic.pointers.pointer_tensor.PointerTensor
def make_pointertensor(**kwargs):
    alice = kwargs["workers"]["alice"]
    tensor = torch.randn(3, 3)
    ptr = tensor.send(alice).child

    def compare(detailed, original):
        assert type(detailed) == syft.generic.pointers.pointer_tensor.PointerTensor
        assert detailed.id == original.id
        assert detailed.id_at_location == original.id_at_location
        assert detailed.location == original.location
        assert detailed.point_to_attr == original.point_to_attr
        # Not testing grabage collect data as we are always setting it as False at receiver end
        # irrespective of its initial value
        assert detailed.garbage_collect_data == original.garbage_collect_data
        assert detailed.get().equal(tensor)
        return True

    return [
        {
            "value": ptr,
            "simplified": (
                CODE[syft.generic.pointers.pointer_tensor.PointerTensor],
                (
                    ptr.id,  # (int or str) id
                    ptr.id_at_location,  # (int or str) id_at_location
                    (CODE[str], (b"alice",)),  # (str) worker_id
                    None,  # (str) point_to_attr
                    (CODE[torch.Size], (3, 3)),  # (torch.Size) _shape
                    True,  # (bool) garbage_collect_data
                    ptr.tags,
                    ptr.description,
                ),
            ),
            "cmp_detailed": compare,
        }
    ]


# syft.generic.pointers.pointer_plan.PointerPlan
def make_pointerplan(**kwargs):
    alice, me = kwargs["workers"]["alice"], kwargs["workers"]["me"]

    @syft.func2plan([torch.Size((1, 3))])
    def plan(x):
        x = x + x
        x = torch.abs(x)
        return x

    plan.send(alice)
    ptr = me.request_search([plan.id], location=alice)[0]

    def compare(detailed, original):
        assert type(detailed) == syft.generic.pointers.pointer_plan.PointerPlan
        assert detailed.id == original.id
        assert detailed.id_at_location == original.id_at_location
        assert detailed.location == original.location
        assert detailed.garbage_collect_data == original.garbage_collect_data
        # execute
        t = torch.randn(3, 3).send(alice)
        assert detailed(t).get().equal(original(t).get())
        return True

    return [
        {
            "value": ptr,
            "simplified": (
                CODE[syft.generic.pointers.pointer_plan.PointerPlan],
                (
                    ptr.id,  # (int) id
                    ptr.id_at_location,  # (int) id_at_location
                    (CODE[str], (b"alice",)),  # (str) worker_id
                    (CODE[set], ()),  # (set or None) tags
                    False,  # (bool) garbage_collect_data
                ),
            ),
            "cmp_detailed": compare,
        }
    ]


# syft.generic.pointers.object_wrapper.ObjectWrapper
def make_objectwrapper(**kwargs):
    obj = torch.randn(3, 3)
    wrapper = syft.generic.pointers.object_wrapper.ObjectWrapper(obj, id=123)

    def compare(detailed, original):
        assert type(detailed) == syft.generic.pointers.object_wrapper.ObjectWrapper
        assert detailed.id == original.id
        # tensors
        assert detailed.obj.equal(original.obj)
        return True

    return [
        {
            "value": wrapper,
            "simplified": (
                CODE[syft.generic.pointers.object_wrapper.ObjectWrapper],
                (
                    123,  # (int) id
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], obj),  # (Any) obj
                ),
            ),
            "cmp_detailed": compare,
        }
    ]


# syft.generic.pointers.object_pointer.ObjectPointer
def make_objectpointer(**kwargs):
    alice = kwargs["workers"]["alice"]
    obj = torch.randn(3, 3)
    obj_ptr = obj.send(alice)
    ptr = syft.generic.pointers.object_pointer.ObjectPointer.create_pointer(obj, alice, obj.id)

    def compare(detailed, original):
        assert type(detailed) == syft.generic.pointers.object_pointer.ObjectPointer
        assert detailed.id == original.id
        assert detailed.id_at_location == original.id_at_location
        assert detailed.location == original.location
        assert detailed.point_to_attr == original.point_to_attr
        assert detailed.garbage_collect_data == original.garbage_collect_data
        return True

    return [
        {
            "value": ptr,
            "simplified": (
                CODE[syft.generic.pointers.object_pointer.ObjectPointer],
                (
                    ptr.id,  # (int or str) id
                    ptr.id_at_location,  # (int or str) id
                    (CODE[str], (b"alice",)),  # (str) location.id
                    None,  # (str) point_to_attr
                    True,  # (bool) garbage_collect_data
                ),
            ),
            "cmp_detailed": compare,
        }
    ]


# syft.generic.string.String
def make_string(**kwargs):
    def compare_simplified(actual, expected):
        """This is a custom comparison functino.
           The reason for using this is that when set is that tags are use. Tags are sets.
           When sets are simplified and converted to tuple, elements order in tuple is random
           We compare tuples as sets because the set order is undefined.

           This function is inspired by the one with the same name defined above in `make_set`.
        """
        assert actual[0] == expected[0]
        assert actual[1][0] == expected[1][0]
        assert actual[1][1] == expected[1][1]
        assert actual[1][2][0] == expected[1][2][0]
        assert set(actual[1][2][1]) == set(expected[1][2][1])
        assert actual[1][3] == expected[1][3]
        return True

    return [
        {
            "value": syft.generic.string.String(
                "Hello World", id=1234, tags=set(["tag1", "tag2"]), description="description"
            ),
            "simplified": (
                CODE[syft.generic.string.String],
                (
                    (CODE[str], (b"Hello World",)),
                    1234,
                    (CODE[set], ((CODE[str], (b"tag1",)), (CODE[str], (b"tag2",)))),
                    (CODE[str], (b"description",)),
                ),
            ),
            "cmp_simplified": compare_simplified,
        }
    ]


# syft.federated.train_config.TrainConfig
def make_trainconfig(**kwargs):
    class Model(torch.jit.ScriptModule):
        def __init__(self):
            super(Model, self).__init__()
            self.w1 = torch.nn.Parameter(torch.randn(10, 1), requires_grad=True)
            self.b1 = torch.nn.Parameter(torch.randn(1), requires_grad=True)

        @torch.jit.script_method
        def forward(self, x):  # pragma: no cover
            x = x @ self.w1 + self.b1
            return x

    class Loss(torch.jit.ScriptModule):
        def __init__(self):
            super(Loss, self).__init__()

        @torch.jit.script_method
        def forward(self, pred, target):  # pragma: no cover
            return ((target.view(pred.shape).float() - pred.float()) ** 2).mean()

    loss = Loss()
    model = Model()
    conf = syft.federated.train_config.TrainConfig(
        model=model, loss_fn=loss, batch_size=2, optimizer="SGD", optimizer_args={"lr": 0.1}
    )

    def compare(detailed, original):
        assert type(detailed) == syft.federated.train_config.TrainConfig
        assert detailed.id == original.id
        assert detailed._model_id == original._model_id
        assert detailed._loss_fn_id == original._loss_fn_id
        assert detailed.batch_size == original.batch_size
        assert detailed.epochs == original.epochs
        assert detailed.optimizer == original.optimizer
        assert detailed.optimizer_args == original.optimizer_args
        assert detailed.max_nr_batches == original.max_nr_batches
        assert detailed.shuffle == original.shuffle
        return True

    return [
        {
            "value": conf,
            "simplified": (
                CODE[syft.federated.train_config.TrainConfig],
                (
                    None,  # (int) _model_id
                    None,  # (int) _loss_fn_id
                    2,  # (int) batch_size
                    1,  # (int) epochs
                    (CODE[str], (b"SGD",)),  # (str) optimizer
                    (CODE[dict], (((CODE[str], (b"lr",)), 0.1),)),  # (dict) optimizer_args
                    conf.id,  # (int or str)
                    -1,  # (int) max_nr_batches
                    True,  # (bool) shuffle
                ),
            ),
            "cmp_detailed": compare,
        }
    ]


# syft.workers.virtual.VirtualWorker
def make_virtual_worker(**kwargs):
    worker = VirtualWorker(
        id=f"serde-worker-{cls.__name__}",
        hook=kwargs["workers"]["serde_worker"].hook,
        auto_add=False,
    )

    t = torch.rand(3, 3)
    with worker.registration_enabled():
        worker.register_obj(t)

    def compare(detailed, original):
        assert isinstance(detailed, syft.workers.virtual.VirtualWorker)
        assert detailed.id == original.id
        return True

    return [
        {
            "value": worker,
            "simplified": (
                CODE[syft.workers.virtual.VirtualWorker],
                ((CODE[str], (b"serde-worker-VirtualWorker",)),),  # id (str)
            ),
            "cmp_detailed": compare,
        },
        # Forced simplification
        {
            "forced": True,
            "value": worker,
            "simplified": (
                FORCED_CODE[syft.workers.virtual.VirtualWorker],
                (
                    (CODE[str], (b"serde-worker-VirtualWorker",)),  # id (str)
                    msgpack.serde._simplify(
                        worker, worker.object_store._objects
                    ),  # (dict) _objects
                    worker.auto_add,  # (bool) auto_add
                ),
            ),
            "cmp_detailed": compare,
        },
    ]


# syft.frameworks.torch.tensors.interpreters.autograd.AutogradTensor
def make_autogradtensor(**kwargs):

    t = torch.tensor([1, 2, 3])
    agt = (
        syft.frameworks.torch.tensors.interpreters.autograd.AutogradTensor(
            owner=kwargs["workers"]["serde_worker"]
        )
        .on(t)
        .child
    )
    agt.tag("aaa")
    agt.describe("desc")

    def compare(detailed, original):
        assert type(detailed) == syft.frameworks.torch.tensors.interpreters.autograd.AutogradTensor
        assert detailed.owner == original.owner
        assert detailed.id == original.id
        assert detailed.child.equal(original.child)
        assert detailed.requires_grad == original.requires_grad
        assert detailed.preinitialize_grad == original.preinitialize_grad
        assert detailed.grad_fn == original.grad_fn
        assert detailed.tags == original.tags
        assert detailed.description == original.description
        return True

    return [
        {
            "value": agt,
            "simplified": (
                CODE[syft.frameworks.torch.tensors.interpreters.autograd.AutogradTensor],
                (
                    agt.id,  # (int)
                    msgpack.serde._simplify(
                        kwargs["workers"]["serde_worker"], agt.child
                    ),  # (AbstractTensor) chain
                    True,  # (bool) requires_grad
                    False,  # (bool) preinitialize_grad
                    None,  # [always None, ignored in constructor] grad_fn
                    (CODE[set], ((CODE[str], (b"aaa",)),)),  # (set of str) tags
                    (CODE[str], (b"desc",)),  # (str) description
                ),
            ),
            "cmp_detailed": compare,
        }
    ]


# syft.frameworks.torch.tensors.interpreters.private.PrivateTensor
def make_privatetensor(**kwargs):
    t = torch.tensor([1, 2, 3])
    pt = t.private_tensor(allowed_users=("test",))
    pt.tag("tag1")
    pt.describe("private")
    pt = pt.child

    def compare(detailed, original):
        assert type(detailed) == syft.frameworks.torch.tensors.interpreters.private.PrivateTensor
        assert detailed.id == original.id
        assert detailed.allowed_users == original.allowed_users
        assert detailed.tags == original.tags
        assert detailed.description == original.description
        assert detailed.child.equal(original.child)
        return True

    return [
        {
            "value": pt,
            "simplified": (
                CODE[syft.frameworks.torch.tensors.interpreters.private.PrivateTensor],
                (
                    pt.id,  # (int or str) id
                    (CODE[tuple], ((CODE[str], (b"test",)),)),  # (tuple of ?) allowed_users
                    (CODE[set], ((CODE[str], (b"tag1",)),)),  # (set of str) tags
                    (CODE[str], (b"private",)),  # (str) description
                    msgpack.serde._simplify(
                        kwargs["workers"]["serde_worker"], t
                    ),  # (AbstractTensor) chain
                ),
            ),
            "cmp_detailed": compare,
        }
    ]


# syft.frameworks.torch.tensors.interpreters.PlaceHolder
def make_placeholder(**kwargs):
    ph = syft.execution.placeholder.PlaceHolder(shape=torch.randn(3, 4).shape)
    ph.tag("tag1")
    ph.describe("just a placeholder")

    def compare(detailed, original):
        assert type(detailed) == syft.execution.placeholder.PlaceHolder
        assert detailed.id == original.id
        assert detailed.tags == original.tags
        assert detailed.description == original.description
        assert detailed.expected_shape == original.expected_shape
        return True

    return [
        {
            "value": ph,
            "simplified": (
                CODE[syft.execution.placeholder.PlaceHolder],
                (
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], ph.id),
                    (CODE[set], ((CODE[str], (b"tag1",)),)),  # (set of str) tags
                    (CODE[str], (b"just a placeholder",)),  # (str) description
                    (CODE[tuple], (3, 4)),  # (tuple of int) expected_shape
                ),
            ),
            "cmp_detailed": compare,
        }
    ]


# syft.execution.communication.CommunicationAction
def make_communication_action(**kwargs):
    bob = kwargs["workers"]["bob"]
    alice = kwargs["workers"]["alice"]
    bob.log_msgs = True

    x = torch.tensor([1, 2, 3, 4]).send(bob)
    x.remote_send(alice)
    com = bob._get_msg(-1).action

    bob.log_msgs = False

    def compare(detailed, original):
        assert type(detailed) == syft.messaging.message.CommunicationAction

        detailed_msg = (
            detailed.name,
            detailed.target,
            detailed.args,
            detailed.kwargs,
            detailed.return_ids,
            detailed.return_value,
        )
        original_msg = (
            original.name,
            original.target,
            original.args,
            original.kwargs,
            original.return_ids,
            original.return_value,
        )

        for i in range(len(original_msg)):
            if type(original_msg[i]) != torch.Tensor:
                assert detailed_msg[i] == original_msg[i], f"{detailed_msg[i]} != {original_msg[i]}"
            else:
                assert detailed_msg[i].equal(
                    original_msg[i]
                ), f"{detailed_msg[i]} != {original_msg[i]}"

        return True

    return [
        {
            "value": com,
            "simplified": (
                CODE[syft.execution.communication.CommunicationAction],
                (
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], com.name),
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], com.target),
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], com.args),
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], com.kwargs),
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], com.return_ids),
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], com.return_value),
                ),
            ),
            "cmp_detailed": compare,
        }
    ]


# syft.execution.computation.ComputationAction
def make_computation_action(**kwargs):
    bob = kwargs["workers"]["bob"]
    bob.log_msgs = True

    x = torch.tensor([1, 2, 3, 4]).send(bob)
    y = x * 2
    op1 = bob._get_msg(-1).action

    a = torch.tensor([[1, 2], [3, 4]]).send(bob)
    b = a.sum(1, keepdim=True)
    op2 = bob._get_msg(-1).action

    bob.log_msgs = False

    def compare(detailed, original):
        assert type(detailed) == syft.execution.computation.ComputationAction

        detailed_msg = (
            detailed.name,
            detailed.target,
            detailed.args,
            detailed.kwargs,
            detailed.return_ids,
            detailed.return_value,
        )

        original_msg = (
            original.name,
            original.target,
            original.args,
            original.kwargs,
            original.return_ids,
            original.return_value,
        )

        for i in range(len(original_msg)):
            if type(original_msg[i]) != torch.Tensor:
                assert detailed_msg[i] == original_msg[i], f"{detailed_msg[i]} != {original_msg[i]}"
            else:
                assert detailed_msg[i].equal(
                    original_msg[i]
                ), f"{detailed_msg[i]} != {original_msg[i]}"

        return True

    return [
        {
            "value": op1,
            "simplified": (
                CODE[syft.execution.computation.ComputationAction],
                (
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], op1.name),
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], op1.target),
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], op1.args),
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], op1.kwargs),
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], op1.return_ids),
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], op1.return_value),
                ),
            ),
            "cmp_detailed": compare,
        },
        {
            "value": op2,
            "simplified": (
                CODE[syft.execution.computation.ComputationAction],
                (
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], op2.name),
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], op2.target),
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], op2.args),
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], op2.kwargs),
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], op2.return_ids),
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], op2.return_value),
                ),
            ),
            "cmp_detailed": compare,
        },
    ]


# syft.messaging.message.TensorCommandMessage
def make_tensor_command_message(**kwargs):
    bob = kwargs["workers"]["bob"]
    alice = kwargs["workers"]["alice"]
    bob.log_msgs = True

    x = torch.tensor([1, 2, 3, 4]).send(bob)
    y = x * 2
    cmd1 = bob._get_msg(-1)

    a = torch.tensor([[1, 2], [3, 4]]).send(bob)
    b = a.sum(1, keepdim=True)
    cmd2 = bob._get_msg(-1)

    x = torch.tensor([1, 2, 3, 4]).send(bob)
    x.remote_send(alice)
    cmd3 = bob._get_msg(-1)

    bob.log_msgs = False

    def compare(detailed, original):
        detailed_action = detailed.action
        original_action = original.action

        detailed_action = (
            detailed.name,
            detailed.target,
            detailed.args,
            detailed.kwargs,
            detailed.return_ids,
            detailed.return_value,
        )

        original_action = (
            original.name,
            original.target,
            original.args,
            original.kwargs,
            original.return_ids,
            original.return_value,
        )

        for i in range(len(original_action)):
            if type(original_action[i]) != torch.Tensor:
                assert detailed_action[i] == original_action[i]
            else:
                assert detailed_action[i].equal(original_action[i])

        return True

    return [
        {
            "value": cmd1,
            "simplified": (
                CODE[syft.messaging.message.TensorCommandMessage],
                (
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], cmd1.action),
                ),  # (Any) message
            ),
            "cmp_detailed": compare,
        },
        {
            "value": cmd2,
            "simplified": (
                CODE[syft.messaging.message.TensorCommandMessage],
                (
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], cmd2.action),
                ),  # (Any) message
            ),
            "cmp_detailed": compare,
        },
        {
            "value": cmd3,
            "simplified": (
                CODE[syft.messaging.message.TensorCommandMessage],
                (msgpack.serde._simplify(kwargs["workers"]["serde_worker"], cmd3.action),),
            ),
            "cmp_detailed": compare,
        },
    ]


# syft.messaging.message.ObjectMessage
def make_objectmessage(**kwargs):
    bob = kwargs["workers"]["bob"]
    bob.log_msgs = True
    x = torch.tensor([1, 2, 3, 4]).send(bob)
    obj = bob._get_msg(-1)
    bob.log_msgs = False

    def compare(detailed, original):
        assert type(detailed) == syft.messaging.message.ObjectMessage
        # torch tensors
        assert detailed.object.equal(original.object)
        return True

    return [
        {
            "value": obj,
            "simplified": (
                CODE[syft.messaging.message.ObjectMessage],
                (
                    msgpack.serde._simplify(
                        kwargs["workers"]["serde_worker"], obj.object
                    ),  # (Any) simplified object
                ),
            ),
            "cmp_detailed": compare,
        }
    ]


# ObjectRequestMessage
def make_objectrequestmessage(**kwargs):
    bob = kwargs["workers"]["bob"]
    bob.log_msgs = True
    x = torch.tensor([1, 2, 3, 4]).send(bob)
    x.get()
    obj_req = bob._get_msg(-1)
    bob.log_msgs = False

    def compare(detailed, original):
        assert type(detailed) == syft.messaging.message.ObjectRequestMessage
        assert detailed.object_id == original.object_id
        assert detailed.user == original.user
        assert detailed.reason == original.reason
        return True

    return [
        {
            "value": obj_req,
            "simplified": (
                CODE[syft.messaging.message.ObjectRequestMessage],
                (
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], obj_req.object_id),
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], obj_req.user),
                    msgpack.serde._simplify(kwargs["workers"]["serde_worker"], obj_req.reason),
                ),
            ),
            "cmp_detailed": compare,
        }
    ]


# IsNoneMessage
def make_isnonemessage(**kwargs):
    bob = kwargs["workers"]["bob"]
    bob.log_msgs = True
    t = torch.tensor([1, 2, 3, 4])
    x = t.send(bob)
    x.child.is_none()
    nm = bob._get_msg(-1)
    bob.log_msgs = False

    def compare(detailed, original):
        assert type(detailed) == syft.messaging.message.IsNoneMessage
        # torch tensors
        assert detailed.object_id == original.object_id
        return True

    return [
        {
            "value": nm,
            "simplified": (
                CODE[syft.messaging.message.IsNoneMessage],
                (msgpack.serde._simplify(kwargs["workers"]["serde_worker"], nm.object_id),),
            ),
            "cmp_detailed": compare,
        }
    ]


# GetShapeMessage
def make_getshapemessage(**kwargs):
    bob = kwargs["workers"]["bob"]
    bob.log_msgs = True
    t = torch.tensor([1, 2, 3, 4])
    x = t.send(bob)
    z = x + x
    s = z.shape
    shape_message = bob._get_msg(-1)
    bob.log_msgs = False

    def compare(detailed, original):
        assert type(detailed) == syft.messaging.message.GetShapeMessage
        # torch tensor
        assert detailed.tensor_id == original.tensor_id
        return True

    return [
        {
            "value": shape_message,
            "simplified": (
                CODE[syft.messaging.message.GetShapeMessage],
                (
                    msgpack.serde._simplify(
                        kwargs["workers"]["serde_worker"], shape_message.tensor_id
                    ),  # (Any) simplified tensor
                ),
            ),
            "cmp_detailed": compare,
        }
    ]


# ForceObjectDeleteMessage
def make_forceobjectdeletemessage(**kwargs):
    bob = kwargs["workers"]["bob"]
    bob.log_msgs = True
    t = torch.tensor([1, 2, 3, 4])
    id = t.id
    x = t.send(bob)
    del x
    del_message = bob._get_msg(-1)
    bob.log_msgs = False

    def compare(detailed, original):
        assert type(detailed) == syft.messaging.message.ForceObjectDeleteMessage
        assert detailed.object_id == original.object_id
        return True

    return [
        {
            "value": del_message,
            "simplified": (
                CODE[syft.messaging.message.ForceObjectDeleteMessage],
                (id,),  # (int) id
            ),
            "cmp_detailed": compare,
        }
    ]


# SearchMessage
def make_searchmessage(**kwargs):
    search_message = syft.messaging.message.SearchMessage([1, "test", 3])

    def compare(detailed, original):
        assert type(detailed) == syft.messaging.message.SearchMessage
        assert detailed.query == original.query
        return True

    return [
        {
            "value": search_message,
            "simplified": (
                CODE[syft.messaging.message.SearchMessage],
                ((CODE[list], (1, (CODE[str], (b"test",)), 3)),),  # (Any) message
            ),
            "cmp_detailed": compare,
        }
    ]


# PlanCommandMessage
def make_plancommandmessage(**kwargs):
    bob = kwargs["workers"]["bob"]
    bob.log_msgs = True

    @syft.func2plan(args_shape=[(1,)])
    def plan(data):
        return data * 3

    plan.send(bob)
    plan.owner.fetch_plan(plan.id, bob)
    fetch_plan_cmd = bob._get_msg(-1)
    bob.log_msgs = False

    def compare(detailed, original):
        assert type(detailed) == syft.messaging.message.PlanCommandMessage
        assert detailed.command_name == original.command_name
        assert detailed.args == original.args
        return True

    return [
        {
            "value": fetch_plan_cmd,
            "simplified": (
                CODE[syft.messaging.message.PlanCommandMessage],
                (
                    (CODE[str], (b"fetch_plan",)),  # (str) command
                    (CODE[tuple], (plan.id, False)),  # (tuple) args
                ),
            ),
            "cmp_detailed": compare,
        }
    ]


# WorkerCommandMessage
def make_workercommandmessage(**kwargs):
    server, remote_proxy = kwargs["start_remote_worker"](
        id=kwargs["id"], hook=kwargs["hook"], port=kwargs["port"]
    )

    remote_proxy._log_msgs_remote(value=True)
    nr_objects = remote_proxy.tensors_count_remote()
    assert nr_objects == 0

    objects_count_msg = remote_proxy._get_msg_remote(
        index=-2
    )  # index -2 as last message is _get_msg message

    remote_proxy.close()
    server.terminate()

    def compare(detailed, original):
        assert type(detailed) == syft.messaging.message.WorkerCommandMessage
        return True

    return [
        {
            "value": objects_count_msg,
            "simplified": (
                CODE[syft.messaging.message.WorkerCommandMessage],
                (
                    (CODE[str], (b"tensors_count",)),  # (str) command
                    (CODE[tuple], ((CODE[tuple], ()), (CODE[dict], ()), (CODE[list], ()))),
                ),
            ),
            "cmp_detailed": compare,
        }
    ]


# syft.exceptions.GetNotPermittedError
def make_getnotpermittederror(**kwargs):
    try:
        raise syft.exceptions.GetNotPermittedError()
    except syft.exceptions.GetNotPermittedError as e:
        err = e

    def compare(detailed, original):
        assert type(detailed) == syft.exceptions.GetNotPermittedError
        assert (
            traceback.format_tb(detailed.__traceback__)[-1]
            == traceback.format_tb(original.__traceback__)[-1]
        )
        return True

    return [
        {
            "value": err,
            "simplified": (
                CODE[syft.exceptions.GetNotPermittedError],
                (
                    (CODE[str], (b"GetNotPermittedError",)),  # (str) __name__
                    msgpack.serde._simplify(
                        kwargs["workers"]["serde_worker"],
                        "Traceback (most recent call last):\n"
                        + "".join(traceback.format_tb(err.__traceback__)),
                    ),  # (str) traceback
                    (CODE[dict], tuple()),  # (dict) attributes
                ),
            ),
            "cmp_detailed": compare,
        }
    ]


# syft.exceptions.ResponseSignatureError
def make_responsesignatureerror(**kwargs):
    try:
        raise syft.exceptions.ResponseSignatureError()
    except syft.exceptions.ResponseSignatureError as e:
        err = e

    def compare(detailed, original):
        assert type(detailed) == syft.exceptions.ResponseSignatureError
        assert (
            traceback.format_tb(detailed.__traceback__)[-1]
            == traceback.format_tb(original.__traceback__)[-1]
        )
        assert detailed.get_attributes() == original.get_attributes()
        return True

    return [
        {
            "value": err,
            "simplified": (
                CODE[syft.exceptions.ResponseSignatureError],
                (
                    (CODE[str], (b"ResponseSignatureError",)),  # (str) __name__
                    msgpack.serde._simplify(
                        kwargs["workers"]["serde_worker"],
                        "Traceback (most recent call last):\n"
                        + "".join(traceback.format_tb(err.__traceback__)),
                    ),  # (str) traceback
                    msgpack.serde._simplify(
                        kwargs["workers"]["serde_worker"], err.get_attributes()
                    ),  # (dict) attributes
                ),
            ),
            "cmp_detailed": compare,
        }
    ]


# syft.frameworks.torch.tensors.interpreters.gradients_core.GradFunc
def make_gradfn(**kwargs):
    alice, bob = kwargs["workers"]["alice"], kwargs["workers"]["bob"]
    t = torch.tensor([1, 2, 3])

    x_share = t.share(alice, bob, requires_grad=True)
    y_share = t.share(alice, bob, requires_grad=True)
    z_share = x_share + y_share  # AddBackward

    # This is bad. We should find something robust
    x_share.child.child.set_garbage_collect_data(False)
    y_share.child.child.set_garbage_collect_data(False)

    grad_fn = z_share.child.grad_fn

    def compare(detailed, original):
        assert isinstance(
            detailed, syft.frameworks.torch.tensors.interpreters.gradients_core.GradFunc
        )
        assert detailed.__class__.__name__ == original.__class__.__name__

        # This block only works only for syft tensor attributes
        for detailed_attr, original_attr in zip(detailed._attributes, original._attributes):
            assert detailed_attr.__class__.__name__ == original_attr.__class__.__name__
            assert detailed_attr.get().equal(t)

        return True

    return [
        {
            "value": grad_fn,
            "simplified": (
                CODE[syft.frameworks.torch.tensors.interpreters.gradients_core.GradFunc],
                (
                    CODE[list],
                    (
                        (CODE[str], (b"AddBackward",)),
                        msgpack.serde._simplify(kwargs["workers"]["serde_worker"], x_share.child),
                        msgpack.serde._simplify(kwargs["workers"]["serde_worker"], y_share.child),
                    ),
                ),
            ),
            "cmp_detailed": compare,
        }
    ]


def make_paillier(**kwargs):
    # TODO: Add proper testing for paillier tensor

    def compare(original, detailed):
        return True

    tensor = syft.frameworks.torch.tensors.interpreters.paillier.PaillierTensor()
    simplfied = syft.frameworks.torch.tensors.interpreters.paillier.PaillierTensor.simplify(
        kwargs["workers"]["serde_worker"], tensor
    )

    return [
        {
            "value": tensor,
            "simplified": (
                CODE[syft.frameworks.torch.tensors.interpreters.paillier.PaillierTensor],
                simplfied,
            ),
            "cmp_detailed": compare,
        }
    ]
