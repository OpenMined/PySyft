from collections import OrderedDict
import pytest
import numpy
import torch
from functools import partial
import traceback
import io

import syft
from syft.serde import msgpack

# Make dict of type codes
CODE = OrderedDict()
for cls, simplifier in msgpack.serde.simplifiers.items():
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
    # NOTE: after serde TopLevelTracedModule or _C.Function become ScriptModule
    # (that's what torch.jit.load returns in detail function)
    assert isinstance(detailed, torch.jit.ScriptModule)
    # Code changes after torch.jit.load(): function becomes `forward` method
    if type(original) != torch._C.Function:
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


# torch._C.Function
def make_torch_cfunction(**kwargs):
    @torch.jit.script
    def func(x):  # pragma: no cover
        return x + 2

    return [
        {
            "value": func,
            "simplified": (
                CODE[torch._C.Function],
                (func.save_to_buffer(),),  # (bytes) serialized torchscript
            ),
            "cmp_detailed": compare_modules,
        }
    ]


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
                    msgpack.serde._simplify(syft.hook.local_worker, param.data),  # (Tensor) data
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
                    ast.field,  # (int) field
                    (CODE[str], (ast.crypto_provider.id.encode("utf-8"),)),  # (str) worker_id
                    msgpack.serde._simplify(
                        syft.hook.local_worker, ast.child
                    ),  # (dict of AbstractTensor) simplified chain
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
    msgpack.serde._simplify(syft.hook.local_worker, fpt)

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
                    fpt.field,  # (int) field
                    12,  # (int) base
                    5,  # (int) precision_fractional
                    fpt.kappa,  # (int) kappa
                    (CODE[set], ((CODE[str], (b"tag1",)),)),  # (set of str) tags
                    (CODE[str], (b"desc",)),  # (str) description
                    msgpack.serde._simplify(
                        syft.hook.local_worker, fpt.child
                    ),  # (AbstractTensor) chain
                ),
            ),
            "cmp_detailed": compare,
        }
    ]


# CRTPrecisionTensor
def make_crtprecisiontensor(**kwargs):
    workers = kwargs["workers"]
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]
    t = torch.tensor([[3.1, 4.3]])
    cpt = t.fix_prec(storage="crt").share(alice, bob, crypto_provider=james).child
    # AdditiveSharingTensor.simplify sets garbage_collect_data=False on child tensors during simplify
    # This changes tensors' internal state in chain and is required to pass the test
    msgpack.serde._simplify(syft.hook.local_worker, cpt)

    def compare(detailed, original):
        assert (
            type(detailed)
            == syft.frameworks.torch.tensors.interpreters.crt_precision.CRTPrecisionTensor
        )
        assert detailed.id == original.id
        assert detailed.base == original.base
        assert detailed.precision_fractional == original.precision_fractional
        return True

    return [
        {
            "value": cpt,
            "simplified": (
                CODE[syft.frameworks.torch.tensors.interpreters.crt_precision.CRTPrecisionTensor],
                (
                    cpt.id,  # (int) id
                    cpt.base,  # (int) base
                    cpt.precision_fractional,  # (int) precision_fractional
                    msgpack.serde._simplify(
                        syft.hook.local_worker, cpt.child
                    ),  # (dict of AbstractTensor) simplified chain
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
                        syft.hook.local_worker, lt.child
                    ),  # (AbstractTensor) chain
                ),
            ),
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
                    msgpack.serde._simplify(syft.hook.local_worker, mpt.child),  # (dict)
                ),
            ),
            "cmp_detailed": compare,
        }
    ]


# syft.messaging.plan.plan.Plan
def make_plan(**kwargs):
    # Function to plan
    @syft.func2plan([torch.Size((3,))])
    def plan(x):
        x = x + x
        x = torch.abs(x)
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

    with syft.hook.local_worker.registration_enabled():
        model_plan = Net()
        model_plan.build(torch.tensor([1.0, 2.0, 3.0]))

    def compare(detailed, original):
        assert type(detailed) == syft.messaging.plan.plan.Plan
        assert detailed.id == original.id
        # Procedure
        assert detailed.procedure.operations == original.procedure.operations
        assert detailed.procedure.arg_ids == original.procedure.arg_ids
        assert detailed.procedure.result_ids == original.procedure.result_ids
        # States for the nested plans
        assert detailed.nested_states == original.nested_states
        # State
        assert detailed.state.state_ids == original.state.state_ids
        assert detailed.include_state == original.include_state
        assert detailed.is_built == original.is_built
        assert detailed.input_shapes == original.input_shapes
        assert detailed._output_shape == original._output_shape
        assert detailed.name == original.name
        assert detailed.tags == original.tags
        assert detailed.description == original.description
        with syft.hook.local_worker.registration_enabled():
            t = torch.tensor([1.1, -2, 3])
            res1 = detailed(t)
            res2 = original(t)
        assert res1.equal(res2)
        return True

    return [
        {
            "value": plan,
            "simplified": (
                CODE[syft.messaging.plan.plan.Plan],
                (
                    plan.id,  # (int or str) id
                    msgpack.serde._simplify(syft.hook.local_worker, plan.procedure),  # (Procedure)
                    msgpack.serde._simplify(syft.hook.local_worker, plan.state),  # (State)
                    plan.include_state,  # (bool) include_state
                    plan.is_built,  # (bool) is_built
                    (CODE[list], ((CODE[torch.Size], (3,)),)),  # (list of torch.Size) input_shapes
                    # NOTE: it's uninitialized until plan.output_shape property is used
                    None,  # (torch.Size) _output_shape
                    msgpack.serde._simplify(syft.hook.local_worker, plan.name),  # (str) name
                    msgpack.serde._simplify(syft.hook.local_worker, plan.tags),  # (set of str) tags
                    msgpack.serde._simplify(
                        syft.hook.local_worker, plan.description
                    ),  # (str) description
                    msgpack.serde._simplify(syft.hook.local_worker, []),  # (list of State)
                ),
            ),
            "cmp_detailed": compare,
        },
        {
            "value": model_plan,
            "simplified": (
                CODE[syft.messaging.plan.plan.Plan],
                (
                    model_plan.id,  # (int or str) id
                    msgpack.serde._simplify(
                        syft.hook.local_worker, model_plan.procedure
                    ),  # (Procedure)
                    msgpack.serde._simplify(syft.hook.local_worker, model_plan.state),  # (State)
                    model_plan.include_state,  # (bool) include_state
                    model_plan.is_built,  # (bool) is_built
                    (CODE[list], ((CODE[torch.Size], (3,)),)),  # (list of torch.Size) input_shapes
                    # NOTE: it's uninitialized until plan.output_shape property is used
                    None,  # (torch.Size) _output_shape
                    msgpack.serde._simplify(syft.hook.local_worker, model_plan.name),  # (str) name
                    msgpack.serde._simplify(syft.hook.local_worker, model_plan.tags),  # (list) tags
                    msgpack.serde._simplify(
                        syft.hook.local_worker, model_plan.description
                    ),  # (str) description
                    msgpack.serde._simplify(syft.hook.local_worker, []),  # (list of State)
                ),
            ),
            "cmp_detailed": compare,
        },
    ]


# State
def make_state(**kwargs):
    me = kwargs["workers"]["me"]

    with me.registration_enabled():
        t1, t2 = torch.randn(3, 3), torch.randn(3, 3)
        me.register_obj(t1), me.register_obj(t2)
        state = syft.messaging.plan.state.State(owner=me, state_ids=[t1.id, t2.id])

    def compare(detailed, original):
        assert type(detailed) == syft.messaging.plan.state.State
        assert detailed.state_ids == original.state_ids
        for i in range(len(original.tensors())):
            assert detailed.tensors()[i].equal(original.tensors()[i])
        return True

    return [
        {
            "value": state,
            "simplified": (
                CODE[syft.messaging.plan.state.State],
                (
                    (CODE[list], (t1.id, t2.id)),  # (list) state_ids
                    (
                        CODE[list],
                        (  # (list) tensors
                            msgpack.serde._simplify(syft.hook.local_worker, t1),
                            msgpack.serde._simplify(syft.hook.local_worker, t2),
                        ),
                    ),
                ),
            ),
            "cmp_detailed": compare,
        }
    ]


# Procedure
def make_procedure(**kwargs):
    @syft.func2plan([torch.Size((1, 3))])
    def plan(x):
        x = x + x
        x = torch.abs(x)
        return x

    procedure = plan.procedure

    def compare(detailed, original):
        assert type(detailed) == syft.messaging.plan.procedure.Procedure
        assert detailed.arg_ids == original.arg_ids
        assert detailed.result_ids == original.result_ids
        assert detailed.operations == original.operations
        assert detailed.promise_out_id == original.promise_out_id
        return True

    return [
        {
            "value": procedure,
            "simplified": (
                CODE[syft.messaging.plan.procedure.Procedure],
                (
                    (
                        CODE[list],
                        (  # (list of Operation) operations
                            procedure.operations[0],  # (unsimplified Operation)
                            procedure.operations[1],
                        ),
                    ),
                    (CODE[tuple], (procedure.arg_ids[0],)),  # (tuple) arg_ids
                    (CODE[tuple], (procedure.result_ids[0],)),  # (tuple) result_ids
                    None,  # (int) promise_out_id
                ),
            ),
            "cmp_detailed": compare,
        }
    ]


# Protocol
def make_protocol(**kwargs):
    me = kwargs["workers"]["me"]

    @syft.func2plan([torch.Size((1, 3))])
    def plan(x):
        x = x + x
        x = torch.abs(x)
        return x

    with me.registration_enabled():
        me.register_obj(plan)

    protocol = syft.messaging.protocol.Protocol(
        [("me", plan), ("me", plan)], tags=["aaa", "bbb"], description="desc"
    )

    def compare(detailed, original):
        assert type(detailed) == syft.messaging.protocol.Protocol
        assert detailed.id == original.id
        assert detailed.tags == original.tags
        assert detailed.description == original.description
        assert detailed.plans == original.plans
        assert detailed.owner == original.owner
        assert detailed.workers_resolved == original.workers_resolved
        return True

    return [
        {
            "value": protocol,
            "simplified": (
                CODE[syft.messaging.protocol.Protocol],
                (
                    protocol.id,  # (int)
                    (
                        CODE[list],
                        ((CODE[str], (b"aaa",)), (CODE[str], (b"bbb",))),
                    ),  # (list of strings) tags
                    (CODE[str], (b"desc",)),  # (str) description
                    (
                        CODE[list],  # (list) plans reference
                        (
                            # (tuple) reference: worker_id (str), plan_id (int)
                            (CODE[tuple], ((CODE[str], (b"me",)), plan.id)),
                            (CODE[tuple], ((CODE[str], (b"me",)), plan.id)),
                        ),
                    ),
                    False,  # (bool) workers_resolved
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
                    False,  # (bool) garbage_collect_data
                ),
            ),
            "cmp_detailed": compare,
        }
    ]


# syft.generic.pointers.pointer_protocol.PointerProtocol
def make_pointerprotocol(**kwargs):
    alice, me = kwargs["workers"]["alice"], kwargs["workers"]["me"]

    @syft.func2plan([torch.Size((1, 3))])
    def plan(x):
        x = x + x
        x = torch.abs(x)
        return x

    protocol = syft.messaging.protocol.Protocol(
        [("worker1", plan), ("worker2", plan)], tags=["aaa", "bbb"], description="desc"
    )
    protocol.send(alice)
    ptr = me.request_search([protocol.id], location=alice)[0]

    def compare(detailed, original):
        assert type(detailed) == syft.generic.pointers.pointer_protocol.PointerProtocol
        assert detailed.id == original.id
        assert detailed.id_at_location == original.id_at_location
        assert detailed.location == original.location
        assert detailed.garbage_collect_data == original.garbage_collect_data
        return True

    return [
        {
            "value": ptr,
            "simplified": (
                CODE[syft.generic.pointers.pointer_protocol.PointerProtocol],
                (
                    ptr.id,  # (int or str) id
                    ptr.id_at_location,  # (int) id_at_location
                    (CODE[str], (b"alice",)),  # (str) location.id
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
                    msgpack.serde._simplify(syft.hook.local_worker, obj),  # (Any) obj
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


# syft.workers.base.BaseWorker
def make_baseworker(**kwargs):
    bob = kwargs["workers"]["bob"]
    t = torch.rand(3, 3)
    with bob.registration_enabled():
        bob.register_obj(t)

    def compare(detailed, original):
        assert isinstance(detailed, syft.workers.base.BaseWorker)
        assert detailed.id == original.id
        return True

    return [
        {
            "value": bob,
            "simplified": (
                CODE[syft.workers.base.BaseWorker],
                ((CODE[str], (b"bob",)),),  # id (str)
            ),
            "cmp_detailed": compare,
        },
        # Forced simplification
        {
            "forced": True,
            "value": bob,
            "simplified": (
                FORCED_CODE[syft.workers.base.BaseWorker],
                (
                    (CODE[str], (b"bob",)),  # id (str)
                    msgpack.serde._simplify(
                        syft.hook.local_worker, bob._objects
                    ),  # (dict) _objects
                    True,  # (bool) auto_add
                ),
            ),
            "cmp_detailed": compare,
        },
    ]


# syft.frameworks.torch.tensors.interpreters.autograd.AutogradTensor
def make_autogradtensor(**kwargs):
    t = torch.tensor([1, 2, 3])
    agt = syft.frameworks.torch.tensors.interpreters.autograd.AutogradTensor().on(t).child
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
                    None,  # owner
                    agt.id,  # (int)
                    msgpack.serde._simplify(
                        syft.hook.local_worker, agt.child
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
                    msgpack.serde._simplify(syft.hook.local_worker, t),  # (AbstractTensor) chain
                ),
            ),
            "cmp_detailed": compare,
        }
    ]


# syft.frameworks.torch.tensors.interpreters.promise.PromiseTensor
def make_promisetensor(**kwargs):
    pt = syft.Promise.FloatTensor(shape=torch.Size((3, 4)))
    pt.tag("tag1")
    pt.describe("I promise")

    @syft.func2plan(args_shape=[(3, 4)])
    def promising_plan(x):
        x = x * 2 + 1
        return x

    # A plan should be added to promise tensor's plans list here
    res = promising_plan(pt)
    pt = pt.child

    def compare(detailed, original):
        assert type(detailed) == syft.frameworks.torch.tensors.interpreters.promise.PromiseTensor
        assert detailed.id == original.id
        assert detailed.shape == original.shape
        assert detailed.obj_type == original.obj_type
        assert detailed.plans == original.plans
        assert detailed.tags == original.tags
        assert detailed.description == original.description
        return True

    return [
        {
            "value": pt,
            "simplified": (
                CODE[syft.frameworks.torch.tensors.interpreters.promise.PromiseTensor],
                (
                    pt.id,  # (int) id
                    (CODE[torch.Size], (3, 4)),  # (torch.Size) shape
                    (CODE[str], (b"torch.FloatTensor",)),  # (str) obj_type (torch tensor type)
                    (CODE[set], (list(pt.plans)[0],)),  # (set of Plans' id) plans
                    (CODE[set], ((CODE[str], (b"tag1",)),)),  # (set of str) tags
                    (CODE[str], (b"I promise",)),  # (str) description
                ),
            ),
            "cmp_detailed": compare,
        }
    ]


# Message
def make_message(**kwargs):
    def compare(detailed, original):
        assert type(detailed) == syft.messaging.message.Message
        assert detailed.contents == original.contents
        return True

    return [
        {
            "value": syft.messaging.message.Message([1, 2, 3]),
            "simplified": (
                CODE[syft.messaging.message.Message],
                ((CODE[list], (1, 2, 3)),),  # (Any) simplified content
            ),
            "cmp_detailed": compare,
        },
        {
            "value": syft.messaging.message.Message((1, 2, 3)),
            "simplified": (
                CODE[syft.messaging.message.Message],
                ((CODE[tuple], (1, 2, 3)),),  # (Any) simplified content
            ),
            "cmp_detailed": compare,
        },
    ]


# syft.messaging.message.Operation
def make_operation(**kwargs):
    bob = kwargs["workers"]["bob"]
    bob.log_msgs = True

    x = torch.tensor([1, 2, 3, 4]).send(bob)
    y = x * 2
    op1 = bob._get_msg(-1)

    a = torch.tensor([[1, 2], [3, 4]]).send(bob)
    b = a.sum(1, keepdim=True)
    op2 = bob._get_msg(-1)

    bob.log_msgs = False

    def compare(detailed, original):
        detailed_msg = (
            detailed.cmd_name,
            detailed.cmd_owner,
            detailed.cmd_args,
            detailed.cmd_kwargs,
        )
        original_msg = (
            original.cmd_name,
            original.cmd_owner,
            original.cmd_args,
            original.cmd_kwargs,
        )
        assert type(detailed) == syft.messaging.message.Operation
        for i in range(len(original_msg)):
            if type(original_msg[i]) != torch.Tensor:
                assert detailed_msg[i] == original_msg[i]
            else:
                assert detailed_msg[i].equal(original_msg[i])
        assert detailed.return_ids == original.return_ids
        return True

    message1 = (op1.cmd_name, op1.cmd_owner, op1.cmd_args, op1.cmd_kwargs)
    message2 = (op2.cmd_name, op2.cmd_owner, op2.cmd_args, op2.cmd_kwargs)

    return [
        {
            "value": op1,
            "simplified": (
                CODE[syft.messaging.message.Operation],
                (
                    msgpack.serde._simplify(syft.hook.local_worker, message1),  # (Any) message
                    (CODE[tuple], (op1.return_ids[0],)),  # (tuple) return_ids
                ),
            ),
            "cmp_detailed": compare,
        },
        {
            "value": op2,
            "simplified": (
                CODE[syft.messaging.message.Operation],
                (
                    msgpack.serde._simplify(syft.hook.local_worker, message2),  # (Any) message
                    (CODE[tuple], (op2.return_ids[0],)),  # (tuple) return_ids
                ),
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
        assert detailed.contents.equal(original.contents)
        return True

    return [
        {
            "value": obj,
            "simplified": (
                CODE[syft.messaging.message.ObjectMessage],
                (
                    msgpack.serde._simplify(
                        syft.hook.local_worker, obj.contents
                    ),  # (Any) simplified contents
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
        assert detailed.contents == original.contents
        return True

    return [
        {
            "value": obj_req,
            "simplified": (
                CODE[syft.messaging.message.ObjectRequestMessage],
                (
                    msgpack.serde._simplify(
                        syft.hook.local_worker, obj_req.contents
                    ),  # (Any) simplified contents
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
        assert detailed.contents.equal(original.contents)
        return True

    return [
        {
            "value": nm,
            "simplified": (
                CODE[syft.messaging.message.IsNoneMessage],
                (
                    msgpack.serde._simplify(
                        syft.hook.local_worker, nm.contents
                    ),  # (Any) simplified contents
                ),
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
        assert detailed.contents.equal(original.contents)
        return True

    return [
        {
            "value": shape_message,
            "simplified": (
                CODE[syft.messaging.message.GetShapeMessage],
                (
                    msgpack.serde._simplify(
                        syft.hook.local_worker, shape_message.contents
                    ),  # (Any) simplified contents
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
        assert detailed.contents == original.contents
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
        assert detailed.contents == original.contents
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
        assert detailed.contents == original.contents
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
                        syft.hook.local_worker,
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
                        syft.hook.local_worker,
                        "Traceback (most recent call last):\n"
                        + "".join(traceback.format_tb(err.__traceback__)),
                    ),  # (str) traceback
                    msgpack.serde._simplify(
                        syft.hook.local_worker, err.get_attributes()
                    ),  # (dict) attributes
                ),
            ),
            "cmp_detailed": compare,
        }
    ]
