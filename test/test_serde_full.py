from collections import OrderedDict
import pytest
import io
import numpy
import torch
from pip._internal import req

import syft
from syft.serde import serde
from syft import codes

# Make dict of type codes
CODE = OrderedDict()
for cls, simplifier in serde.simplifiers.items():
    CODE[cls] = simplifier[0]


########################################################################
# Functions that return list of serde samples in the following format:
# [
#   tuple(
#    original_value,
#    simplified_value,
#    custom_simplified_values_comparison_function, # optional
#    custom_detailed_values_comparison_function, # optional
#   ),
#   ...
# ]
########################################################################

########################################################################
# Native types.
########################################################################

# Dict.
def make_dict(**kwargs):
    return [
        (
            {1: "hello", 2: "world"},
            (CODE[dict], ((1, (CODE[str], (b"hello",))), (2, (CODE[str], (b"world",))))),
        ),
        ({"hello": "world"}, (CODE[dict], (((CODE[str], (b"hello",)), (CODE[str], (b"world",))),))),
        ({}, (CODE[dict], ())),
    ]


# List.
def make_list(**kwargs):
    return [
        (["hello", "world"], (CODE[list], ((CODE[str], (b"hello",)), (CODE[str], (b"world",))))),
        (["hello"], (CODE[list], ((CODE[str], (b"hello",)),))),
        ([], (CODE[list], ())),
    ]


# Tuple.
def make_tuple(**kwargs):
    return [
        (("hello", "world"), (CODE[tuple], ((CODE[str], (b"hello",)), (CODE[str], (b"world",))))),
        (("hello",), (CODE[tuple], ((CODE[str], (b"hello",)),))),
        (tuple(), (CODE[tuple], ())),
    ]


# Set.
def make_set(**kwargs):
    return [
        (
            {"hello", "world"},
            (CODE[set], ((CODE[str], (b"world",)), (CODE[str], (b"hello",)))),
            # Compare tuples as sets because set order is undefined
            lambda simplified, expected: simplified[0] == expected[0]
            and set(simplified[1]) == set(simplified[1]),
        ),
        ({"hello"}, (CODE[set], ((CODE[str], (b"hello",)),))),
        (set([]), (CODE[set], ())),
    ]


# Slice.
def make_slice(**kwargs):
    return [
        (slice(10, 20, 30), (CODE[slice], (10, 20, 30))),
        (slice(10, 20), (CODE[slice], (10, 20, None))),
        (slice(10), (CODE[slice], (None, 10, None))),
    ]


# Range.
def make_range(**kwargs):
    return [(range(1, 3, 4), (CODE[range], (1, 3, 4))), (range(1, 3), (CODE[range], (1, 3, 1)))]


# String.
def make_str(**kwargs):
    return [("a string", (CODE[str], (b"a string",))), ("", (CODE[str], (b"",)))]


# Int.
def make_int(**kwargs):
    return [(5, 5)]


# Float.
def make_float(**kwargs):
    return [(5.1, 5.1)]


# Ellipsis.
def make_ellipsis(**kwargs):
    return [(..., (CODE[type(Ellipsis)], b""))]


########################################################################
# Numpy.
########################################################################

# numpy.ndarray
def make_numpy_ndarray(**kwargs):
    np_array = numpy.random.random((2, 2))
    return [
        (
            np_array,
            (CODE[type(np_array)], (np_array.tobytes(), np_array.shape, np_array.dtype.name)),
            None,
            lambda detailed, original: numpy.array_equal(detailed, original),
        )
    ]


########################################################################
# PyTorch.
########################################################################

# Utility functions.


def compare_modules(detailed, original):
    """Compare ScriptModule instances"""
    input = torch.randn(10, 3)
    # NOTE: after serde TopLevelTracedModule becomes ScriptModule (that's what torch.jit.load returns)
    assert isinstance(detailed, torch.jit.ScriptModule)
    assert detailed.code == original.code
    # model outputs match
    assert detailed(input).equal(original(input))
    return True


def save_to_buffer(tensor) -> bin:
    binary_stream = io.BytesIO()
    torch.save(tensor, binary_stream)
    return binary_stream.getvalue()


# torch.device
def make_torch_device(**kwargs):
    torch_device = torch.device("cpu")
    return [(torch_device, (CODE[type(torch_device)], "cpu"))]


# torch.jit.ScriptModule
def make_torch_scriptmodule(**kwargs):
    class ScriptModule(torch.jit.ScriptModule):
        def __init__(self):
            super(ScriptModule, self).__init__()

        @torch.jit.script_method
        def forward(self, x):
            return x + 2

    sm = ScriptModule()
    return [(sm, (CODE[torch.jit.ScriptModule], sm.save_to_buffer()), None, compare_modules)]


# torch.jit.TopLevelTracedModule
def make_torch_topleveltracedmodule(**kwargs):
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.w1 = torch.nn.Parameter(torch.randn(3, 1), requires_grad=True)
            self.b1 = torch.nn.Parameter(torch.randn(1), requires_grad=True)

        def forward(self, x):
            x = x @ self.w1 + self.b1
            return x

    model = Model()
    tm = torch.jit.trace(model, torch.randn(10, 3))
    return [
        (tm, (CODE[torch.jit.TopLevelTracedModule], tm.save_to_buffer()), None, compare_modules)
    ]


# torch.nn.parameter.Parameter
def make_torch_parameter(**kwargs):
    param = torch.nn.Parameter(torch.randn(3, 3))

    def compare(detailed, original):
        assert type(detailed) == torch.nn.Parameter
        assert detailed.data.equal(original.data)
        assert detailed.id == original.id
        assert detailed.requires_grad == original.requires_grad
        return True

    return [
        (
            param,
            (
                CODE[torch.nn.Parameter],
                (param.id, serde._simplify(param.data), param.requires_grad, None),
            ),
            None,
            compare,
        )
    ]


# torch.Tensor
def make_torch_tensor(**kwargs):
    tensor = torch.randn(3, 3)
    tensor.tags = ["tag1", "tag2"]
    tensor.description = "desc"

    def compare(detailed, original):
        assert type(detailed) == torch.Tensor
        assert detailed.data.equal(original.data)
        assert detailed.id == original.id
        assert detailed.requires_grad == original.requires_grad
        return True

    return [
        (
            tensor,
            (
                CODE[torch.Tensor],
                (tensor.id, save_to_buffer(tensor), None, None, ["tag1", "tag2"], "desc"),
            ),
            None,
            compare,
        )
    ]


# torch.Size
def make_torch_size(**kwargs):
    return [(torch.randn(3, 3).size(), (CODE[torch.Size], (3, 3)))]


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
        (
            ast,
            (
                CODE[
                    syft.frameworks.torch.tensors.interpreters.additive_shared.AdditiveSharingTensor
                ],
                (ast.id, ast.field, ast.crypto_provider.id, serde._simplify(ast.child), 1),
            ),
            None,
            compare,
        )
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
    # It is OK to assign like that?
    fpt.tags = ["tag1", "tag2"]
    fpt.description = "desc"

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
        return True

    return [
        (
            fpt,
            (
                CODE[syft.frameworks.torch.tensors.interpreters.precision.FixedPrecisionTensor],
                (
                    fpt.id,
                    fpt.field,
                    12,
                    5,
                    fpt.kappa,
                    (CODE[list], ((CODE[str], (b"tag1",)), (CODE[str], (b"tag2",)))),
                    (CODE[str], (b"desc",)),
                    serde._simplify(fpt.child),
                ),
            ),
            None,
            compare,
        )
    ]


# CRTPrecisionTensor
def make_crtprecisiontensor(**kwargs):
    workers = kwargs["workers"]
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]
    t = torch.tensor([[3.1, 4.3]])
    cpt = t.fix_prec(storage="crt").share(alice, bob, crypto_provider=james).child

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
        (
            cpt,
            (
                CODE[syft.frameworks.torch.tensors.interpreters.crt_precision.CRTPrecisionTensor],
                (cpt.id, cpt.base, cpt.precision_fractional, serde._simplify(cpt.child)),
            ),
            None,
            compare,
        )
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
        (
            lt,
            (
                CODE[syft.frameworks.torch.tensors.decorators.logging.LoggingTensor],
                (lt.id, serde._simplify(lt.child)),
            ),
            None,
            compare,
        )
    ]


# MultiPointerTensor
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
        (
            mpt,
            (
                CODE[syft.generic.pointers.multi_pointer.MultiPointerTensor],
                (mpt.id, serde._simplify(mpt.child)),
            ),
            None,
            compare,
        )
    ]


# Plan
def make_plan(**kwargs):
    @syft.func2plan([torch.Size((1, 3))])
    def plan(x):
        x = x + x
        x = torch.abs(x)
        return x

    def compare(detailed, original):
        assert type(detailed) == syft.messaging.plan.plan.Plan
        assert detailed.id == original.id
        # assert detailed.procedure == original.procedure
        # assert detailed.state == original.state
        assert detailed.include_state == original.include_state
        assert detailed.is_built == original.is_built
        assert detailed.name == original.name
        assert detailed.tags == original.tags
        assert detailed.description == original.description
        assert detailed(torch.tensor([1, 2, 3])).equal(original(torch.tensor([1, 2, 3])))
        return True

    return [
        (
            plan,
            (
                CODE[syft.messaging.plan.plan.Plan],
                (
                    serde._simplify(plan.id),
                    serde._simplify(plan.procedure),
                    serde._simplify(plan.state),
                    serde._simplify(plan.include_state),
                    serde._simplify(plan.is_built),
                    serde._simplify(plan.name),
                    serde._simplify(plan.tags),
                    serde._simplify(plan.description),
                ),
            ),
            None,
            compare,
        )
    ]


# State
def make_state(**kwargs):
    bob = kwargs["workers"]["bob"]
    t1, t2 = torch.randn(3, 3), torch.randn(3, 3)
    state = syft.messaging.plan.state.State(owner=bob, state_ids=[t1.id, t2.id])
    t1.send(bob), t2.send(bob)

    def compare(detailed, original):
        assert type(detailed) == syft.messaging.plan.state.State
        assert detailed.state_ids == original.state_ids
        return True

    return [
        (
            state,
            (
                CODE[syft.messaging.plan.state.State],
                (
                    (CODE[list], (t1.id, t2.id)),
                    (CODE[list], (serde._simplify(t1), serde._simplify(t2))),
                ),
            ),
            None,
            compare,
        )
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
        return True

    return [
        (
            procedure,
            (
                CODE[syft.messaging.plan.procedure.Procedure],
                (
                    (procedure.operations[0], procedure.operations[1]),
                    (CODE[tuple], (procedure.arg_ids[0],)),
                    (CODE[tuple], (procedure.result_ids[0],)),
                ),
            ),
            None,
            compare,
        )
    ]


# Protocol
def make_protocol(**kwargs):
    alice, bob = kwargs["workers"]["alice"], kwargs["workers"]["bob"]

    @syft.func2plan([torch.Size((1, 3))])
    def plan(x):
        x = x + x
        x = torch.abs(x)
        return x

    protocol = syft.messaging.protocol.Protocol(
        [("worker1", plan), ("worker2", plan)], tags=["aaa", "bbb"], description="desc"
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
        (
            protocol,
            (
                CODE[syft.messaging.protocol.Protocol],
                (
                    protocol.id,  # (int)
                    (CODE[list], ((CODE[str], (b"aaa",)), (CODE[str], (b"bbb",)))),
                    (CODE[str], (b"desc",)),
                    (
                        CODE[list],
                        (
                            (CODE[tuple], ((CODE[str], (b"worker1",)), plan.id)),
                            (CODE[tuple], ((CODE[str], (b"worker2",)), plan.id)),
                        ),
                    ),
                    False,  # workers_resolved
                ),
            ),
            None,
            compare,
        )
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
        assert detailed.get().equal(original.get())
        return True

    return [
        (
            ptr,
            (
                CODE[syft.generic.pointers.pointer_tensor.PointerTensor],
                (ptr.id, ptr.id_at_location, "alice", None, (CODE[torch.Size], (3, 3)), True),
            ),
            None,
            compare,
        )
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
        (
            ptr,
            (
                CODE[syft.generic.pointers.pointer_plan.PointerPlan],
                (ptr.id, ptr.id_at_location, "alice", False),  # location.id  # garbage_collect_data
            ),
            None,
            compare,
        )
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
        (
            ptr,
            (
                CODE[syft.generic.pointers.pointer_protocol.PointerProtocol],
                (
                    ptr.id,  # (int)
                    ptr.id_at_location,  # (int)
                    "alice",  # location.id
                    False,  # garbage_collect_data
                ),
            ),
            None,
            compare,
        )
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
        (
            wrapper,
            (CODE[syft.generic.pointers.object_wrapper.ObjectWrapper], (123, serde._simplify(obj))),
            None,
            compare,
        )
    ]


# syft.federated.train_config.TrainConfig
def make_trainconfig(**kwargs):
    class Model(torch.jit.ScriptModule):
        def __init__(self):
            super(Model, self).__init__()
            self.w1 = torch.nn.Parameter(torch.randn(10, 1), requires_grad=True)
            self.b1 = torch.nn.Parameter(torch.randn(1), requires_grad=True)

        @torch.jit.script_method
        def forward(self, x):
            x = x @ self.w1 + self.b1
            return x

    class Loss(torch.jit.ScriptModule):
        def __init__(self):
            super(Loss, self).__init__()

        @torch.jit.script_method
        def forward(self, pred, target):
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
        assert detailed.model == original.model
        assert detailed.loss_fn == original.loss_fn
        return True

    return [
        (
            conf,
            (
                CODE[syft.federated.train_config.TrainConfig],
                (
                    None,  # _model_id (int)
                    None,  # _loss_fn_id (int)
                    2,  # batch_size (int)
                    1,  # epochs (int)
                    (CODE[str], (b"SGD",)),  # optimizer (str)
                    (CODE[dict], (((CODE[str], (b"lr",)), 0.1),)),  # optimizer_args (dict)
                    conf.id,  # (int)
                    -1,  # max_nr_batches (int)
                    True,  # shuffle
                ),
            ),
            None,
            compare,
        )
    ]


# syft.workers.base.BaseWorker
def make_baseworker(**kwargs):
    bob = kwargs["workers"]["bob"]

    def compare(detailed, original):
        assert isinstance(detailed, syft.workers.base.BaseWorker)
        assert detailed.id == original.id
        return True

    return [
        (
            bob,
            (CODE[syft.workers.base.BaseWorker], ((CODE[str], (b"bob",)),)),  # id (str)
            None,
            compare,
        )
    ]


# syft.frameworks.torch.tensors.interpreters.autograd.AutogradTensor
def make_autogradtensor(**kwargs):
    t = torch.tensor([1, 2, 3])
    agt = syft.frameworks.torch.tensors.interpreters.autograd.AutogradTensor().on(t).child
    agt.tag("aaa")
    agt.description = "desc"

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
        (
            agt,
            (
                CODE[syft.frameworks.torch.tensors.interpreters.autograd.AutogradTensor],
                (
                    None,  # owner (Any)
                    agt.id,  # (int)
                    serde._simplify(agt.child),
                    True,  # requires_grad
                    False,  # preinitialize_grad
                    None,  # grad_fn (always None, ignored in constructor)
                    (CODE[set], ((CODE[str], (b"aaa",)),)),  # tags (set of str)
                    (CODE[str], (b"desc",)),  # description (str)
                ),
            ),
            None,
            compare,
        )
    ]


# Message
def make_message(**kwargs):
    def compare(detailed, original):
        assert type(detailed) == syft.messaging.message.Message
        assert detailed.contents == original.contents
        return True

    return [
        (
            syft.messaging.message.Message(0, [1, 2, 3]),
            (CODE[syft.messaging.message.Message], (0, (CODE[list], (1, 2, 3)))),
            None,
            compare,
        ),
        (
            syft.messaging.message.Message(0, (1, 2, 3)),
            (CODE[syft.messaging.message.Message], (0, (CODE[tuple], (1, 2, 3)))),
            None,
            compare,
        ),
    ]


# Operation
def make_operation(**kwargs):
    bob = kwargs["workers"]["bob"]
    bob.log_msgs = True
    x = torch.tensor([1, 2, 3, 4]).send(bob)
    y = x * 2
    op = bob._get_msg(-1)
    bob.log_msgs = False

    def compare(detailed, original):
        assert type(detailed) == syft.messaging.message.Operation
        assert detailed.message == original.message
        assert detailed.return_ids == original.return_ids
        return True

    return [
        (
            op,
            (
                CODE[syft.messaging.message.Operation],
                (
                    codes.MSGTYPE.CMD,
                    ((serde._simplify(op.message)), op.return_ids),  # tuple  # tuple
                ),
            ),
            None,
            compare,
        )
    ]


# ObjectMessage
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
        (
            obj,
            (
                CODE[syft.messaging.message.ObjectMessage],
                (codes.MSGTYPE.OBJ, serde._simplify(obj.contents)),
            ),
            None,
            compare,
        )
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
        (
            obj_req,
            (
                CODE[syft.messaging.message.ObjectRequestMessage],
                (codes.MSGTYPE.OBJ_REQ, serde._simplify(obj_req.contents)),
            ),
            None,
            compare,
        )
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
        (
            nm,
            (
                CODE[syft.messaging.message.IsNoneMessage],
                (codes.MSGTYPE.IS_NONE, serde._simplify(nm.contents)),
            ),
            None,
            compare,
        )
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
        (
            shape_message,
            (
                CODE[syft.messaging.message.GetShapeMessage],
                (codes.MSGTYPE.GET_SHAPE, serde._simplify(shape_message.contents)),
            ),
            None,
            compare,
        )
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
        (
            del_message,
            (
                CODE[syft.messaging.message.ForceObjectDeleteMessage],
                (codes.MSGTYPE.FORCE_OBJ_DEL, id),  # (int)  # (int)
            ),
            None,
            compare,
        )
    ]


# SearchMessage
def make_searchmessage(**kwargs):
    search_message = syft.messaging.message.SearchMessage([1, "test", 3])

    def compare(detailed, original):
        assert type(detailed) == syft.messaging.message.SearchMessage
        assert detailed.contents == original.contents
        return True

    return [
        (
            search_message,
            (
                CODE[syft.messaging.message.SearchMessage],
                (
                    codes.MSGTYPE.SEARCH,  # (int)
                    (CODE[list], (1, (CODE[str], (b"test",)), 3)),  # (list)
                ),
            ),
            None,
            compare,
        )
    ]


# PlanCommandMessage
def make_plancommandmessage(**kwargs):
    bob = kwargs["workers"]["bob"]
    bob.log_msgs = True

    # hook.local_worker.is_client_worker = False

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
        (
            fetch_plan_cmd,
            (
                CODE[syft.messaging.message.PlanCommandMessage],
                (
                    codes.MSGTYPE.PLAN_CMD,  # (int)
                    ((CODE[str], (b"fetch_plan",)), (CODE[tuple], (plan.id, False))),  # (int)
                ),
            ),
            None,
            compare,
        )
    ]


# syft.exceptions.GetNotPermittedError
def make_getnotpermittederror(**kwargs):
    bob = kwargs["workers"]["bob"]
    bob.log_msgs = True

    # hook.local_worker.is_client_worker = False

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
        (
            fetch_plan_cmd,
            (
                CODE[syft.messaging.message.PlanCommandMessage],
                (
                    codes.MSGTYPE.PLAN_CMD,  # (int)
                    ((CODE[str], (b"fetch_plan",)), (CODE[tuple], (plan.id, False))),  # (int)
                ),
            ),
            None,
            compare,
        )
    ]


# Dictionary containing test samples functions
samples = OrderedDict()

# Native
samples[float] = make_float
samples[int] = make_int
samples[dict] = make_dict
samples[tuple] = make_tuple
samples[list] = make_list
samples[set] = make_set
samples[slice] = make_slice
samples[str] = make_str
samples[range] = make_range
samples[type(Ellipsis)] = make_ellipsis

# Numpy
samples[numpy.ndarray] = make_numpy_ndarray

# PyTorch
samples[torch.device] = make_torch_device
samples[torch.jit.ScriptModule] = make_torch_scriptmodule
samples[torch.jit.TopLevelTracedModule] = make_torch_topleveltracedmodule
samples[torch.nn.Parameter] = make_torch_parameter
samples[torch.Tensor] = make_torch_tensor
samples[torch.Size] = make_torch_size

# PySyft
samples[
    syft.frameworks.torch.tensors.interpreters.additive_shared.AdditiveSharingTensor
] = make_additivesharingtensor
samples[
    syft.frameworks.torch.tensors.interpreters.precision.FixedPrecisionTensor
] = make_fixedprecisiontensor
samples[
    syft.frameworks.torch.tensors.interpreters.crt_precision.CRTPrecisionTensor
] = make_crtprecisiontensor
samples[syft.frameworks.torch.tensors.decorators.logging.LoggingTensor] = make_loggingtensor
samples[syft.generic.pointers.multi_pointer.MultiPointerTensor] = make_multipointertensor
samples[syft.messaging.plan.plan.Plan] = make_plan
samples[syft.messaging.plan.state.State] = make_state
samples[syft.messaging.plan.procedure.Procedure] = make_procedure
samples[syft.messaging.protocol.Protocol] = make_protocol
samples[syft.generic.pointers.pointer_tensor.PointerTensor] = make_pointertensor
samples[syft.generic.pointers.pointer_plan.PointerPlan] = make_pointerplan
samples[syft.generic.pointers.pointer_protocol.PointerProtocol] = make_pointerprotocol
samples[syft.generic.pointers.object_wrapper.ObjectWrapper] = make_objectwrapper
samples[syft.federated.train_config.TrainConfig] = make_trainconfig
samples[syft.workers.base.BaseWorker] = make_baseworker
samples[syft.frameworks.torch.tensors.interpreters.autograd.AutogradTensor] = make_autogradtensor

samples[syft.messaging.message.Message] = make_message
samples[syft.messaging.message.Operation] = make_operation
samples[syft.messaging.message.ObjectMessage] = make_objectmessage
samples[syft.messaging.message.ObjectRequestMessage] = make_objectrequestmessage
samples[syft.messaging.message.IsNoneMessage] = make_isnonemessage
samples[syft.messaging.message.GetShapeMessage] = make_getshapemessage
samples[syft.messaging.message.ForceObjectDeleteMessage] = make_forceobjectdeletemessage
samples[syft.messaging.message.SearchMessage] = make_searchmessage
samples[syft.messaging.message.PlanCommandMessage] = make_plancommandmessage

samples[syft.exceptions.GetNotPermittedError] = make_getnotpermittederror
# samples[syft.messaging.message.PlanCommandMessage] = make_plancommandmessage


def test_serde_coverage():
    """Checks all types in serde are tested"""
    for cls, _ in serde.simplifiers.items():
        has_sample = cls in samples
        assert has_sample is True, "Serde for %s is not tested" % cls


@pytest.mark.parametrize("cls", samples)
def test_serde_roundtrip(cls, workers):
    """Checks that values passed through serialization-deserialization stay same"""
    _samples = samples[cls](workers=workers)
    for obj, *params in _samples:
        simplified_obj = serde._simplify(obj)
        detailed_obj = serde._detail(syft.hook.local_worker, simplified_obj)
        if len(params) >= 3 and params[2] is not None:
            # Custom detailed objects comparison function.
            comp_func = params[2]
            assert comp_func(detailed_obj, obj) is True
        else:
            assert type(detailed_obj) == type(obj)
            assert detailed_obj == obj


@pytest.mark.parametrize("cls", samples)
def test_serde_simplify(cls, workers):
    """Checks that simplified structures match expected"""
    _samples = samples[cls](workers=workers)
    for sample in _samples:
        obj, expected_simplified_obj = sample[0], sample[1]
        simplified_obj = serde._simplify(obj)

        if len(sample) >= 3 and sample[2] is not None:
            # Custom simplified objects comparison function.
            comp_func = sample[2]
            assert comp_func(simplified_obj, expected_simplified_obj) is True
        else:
            assert simplified_obj == expected_simplified_obj
