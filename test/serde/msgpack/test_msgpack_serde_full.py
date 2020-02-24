from collections import OrderedDict
import pytest
import numpy
import torch
from functools import partial
import traceback
import io

import syft
from syft.serde import msgpack
from test.serde.serde_helpers import *

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
samples[numpy.float32] = partial(make_numpy_number, numpy.float32)
samples[numpy.float64] = partial(make_numpy_number, numpy.float64)
samples[numpy.int32] = partial(make_numpy_number, numpy.int32)
samples[numpy.int64] = partial(make_numpy_number, numpy.int64)

# PyTorch
samples[torch.device] = make_torch_device
samples[torch.jit.ScriptModule] = make_torch_scriptmodule
samples[torch.jit.ScriptFunction] = make_torch_scriptfunction
samples[torch.jit.TopLevelTracedModule] = make_torch_topleveltracedmodule
samples[torch.nn.Parameter] = make_torch_parameter
samples[torch.Tensor] = make_torch_tensor
samples[torch.Size] = make_torch_size
samples[torch.memory_format] = make_torch_memoryformat

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
samples[syft.execution.plan.Plan] = make_plan
samples[syft.execution.state.State] = make_state
samples[syft.execution.protocol.Protocol] = make_protocol
samples[syft.generic.pointers.pointer_tensor.PointerTensor] = make_pointertensor
samples[syft.generic.pointers.pointer_plan.PointerPlan] = make_pointerplan
samples[syft.generic.pointers.pointer_protocol.PointerProtocol] = make_pointerprotocol
samples[syft.generic.pointers.object_wrapper.ObjectWrapper] = make_objectwrapper
samples[syft.generic.pointers.object_pointer.ObjectPointer] = make_objectpointer
samples[syft.generic.string.String] = make_string
samples[syft.federated.train_config.TrainConfig] = make_trainconfig
samples[syft.workers.base.BaseWorker] = make_baseworker
samples[syft.frameworks.torch.tensors.interpreters.autograd.AutogradTensor] = make_autogradtensor
samples[syft.frameworks.torch.tensors.interpreters.private.PrivateTensor] = make_privatetensor
samples[syft.frameworks.torch.tensors.interpreters.placeholder.PlaceHolder] = make_placeholder

samples[syft.messaging.message.Message] = make_message
samples[syft.messaging.message.OperationMessage] = make_operation
samples[syft.messaging.message.ObjectMessage] = make_objectmessage
samples[syft.messaging.message.ObjectRequestMessage] = make_objectrequestmessage
samples[syft.messaging.message.IsNoneMessage] = make_isnonemessage
samples[syft.messaging.message.GetShapeMessage] = make_getshapemessage
samples[syft.messaging.message.ForceObjectDeleteMessage] = make_forceobjectdeletemessage
samples[syft.messaging.message.SearchMessage] = make_searchmessage
samples[syft.messaging.message.PlanCommandMessage] = make_plancommandmessage
samples[syft.messaging.message.ExecuteWorkerFunctionMessage] = make_executeworkerfunctionmessage

samples[syft.frameworks.torch.tensors.interpreters.gradients_core.GradFunc] = make_gradfn

samples[syft.exceptions.GetNotPermittedError] = make_getnotpermittederror
samples[syft.exceptions.ResponseSignatureError] = make_responsesignatureerror

# Dynamically added to msgpack.serde.simplifiers by some other test
samples[syft.workers.virtual.VirtualWorker] = make_baseworker


def test_serde_coverage():
    """Checks all types in serde are tested"""
    for cls, _ in msgpack.serde.simplifiers.items():
        has_sample = cls in samples
        assert has_sample is True, "Serde for %s is not tested" % cls


@pytest.mark.parametrize("cls", samples)
def test_serde_roundtrip(cls, workers, hook, start_remote_worker):
    """Checks that values passed through serialization-deserialization stay same"""
    _samples = samples[cls](
        workers=workers,
        hook=hook,
        start_remote_worker=start_remote_worker,
        port=9000,
        id="roundtrip",
    )
    for sample in _samples:
        _simplify = (
            msgpack.serde._simplify
            if not sample.get("forced", False)
            else msgpack.serde._force_full_simplify
        )
        serde_worker = syft.hook.local_worker
        serde_worker.framework = sample.get("framework", torch)
        obj = sample.get("value")
        simplified_obj = _simplify(serde_worker, obj)
        if not isinstance(obj, Exception):
            detailed_obj = msgpack.serde._detail(serde_worker, simplified_obj)
        else:
            try:
                msgpack.serde._detail(serde_worker, simplified_obj)
            except Exception as e:
                detailed_obj = e

        if sample.get("cmp_detailed", None):
            # Custom detailed objects comparison function.
            assert sample.get("cmp_detailed")(detailed_obj, obj) is True
        else:
            assert type(detailed_obj) == type(obj)
            assert detailed_obj == obj


@pytest.mark.parametrize("cls", samples)
def test_serde_simplify(cls, workers, hook, start_remote_worker):
    """Checks that simplified structures match expected"""
    _samples = samples[cls](
        workers=workers,
        hook=hook,
        start_remote_worker=start_remote_worker,
        port=9001,
        id="simplify",
    )
    for sample in _samples:
        obj, expected_simplified_obj = sample.get("value"), sample.get("simplified")
        _simplify = (
            msgpack.serde._simplify
            if not sample.get("forced", False)
            else msgpack.serde._force_full_simplify
        )
        serde_worker = syft.hook.local_worker
        serde_worker.framework = sample.get("framework", torch)
        simplified_obj = _simplify(syft.hook.local_worker, obj)

        if sample.get("cmp_simplified", None):
            # Custom simplified objects comparison function.
            assert sample.get("cmp_simplified")(simplified_obj, expected_simplified_obj) is True
        else:
            assert simplified_obj == expected_simplified_obj
