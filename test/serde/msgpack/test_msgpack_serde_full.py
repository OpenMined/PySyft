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
samples[type] = make_type

# Numpy
samples[numpy.float32] = partial(make_numpy_number, numpy.float32)
samples[numpy.float64] = partial(make_numpy_number, numpy.float64)
samples[numpy.int32] = partial(make_numpy_number, numpy.int32)
samples[numpy.int64] = partial(make_numpy_number, numpy.int64)
samples[numpy.ndarray] = make_numpy_ndarray

# PyTorch
samples[torch.device] = make_torch_device
samples[torch.dtype] = make_torch_dtype
samples[torch.jit.ScriptModule] = make_torch_scriptmodule
samples[torch.jit.ScriptFunction] = make_torch_scriptfunction
samples[torch.jit.TopLevelTracedModule] = make_torch_topleveltracedmodule
samples[torch.memory_format] = make_torch_memoryformat
samples[torch.nn.Parameter] = make_torch_parameter
samples[torch.Tensor] = make_torch_tensor
samples[torch.Size] = make_torch_size

# PySyft
samples[syft.exceptions.GetNotPermittedError] = make_getnotpermittederror
samples[syft.exceptions.ResponseSignatureError] = make_responsesignatureerror

samples[syft.execution.communication.CommunicationAction] = make_communication_action
samples[syft.execution.computation.ComputationAction] = make_computation_action
samples[syft.execution.placeholder.PlaceHolder] = make_placeholder
samples[syft.execution.placeholder_id.PlaceholderId] = make_placeholder_id
samples[syft.execution.plan.NestedTypeWrapper] = make_nested_type_wrapper
samples[syft.execution.plan.Plan] = make_plan
samples[syft.execution.protocol.Protocol] = make_protocol
samples[syft.execution.role.Role] = make_role
samples[syft.execution.state.State] = make_state

samples[syft.federated.train_config.TrainConfig] = make_trainconfig

samples[syft.frameworks.torch.fl.dataset.BaseDataset] = make_basedataset
samples[syft.frameworks.torch.tensors.decorators.logging.LoggingTensor] = make_loggingtensor
samples[
    syft.frameworks.torch.tensors.interpreters.additive_shared.AdditiveSharingTensor
] = make_additivesharingtensor
samples[syft.frameworks.torch.tensors.interpreters.autograd.AutogradTensor] = make_autogradtensor
samples[syft.frameworks.torch.tensors.interpreters.gradients_core.GradFunc] = make_gradfn
samples[syft.frameworks.torch.tensors.interpreters.paillier.PaillierTensor] = make_paillier
samples[
    syft.frameworks.torch.tensors.interpreters.precision.FixedPrecisionTensor
] = make_fixedprecisiontensor
samples[syft.frameworks.torch.tensors.interpreters.private.PrivateTensor] = make_privatetensor

samples[syft.generic.pointers.multi_pointer.MultiPointerTensor] = make_multipointertensor
samples[syft.generic.pointers.object_pointer.ObjectPointer] = make_objectpointer
samples[syft.generic.pointers.object_wrapper.ObjectWrapper] = make_objectwrapper
samples[syft.generic.pointers.pointer_tensor.PointerTensor] = make_pointertensor
samples[syft.generic.pointers.pointer_plan.PointerPlan] = make_pointerplan
samples[syft.generic.pointers.pointer_dataset.PointerDataset] = make_pointerdataset
samples[syft.generic.string.String] = make_string

samples[syft.messaging.message.ForceObjectDeleteMessage] = make_forceobjectdeletemessage
samples[syft.messaging.message.GetShapeMessage] = make_getshapemessage
samples[syft.messaging.message.IsNoneMessage] = make_isnonemessage
samples[syft.messaging.message.ObjectMessage] = make_objectmessage
samples[syft.messaging.message.ObjectRequestMessage] = make_objectrequestmessage
samples[syft.messaging.message.PlanCommandMessage] = make_plancommandmessage
samples[syft.messaging.message.SearchMessage] = make_searchmessage
samples[syft.messaging.message.TensorCommandMessage] = make_tensor_command_message
samples[syft.messaging.message.WorkerCommandMessage] = make_workercommandmessage

samples[syft.workers.virtual.VirtualWorker] = make_virtual_worker


def test_serde_coverage():
    """Checks all types in serde are tested"""
    for cls, _ in msgpack.serde.simplifiers.items():
        has_sample = cls in samples
        assert has_sample, f"Serde for {cls} is not tested"


@pytest.mark.parametrize("cls", samples)
def test_serde_roundtrip(cls, workers, hook, start_remote_worker):
    """Checks that values passed through serialization-deserialization stay same"""
    serde_worker = syft.VirtualWorker(id=f"serde-worker-{cls.__name__}", hook=hook, auto_add=False)
    workers["serde_worker"] = serde_worker
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
    serde_worker = syft.VirtualWorker(id=f"serde-worker-{cls.__name__}", hook=hook, auto_add=False)
    workers["serde_worker"] = serde_worker
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
        serde_worker.framework = sample.get("framework", torch)
        simplified_obj = _simplify(serde_worker, obj)

        if sample.get("cmp_simplified", None):
            # Custom simplified objects comparison function.
            assert sample.get("cmp_simplified")(simplified_obj, expected_simplified_obj) is True
        else:
            assert simplified_obj == expected_simplified_obj
