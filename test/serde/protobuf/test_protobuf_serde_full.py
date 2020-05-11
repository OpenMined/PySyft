from collections import OrderedDict
import pytest
import numpy
import torch
from functools import partial
import traceback
import io

import syft
from syft.serde import protobuf
from test.serde.serde_helpers import *

# Dictionary containing test samples functions
samples = OrderedDict()

# Native
samples[type(None)] = make_none
samples[type] = make_type

# PyTorch
samples[torch.device] = make_torch_device
samples[torch.jit.ScriptModule] = make_torch_scriptmodule
samples[torch.jit.ScriptFunction] = make_torch_scriptfunction
samples[torch.jit.TopLevelTracedModule] = make_torch_topleveltracedmodule
samples[torch.nn.Parameter] = make_torch_parameter
samples[torch.Tensor] = make_torch_tensor
samples[torch.Size] = make_torch_size

# PySyft
samples[
    syft.frameworks.torch.tensors.interpreters.additive_shared.AdditiveSharingTensor
] = make_additivesharingtensor
samples[syft.execution.placeholder.PlaceHolder] = make_placeholder
samples[syft.execution.computation.ComputationAction] = make_computation_action
samples[syft.execution.communication.CommunicationAction] = make_communication_action
samples[syft.execution.plan.Plan] = make_plan
samples[syft.execution.protocol.Protocol] = make_protocol
samples[syft.execution.role.Role] = make_role
samples[syft.execution.state.State] = make_state
samples[syft.execution.placeholder_id.PlaceholderId] = make_placeholder_id
samples[syft.execution.plan.NestedTypeWrapper] = make_nested_type_wrapper
samples[syft.generic.pointers.pointer_tensor.PointerTensor] = make_pointertensor

# Syft Messages
samples[syft.messaging.message.ObjectMessage] = make_objectmessage
samples[syft.messaging.message.TensorCommandMessage] = make_tensor_command_message


def test_serde_coverage():
    """Checks all types in serde are tested"""
    for cls, _ in protobuf.serde.get_bufferizers():
        has_sample = cls in samples
        assert has_sample, f"Serde for {cls} is not tested"


@pytest.mark.parametrize("cls", samples)
def test_serde_roundtrip_protobuf(cls, workers, hook):
    """Checks that values passed through serialization-deserialization stay same"""
    serde_worker = syft.VirtualWorker(id=f"serde-worker-{cls.__name__}", hook=hook, auto_add=False)
    original_framework = serde_worker.framework
    workers["serde_worker"] = serde_worker
    _samples = samples[cls](workers=workers)
    for sample in _samples:
        _to_protobuf = (
            protobuf.serde._bufferize
            if not sample.get("forced", False)
            else protobuf.serde._force_full_bufferize
        )
        serde_worker.framework = sample.get("framework", torch)
        obj = sample.get("value")
        protobuf_obj = _to_protobuf(serde_worker, obj)
        roundtrip_obj = None
        if not isinstance(obj, Exception):
            roundtrip_obj = protobuf.serde._unbufferize(serde_worker, protobuf_obj)

        serde_worker.framework = original_framework

        if sample.get("cmp_detailed", None):
            # Custom detailed objects comparison function.
            assert sample.get("cmp_detailed")(roundtrip_obj, obj)
        else:
            assert type(roundtrip_obj) == type(obj)
            assert roundtrip_obj == obj
