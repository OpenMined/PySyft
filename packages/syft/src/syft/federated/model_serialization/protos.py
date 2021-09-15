# flake8: noqa
# These are the old syft_proto protobufs being re-exported from a single file because
# the python plugin absolutify freaks out and breaks them
# third party
from syft_proto.execution.v1.placeholder_id_pb2 import PlaceholderId as PlaceholderId_PB
from syft_proto.execution.v1.placeholder_pb2 import Placeholder as Placeholder_PB
from syft_proto.execution.v1.plan_pb2 import Plan as Plan_PB
from syft_proto.execution.v1.state_pb2 import State as State_PB
from syft_proto.execution.v1.state_tensor_pb2 import StateTensor as StateTensor_PB
from syft_proto.types.torch.v1.tensor_data_pb2 import TensorData as TensorData_PB
from syft_proto.types.torch.v1.tensor_pb2 import TorchTensor as TorchTensor_PB
