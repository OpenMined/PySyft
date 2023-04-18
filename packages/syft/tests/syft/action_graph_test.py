# third party
import numpy as np

# syft absolute
from syft.service.action.action_graph import ActionGraph
from syft.service.action.numpy import NumpyArrayObject


def test_node_creation(worker):
    ActionGraph(node_uid=worker.id)
    NumpyArrayObject(syft_action_data=np.array([1, 2, 4]))
    NumpyArrayObject(syft_action_data=np.array([2, 4, 6]))
