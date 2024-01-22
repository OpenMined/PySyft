# stdlib
import random

# third party
from hagrid.orchestra import NodeHandle
import pytest

# syft absolute
import syft as sy


@pytest.fixture
def node_based_on_type(request) -> NodeHandle:
    """
    This fixture serves as a shared test environment for some tests
    While this saves time to setup and teardown the node, it requires careful management
    to ensure tests using this fixture do not interfere with each other and to handle
    state reset between tests.
    """
    if request.param == "python":
        deployment_type = "python"
        in_memory_workers = True
        port = None
    elif request.param == "container_stack":
        deployment_type = "container_stack"
        in_memory_workers = False
        port = 8081
    else:
        raise ValueError(
            f"node type has to be either 'python' or 'container_stack'. Got {request.param}"
        )
    _node: NodeHandle = sy.orchestra.launch(
        name="container_workload_test_domain",
        deploy_to=deployment_type,
        dev_mode=True,
        reset=True,
        n_consumers=3,
        create_producer=True,
        queue_port=random.randint(13000, 13300),
        in_memory_workers=in_memory_workers,
        port=port,
    )
    # startup code here
    yield _node
    # Cleanup code
    _node.land()
