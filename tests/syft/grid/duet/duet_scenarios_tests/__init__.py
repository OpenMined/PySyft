# stdlib
from typing import Callable
from typing import List
from typing import Tuple

# syft relative
from .duet_init_test import test_scenario_init
from .duet_sanity_test import test_scenario_sanity
from .duet_torch_test import test_scenario_torch_tensor_sanity


def register_duet_scenarios(
    registered_tests: List[Tuple[str, Callable, Callable]]
) -> None:
    registered_tests.append(test_scenario_init)
    registered_tests.append(test_scenario_sanity)
    registered_tests.append(test_scenario_torch_tensor_sanity)
