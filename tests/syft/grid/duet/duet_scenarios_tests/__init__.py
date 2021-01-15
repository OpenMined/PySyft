# stdlib
from typing import Callable
from typing import List
from typing import Tuple

# syft relative
from .duet_init_test import test_scenario_init


def register_duet_scenarios(registered_tests: List[Tuple[Callable, Callable]]) -> None:
    registered_tests.append(test_scenario_init)
