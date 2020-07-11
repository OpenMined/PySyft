import syft
from .policy import Policy
from typing import Final, Dict
from ..... import type_hints

REGISTERED_POLICIES: Final = dict()


@type_hints
def register_policy(policy: Policy) -> None:
    REGISTERED_POLICIES[type(policy)] = policy


@type_hints
def get_registered_policies() -> Dict[type, Policy]:
    return REGISTERED_POLICIES
