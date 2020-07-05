import syft
from .policy import Policy
from typing import Final, Dict

REGISTERED_POLICIES: Final = dict()

@syft.typecheck.type_hints
def register_policy(policy: Policy) -> None:
    REGISTERED_POLICIES[type(policy)] = policy

@syft.typecheck.type_hints
def get_registered_policies() -> Dict[type, Policy]:
    return REGISTERED_POLICIES