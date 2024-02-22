# stdlib
from typing import Optional

# relative
from ..service.sync.diff_state import DiffState
from ..service.sync.diff_state import ResolvedSyncState
from ..service.sync.diff_state import display_diff_hierarchy
from ..service.sync.diff_state import resolve_diff


def compare_states(low_state, high_state) -> DiffState:
    return DiffState.from_sync_state(low_state=low_state, high_state=high_state)


def get_user_input_for_resolve():
    print(
        "Do you want to keep the low state or the high state for these objects? choose 'low' or 'high'"
    )

    while True:
        decision = input()
        decision = decision.lower()

        if decision in ["low", "high"]:
            return decision
        else:
            print("Please choose between `low` or `high`")


def resolve(state: DiffState, decision: Optional[str] = None):
    resolved_state_low = ResolvedSyncState()
    resolved_state_high = ResolvedSyncState()

    for diff_hierarchy in state.hierarchies:
        if all(item.merge_state == "SAME" for item, _ in diff_hierarchy):
            # Hierarchy has no diffs
            continue

        display_diff_hierarchy(diff_hierarchy)

        if decision is None:
            decision = get_user_input_for_resolve()
        else:
            print(f"Decision: Syncing all objects from {decision} side")

        for diff, _ in diff_hierarchy:
            low_resolved_diff, high_resolved_diff = resolve_diff(
                diff, decision=decision
            )
            resolved_state_low.add(low_resolved_diff)
            resolved_state_high.add(high_resolved_diff)

    return resolved_state_low, resolved_state_high
