# stdlib
from typing import Optional

# relative
from ..service.sync.diff_state import NodeDiff
from ..service.sync.diff_state import ObjectDiff
from ..service.sync.diff_state import ResolvedSyncState
from ..service.sync.diff_state import resolve_diff


def compare_states(low_state, high_state) -> NodeDiff:
    return NodeDiff.from_sync_state(low_state=low_state, high_state=high_state)


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


def resolve(state: NodeDiff, decision: Optional[str] = None):
    # TODO: only add permissions for objects where we manually give permission
    # Maybe default read permission for some objects (high -> low)
    resolved_state_low: ResolvedSyncState = ResolvedSyncState()
    resolved_state_high: ResolvedSyncState = ResolvedSyncState()

    for batch_hierarchy in state.hierarchies:
        batch_decision = decision
        if all(diff.status == "SAME" for diff in batch_hierarchy.diffs):
            # Hierarchy has no diffs
            continue

        print(batch_hierarchy.__repr__())

        if batch_decision is None:
            batch_decision = get_user_input_for_resolve()

        print(
            f"Decision: Syncing {len(batch_hierarchy)} objects from {batch_decision} side"
        )

        object_diff: ObjectDiff
        for object_diff in batch_hierarchy.diffs:
            low_resolved_batch_state, high_resolved_batch_state = resolve_diff(
                object_diff, decision=batch_decision
            )
            resolved_state_low.add(low_resolved_batch_state)
            resolved_state_high.add(high_resolved_batch_state)

    return resolved_state_low, resolved_state_high
