# stdlib
from typing import Optional

# relative
from ..service.action.action_object import ActionObject
from ..service.log.log import SyftLog
from ..service.sync.diff_state import NodeDiff
from ..service.sync.diff_state import ObjectDiff
from ..service.sync.diff_state import ResolvedSyncState
from ..service.sync.diff_state import display_diff_hierarchy
from ..service.sync.diff_state import resolve_diff


def compare_states(low_state, high_state) -> NodeDiff:
    return NodeDiff.from_sync_state(low_state=low_state, high_state=high_state)


def get_read_permission_for_input(diff: ObjectDiff):
    print(
        f"Object {diff.high_obj.id} is a {diff.object_type} and contains private information.\n\
Do you want to allow the data scientist to read this object? choose 'yes' or 'no'",
        flush=True,
    )

    while True:
        permission = input()
        permission = permission.lower()
        print(f"YOU CHOSE {permission}")
        if permission in ["yes", "no"]:
            return permission
        else:
            print("Please choose between 'yes' or 'no'")


def get_user_input_for_resolve():
    print(
        "Do you want to keep the low state or the high state for these objects? choose 'low' or 'high'",
        flush=True,
    )

    while True:
        decision = input()
        decision = decision.lower()
        print(f"YOU CHOSE {decision}")
        if decision in ["low", "high"]:
            return decision
        else:
            print("Please choose between `low` or `high`")


def resolve(
    state: NodeDiff,
    decision: Optional[str] = None,
    all_permissions: Optional[bool] = False,
):
    # TODO: only add permissions for objects where we manually give permission
    # Maybe default read permission for some objects (high -> low)
    resolved_state_low: ResolvedSyncState = ResolvedSyncState()
    resolved_state_high: ResolvedSyncState = ResolvedSyncState()

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
            low_resolved_diff: ResolvedSyncState
            high_resolved_diff: ResolvedSyncState
            if all_permissions:
                permission = "yes"
            elif (
                isinstance(diff.high_obj, (ActionObject, SyftLog))
                and decision == "high"
            ):
                permission = get_read_permission_for_input(diff)
            else:
                permission = "yes"
            low_resolved_diff, high_resolved_diff = resolve_diff(
                diff, decision=decision, permission=permission
            )
            resolved_state_low.add(low_resolved_diff)
            resolved_state_high.add(high_resolved_diff)

    return resolved_state_low, resolved_state_high
