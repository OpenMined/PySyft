# stdlib
from typing import Optional

# third party
from IPython.display import Markdown
from IPython.display import display

# relative
from ..service.sync.diff_state import DiffState
from ..service.sync.diff_state import ResolveState
from ..service.sync.diff_state import display_diff_hierarchy
from ..service.sync.diff_state import resolve_diff


def compare_states(low_state, high_state) -> DiffState:
    return DiffState.from_sync_state(low_state=low_state, high_state=high_state)


def resolve(state: DiffState, force_approve: bool = False):
    low_new_objs = []
    high_new_objs = []
    # new_objs = state.objs_to_sync()
    for new_obj in state.diffs:
        if new_obj.merge_state == "NEW":
            if new_obj.low_obj is None:
                state_list = low_new_objs
                source = "LOW"
                destination = "HIGH"
                obj_to_sync = new_obj.high_obj
            if new_obj.high_obj is None:
                state_list = high_new_objs
                source = "HIGH"
                destination = "LOW"
                obj_to_sync = new_obj.low_obj
            if hasattr(obj_to_sync, "_repr_markdown_"):
                display(Markdown(obj_to_sync._repr_markdown_()))
            else:
                display(obj_to_sync)

            if force_approve:
                state_list.append(obj_to_sync)
            else:
                print(
                    f"Do you approve moving this object from the {source} side to the {destination} side (approve/deny): ",  # noqa: E501
                    flush=True,
                )
                while True:
                    decision = input()
                    if decision == "approve":
                        state_list.append(obj_to_sync)
                        break
                    elif decision == "deny":
                        break
                    else:
                        print("Please write `approve` or `deny`:", flush=True)
        if new_obj.merge_state == "DIFF":
            # TODO: this is a shortcut
            state_list = low_new_objs
            state_list.append(new_obj.high_obj)
            # pass

    return low_new_objs, high_new_objs


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


def resolve_hierarchical(state: DiffState, decision: Optional[str] = None):
    resolved_state_low = ResolveState()
    resolved_state_high = ResolveState()

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
