from ..service.sync.diff_state import DiffState

from IPython.display import display, Markdown


def compare_states(low_state, high_state) -> DiffState:
    return DiffState.from_sync_states(low_state=low_state, high_state=high_state)


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
                    f"Do you approve moving this object from the {source} side to the {destination} side (approve/deny): ",
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
            pass

    return low_new_objs, high_new_objs
