# stdlib
from collections.abc import Callable
from time import sleep

# relative
from ..abstract_node import NodeSideType
from ..service.action.action_permissions import ActionObjectPermission
from ..service.action.action_permissions import ActionPermission
from ..service.action.action_permissions import StoragePermission
from ..service.code.user_code import UserCode
from ..service.job.job_stash import Job
from ..service.sync.diff_state import NodeDiff
from ..service.sync.diff_state import ObjectDiff
from ..service.sync.diff_state import ObjectDiffBatch
from ..service.sync.diff_state import ResolvedSyncState
from ..service.sync.diff_state import SyncInstruction
from ..service.sync.resolve_widget import ResolveWidget
from ..service.sync.sync_state import SyncState
from .client import SyftClient
from .sync_decision import SyncDecision
from .sync_decision import SyncDirection


def compare_states(state1: SyncState, state2: SyncState) -> NodeDiff:
    # NodeDiff
    if (
        state1.node_side_type == NodeSideType.LOW_SIDE
        and state2.node_side_type == NodeSideType.HIGH_SIDE
    ):
        low_state = state1
        high_state = state2
        direction = SyncDirection.LOW_TO_HIGH
    elif (
        state1.node_side_type == NodeSideType.HIGH_SIDE
        and state2.node_side_type == NodeSideType.LOW_SIDE
    ):
        low_state = state2
        high_state = state1
        direction = SyncDirection.HIGH_TO_LOW
    else:
        raise ValueError("Invalid SyncStates")
    return NodeDiff.from_sync_state(
        low_state=low_state, high_state=high_state, direction=direction
    )


def compare_clients(low_client: SyftClient, high_client: SyftClient) -> NodeDiff:
    return compare_states(low_client.get_sync_state(), high_client.get_sync_state())


def get_user_input_for_resolve() -> SyncDecision:
    options = [x.value for x in SyncDecision]
    options_str = ", ".join(options[:-1]) + f" or {options[-1]}"
    print(f"How do you want to sync these objects? choose between {options_str}")

    while True:
        decision = input()
        decision = decision.lower()

        try:
            return SyncDecision(decision)
        except ValueError:
            print(f"Please choose between {options_str}")


def handle_ignore_skip(
    batch: ObjectDiffBatch, decision: SyncDecision, other_batches: list[ObjectDiffBatch]
) -> None:
    # make sure type is SyncDecision at runtime
    decision = SyncDecision(decision)

    if decision == SyncDecision.skip or decision == SyncDecision.ignore:
        skipped_or_ignored_ids = {
            x.object_id for x in batch.get_dependents(include_roots=False)
        }
        for other_batch in other_batches:
            if other_batch.decision != decision:
                # Currently, this is not recursive, in the future it might be
                other_batch_ids = {
                    d.object_id
                    for d in other_batch.get_dependencies(include_roots=True)
                }
                if len(other_batch_ids & skipped_or_ignored_ids) != 0:
                    other_batch.decision = decision
                    skipped_or_ignored_ids.update(other_batch_ids)
                    action = "Skipping" if decision == SyncDecision.skip else "Ignoring"
                    print(
                        f"\n{action} other batch with root {other_batch.root_type.__name__}\n"
                    )


def resolve_single(obj_diff_batch: ObjectDiffBatch) -> ResolveWidget:
    widget = ResolveWidget(obj_diff_batch)
    return widget


def resolve(
    state: NodeDiff,
    decision: str | None = None,
    decision_callback: Callable[[ObjectDiffBatch], SyncDecision] | None = None,
    share_private_objects: bool = False,
    ask_for_input: bool = True,
) -> tuple[ResolvedSyncState, ResolvedSyncState]:
    # TODO: fix this
    previously_ignored_batches = state.low_state.ignored_batches
    # TODO: only add permissions for objects where we manually give permission
    # Maybe default read permission for some objects (high -> low)
    resolved_state_low = ResolvedSyncState(node_uid=state.low_node_uid, alias="low")
    resolved_state_high = ResolvedSyncState(node_uid=state.high_node_uid, alias="high")

    for batch_diff in state.batches:
        if batch_diff.is_unchanged:
            # Hierarchy has no diffs
            continue

        if batch_diff.decision is not None:
            # handles ignores
            batch_decision = batch_diff.decision
        elif decision is not None:
            print(batch_diff.__repr__())
            batch_decision = SyncDecision(decision)
        elif decision_callback is not None:
            batch_decision = decision_callback(batch_diff)
        else:
            print(batch_diff.__repr__())
            batch_decision = get_user_input_for_resolve()

        batch_diff.decision = batch_decision

        other_batches = [b for b in state.batches if b is not batch_diff]
        handle_ignore_skip(batch_diff, batch_decision, other_batches)

        if batch_decision not in [SyncDecision.skip, SyncDecision.ignore]:
            sync_instructions = get_sync_instructions_for_batch_items_for_add(
                batch_diff,
                batch_decision,
                share_private_objects=share_private_objects,
                ask_for_input=ask_for_input,
            )
        else:
            sync_instructions = []
            if batch_decision == SyncDecision.ignore:
                resolved_state_high.add_ignored(batch_diff)
                resolved_state_low.add_ignored(batch_diff)

        if (
            batch_diff.root_id in previously_ignored_batches
            and batch_diff.decision != SyncDecision.ignore
        ):
            resolved_state_high.add_unignored(batch_diff.root_id)
            resolved_state_low.add_unignored(batch_diff.root_id)

        print(f"Decision: Syncing {len(sync_instructions)} objects")

        for sync_instruction in sync_instructions:
            resolved_state_low.add_sync_instruction(sync_instruction)
            resolved_state_high.add_sync_instruction(sync_instruction)

        print()
        print("=" * 100)
        print()

    return resolved_state_low, resolved_state_high


def get_sync_instructions_for_batch_items_for_add(
    batch_diff: ObjectDiffBatch,
    decision: SyncDecision,
    share_private_objects: bool = False,
    ask_for_input: bool = True,
) -> list[SyncInstruction]:
    sync_decisions: list[SyncInstruction] = []

    unpublished_private_high_diffs: list[ObjectDiff] = []
    for diff in batch_diff.get_dependents(include_roots=False):
        is_high_private_object = (
            diff.high_obj is not None and diff.high_obj._has_private_sync_attrs()
        )
        is_low_published_object = diff.low_node_uid in diff.low_storage_permissions
        if is_high_private_object and not is_low_published_object:
            unpublished_private_high_diffs.append(diff)

    user_codes_high: list[UserCode] = [
        diff.high_obj
        for diff in batch_diff.get_dependencies(include_roots=True)
        if isinstance(diff.high_obj, UserCode)
    ]

    if len(user_codes_high) == 0:
        user_code_high = None
    else:
        # NOTE we can always assume the first usercode is
        # not a nested code, because diffs are sorted in depth-first order
        user_code_high = user_codes_high[0]

    if user_code_high is None and len(unpublished_private_high_diffs):
        raise ValueError("Found unpublished private objects without user code")

    if share_private_objects:
        private_high_diffs_to_share = unpublished_private_high_diffs
    elif ask_for_input:
        private_high_diffs_to_share = ask_user_input_permission(
            user_code_high, unpublished_private_high_diffs
        )
    else:
        private_high_diffs_to_share = []

    for diff in batch_diff.get_dependencies(include_roots=False):
        is_unpublished_private_diff = diff in unpublished_private_high_diffs
        has_share_decision = diff in private_high_diffs_to_share

        if isinstance(diff.high_obj, Job):
            if user_code_high is None:
                raise ValueError("Job without user code")
            # Jobs are always shared
            new_permissions_low_side = [
                ActionObjectPermission(
                    uid=diff.object_id,
                    permission=ActionPermission.READ,
                    credentials=user_code_high.user_verify_key,
                )
            ]
            mockify = False

        elif is_unpublished_private_diff and has_share_decision:
            # private + want to share
            new_permissions_low_side = [
                ActionObjectPermission(
                    uid=diff.object_id,
                    permission=ActionPermission.READ,
                    credentials=user_code_high.user_verify_key,  # type: ignore
                )
            ]
            mockify = False

        elif is_unpublished_private_diff and not has_share_decision:
            # private + do not share
            new_permissions_low_side = []
            mockify = True

        else:
            # any other object is shared
            new_permissions_low_side = []
            mockify = False

        new_storage_permissions_lowside = []
        if not mockify:
            new_storage_permissions_lowside = [
                StoragePermission(uid=diff.object_id, node_uid=diff.low_node_uid)
            ]

        if (
            diff.status == "NEW"
            and diff.high_obj is None
            and decision == SyncDecision.low
        ):
            new_storage_permissions_highside = [
                StoragePermission(uid=diff.object_id, node_uid=diff.high_node_uid)
            ]
        else:
            new_storage_permissions_highside = []

        sync_decisions.append(
            SyncInstruction(
                diff=diff,
                decision=decision,
                new_permissions_lowside=new_permissions_low_side,
                new_storage_permissions_lowside=new_storage_permissions_lowside,
                new_storage_permissions_highside=new_storage_permissions_highside,
                mockify=mockify,
            )
        )

    return sync_decisions


QUESTION_SHARE_PRIVATE_OBJS = """You currently have the following private objects:

{objects_str}

Do you want to share some of these private objects? If so type the first 3 characters of the id e.g. 'abc'.
If you want to share all private objects, type "all".
If you dont want to share any more private objects, type "no".
"""

CONFIRMATION_SHARE_PRIVATE_OBJ = """Setting permissions for {object_type} #{object_id} to share with {user_verify_key},
this will become effective when you call client.apply_state(<resolved_state>))
"""


def ask_user_input_permission(
    user_code: UserCode, all_private_high_diffs: list[ObjectDiff]
) -> list[ObjectDiff]:
    if len(all_private_high_diffs) == 0:
        return []

    user_verify_key = user_code.user_verify_key
    private_high_diffs_to_share = []
    print(
        f"""This batch of updates contains new private objects on the high side that you may want \
    to share with user {user_verify_key}."""
    )

    remaining_private_high_diffs = all_private_high_diffs[:]
    while len(remaining_private_high_diffs):
        objects_str = "\n".join(
            [
                f"{diff.object_type} #{diff.object_id}"
                for diff in remaining_private_high_diffs
            ]
        )
        print(QUESTION_SHARE_PRIVATE_OBJS.format(objects_str=objects_str), flush=True)

        sleep(0.1)
        res = input()
        if res == "no":
            break

        if res == "all":
            private_high_diffs_to_share.extend(remaining_private_high_diffs)
            remaining_private_high_diffs = []
        elif len(res) >= 3:
            matches = [
                diff
                for diff in remaining_private_high_diffs
                if str(diff.object_id).startswith(res)
            ]
            if len(matches) == 0:
                print("Invalid input")
                continue
            elif len(matches) == 1:
                diff = matches[0]
                print()
                print("=" * 100)
                print()
                print(
                    CONFIRMATION_SHARE_PRIVATE_OBJ.format(
                        object_type=diff.object_type,
                        object_id=diff.object_id,
                        user_verify_key=user_verify_key,
                    )
                )

                remaining_private_high_diffs.remove(diff)
                private_high_diffs_to_share.append(diff)

            else:
                print("Found multiple matches for provided id, exiting")
                break
        else:
            print("invalid input")

    return private_high_diffs_to_share
