# stdlib
from time import sleep
from typing import List
from typing import Optional
from typing import Union

# relative
from ..service.action.action_object import ActionObject
from ..service.action.action_permissions import ActionObjectPermission
from ..service.action.action_permissions import ActionPermission
from ..service.code.user_code import UserCode
from ..service.job.job_stash import Job
from ..service.log.log import SyftLog
from ..service.sync.diff_state import NodeDiff
from ..service.sync.diff_state import ResolvedSyncState
from ..service.sync.sync_state import SyncState


def compare_states(low_state: SyncState, high_state: SyncState) -> NodeDiff:
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


def resolve(
    state: NodeDiff, decision: Optional[str] = None, share_private_objects=False
):
    # TODO: only add permissions for objects where we manually give permission
    # Maybe default read permission for some objects (high -> low)
    resolved_state_low: ResolvedSyncState = ResolvedSyncState(alias="low")
    resolved_state_high: ResolvedSyncState = ResolvedSyncState(alias="high")

    for batch_diff in state.hierarchies:
        batch_decision = decision
        if all(diff.status == "SAME" for diff in batch_diff.diffs):
            # Hierarchy has no diffs
            continue

        print(batch_diff.__repr__())

        # ask question: which side do you want
        # ask question: The batch has private items that you may want to share with the related user
        # user with verify key: abc. The items are
        # Log with id (123)
        # Result with id (567)
        # do you want to give read permission to items
        # TODO: get decision
        # get items
        if batch_decision is None:
            batch_decision = get_user_input_for_resolve()

        get_user_input_for_batch_permissions(
            batch_diff, share_private_objects=share_private_objects
        )

        print(f"Decision: Syncing {len(batch_diff)} objects from {batch_decision} side")

        for object_diff in batch_diff.diffs:
            resolved_state_low.add_cruds_from_diff(object_diff, batch_decision)
            resolved_state_high.add_cruds_from_diff(object_diff, batch_decision)

            resolved_state_low.new_permissions += object_diff.new_low_permissions

        print()
        print("=" * 100)
        print()

    return resolved_state_low, resolved_state_high


def get_user_input_for_batch_permissions(batch_diff, share_private_objects=False):
    private_high_objects: List[Union[SyftLog, ActionObject]] = []

    for diff in batch_diff.diffs:
        if isinstance(diff.high_obj, (SyftLog, ActionObject)):
            private_high_objects.append(diff)

    user_codes_high: List[UserCode] = [
        diff.high_obj
        for diff in batch_diff.diffs
        if isinstance(diff.high_obj, UserCode)
    ]
    if not len(user_codes_high) < 2:
        raise ValueError("too many user codes")

    if user_codes_high:
        user_code_high = user_codes_high[0]

        # TODO: only do this under condition that its accepted to sync
        high_job_diffs = [
            diff for diff in batch_diff.diffs if isinstance(diff.high_obj, Job)
        ]

        for diff in high_job_diffs:
            read_permission_job = ActionObjectPermission(
                uid=diff.object_id,
                permission=ActionPermission.READ,
                credentials=user_code_high.user_verify_key,
            )
            diff.new_low_permissions.append(read_permission_job)

        if share_private_objects:
            for diff in private_high_objects:
                read_permission_private_obj = ActionObjectPermission(
                    uid=diff.object_id,
                    permission=ActionPermission.READ,
                    credentials=user_code_high.user_verify_key,
                )
                diff.new_low_permissions.append(read_permission_private_obj)

        else:
            print(
                f"""This batch of updates contains new private objects on the high side that you may want \
            to share with user {user_code_high.user_verify_key}."""
            )
            while True:
                if len(private_high_objects) > 0:
                    if user_code_high is None:
                        raise ValueError("No usercode found for private objects")
                    objects_str = "\n".join(
                        [
                            f"{diff.object_type} #{diff.object_id}"
                            for diff in private_high_objects
                        ]
                    )
                    print(
                        f"""
            You currently have the following private objects:

            {objects_str}

            Do you want to share some of these private objects? If so type the first 3 characters of the id e.g. 'abc'.
            If you dont want to share any more private objects, type "no"
            """,
                        flush=True,
                    )
                else:
                    break

                sleep(0.1)
                res = input()
                if res == "no":
                    break
                elif len(res) >= 3:
                    matches = [
                        diff
                        for diff in private_high_objects
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
                            f"""
            Setting permissions for {diff.object_type} #{diff.object_id} to share with ABC,
            this will become effective when you call client.apply_state(<resolved_state>))
            """
                        )
                        private_high_objects.remove(diff)
                        read_permission_private_obj = ActionObjectPermission(
                            uid=diff.object_id,
                            permission=ActionPermission.READ,
                            credentials=user_code_high.user_verify_key,
                        )
                        diff.new_low_permissions.append(read_permission_private_obj)

                        # questions
                        # Q:do we also want to give read permission if we defined that by accept_by_depositing_result?
                        # A:only if we pass: sync_read_permission to resolve
                    else:
                        print("Found multiple matches for provided id, exiting")
                        break
                else:
                    print("invalid input")
