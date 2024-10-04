# stdlib

# stdlib
from collections.abc import Collection
import logging

# third party
from IPython.display import display

# relative
from ..abstract_server import ServerSideType
from ..server.credentials import SyftVerifyKey
from ..service.response import SyftSuccess
from ..service.response import SyftWarning
from ..service.sync.diff_state import ObjectDiffBatch
from ..service.sync.diff_state import ServerDiff
from ..service.sync.diff_state import SyncInstruction
from ..service.sync.resolve_widget import PaginatedResolveWidget
from ..service.sync.resolve_widget import ResolveWidget
from ..service.sync.sync_state import SyncState
from ..types.errors import SyftException
from ..types.uid import UID
from ..util.decorators import deprecated
from ..util.util import prompt_warning_message
from .datasite_client import DatasiteClient
from .sync_decision import SyncDecision
from .sync_decision import SyncDirection

logger = logging.getLogger(__name__)


def sync(
    from_client: DatasiteClient,
    to_client: DatasiteClient,
    include_ignored: bool = False,
    include_same: bool = False,
    filter_by_email: str | None = None,
    include_types: Collection[str | type] | None = None,
    exclude_types: Collection[str | type] | None = None,
    hide_usercode: bool = True,
) -> PaginatedResolveWidget | SyftSuccess:
    diff = compare_clients(
        from_client=from_client,
        to_client=to_client,
        include_ignored=include_ignored,
        include_same=include_same,
        filter_by_email=filter_by_email,
        include_types=include_types,
        exclude_types=exclude_types,
        hide_usercode=hide_usercode,
    )

    if diff.low_state.errors:
        error_list_text = "<br>".join("- " + x for x in diff.low_state.errors.values())
        warning = SyftWarning(
            message=(
                f"Server {to_client.name} had the following errors while trying to retrieve its sync state. "
                + f"Objects corresponding to these errors will be ignored in comparison.<br>{error_list_text}"
            )
        )
        display(warning)
    if diff.high_state.errors:
        error_list_text = "<br>".join("- " + x for x in diff.high_state.errors.values())
        warning = SyftWarning(
            message=(
                f"Server {to_client.name} had the following errors while trying to retrieve its sync state. "
                + f"Objects corresponding to these errors will be ignored in comparison.<br>{error_list_text}"
            )
        )
        display(warning)

    return diff.resolve()


def compare_states(
    from_state: SyncState,
    to_state: SyncState,
    include_ignored: bool = False,
    include_same: bool = False,
    filter_by_email: str | None = None,
    include_types: Collection[str | type] | None = None,
    exclude_types: Collection[str | type] | None = None,
    hide_usercode: bool = True,
) -> ServerDiff:
    # ServerDiff
    if (
        from_state.server_side_type == ServerSideType.LOW_SIDE
        and to_state.server_side_type == ServerSideType.HIGH_SIDE
    ):
        low_state = from_state
        high_state = to_state
        direction = SyncDirection.LOW_TO_HIGH
    elif (
        from_state.server_side_type == ServerSideType.HIGH_SIDE
        and to_state.server_side_type == ServerSideType.LOW_SIDE
    ):
        low_state = to_state
        high_state = from_state
        direction = SyncDirection.HIGH_TO_LOW
    else:
        raise SyftException(
            public_message="Invalid server side types: can only compare a high and low server"
        )

    if hide_usercode:
        prompt_warning_message(
            "UserCodes are hidden by default, and are part of the Requests."
            " If you want to include them as separate objects, set `hide_usercode=False`"
        )
        exclude_types = exclude_types or []
        exclude_types.append("usercode")

    return ServerDiff.from_sync_state(
        low_state=low_state,
        high_state=high_state,
        direction=direction,
        include_ignored=include_ignored,
        include_same=include_same,
        filter_by_email=filter_by_email,
        include_types=include_types,
        exclude_types=exclude_types,
    )


def compare_clients(
    from_client: DatasiteClient,
    to_client: DatasiteClient,
    include_ignored: bool = False,
    include_same: bool = False,
    filter_by_email: str | None = None,
    include_types: Collection[str | type] | None = None,
    exclude_types: Collection[str | type] | None = None,
    hide_usercode: bool = True,
) -> ServerDiff:
    from_state = from_client.get_sync_state()
    to_state = to_client.get_sync_state()
    return compare_states(
        from_state=from_state,
        to_state=to_state,
        include_ignored=include_ignored,
        include_same=include_same,
        filter_by_email=filter_by_email,
        include_types=include_types,
        exclude_types=exclude_types,
        hide_usercode=hide_usercode,
    )


def resolve(
    obj: ObjectDiffBatch | ServerDiff,
) -> ResolveWidget | PaginatedResolveWidget | SyftSuccess:
    if not isinstance(obj, ObjectDiffBatch | ServerDiff):
        raise ValueError(
            f"Invalid type: could not resolve object with type {type(obj).__qualname__}"
        )
    return obj.resolve()


@deprecated(reason="resolve_single has been renamed to resolve", return_syfterror=True)
def resolve_single(
    obj_diff_batch: ObjectDiffBatch,
) -> ResolveWidget | PaginatedResolveWidget | SyftSuccess:
    return resolve(obj_diff_batch)


def handle_sync_batch(
    obj_diff_batch: ObjectDiffBatch,
    share_private_data: dict[UID, bool],
    mockify: dict[UID, bool],
) -> SyftSuccess:
    # Infer SyncDecision
    sync_direction = obj_diff_batch.sync_direction
    if sync_direction is None:
        raise SyftException(
            message="Cannot sync an object without a specified sync direction."
        )

    decision = sync_direction.to_sync_decision()

    # Validate decision
    if decision not in [SyncDecision.LOW, SyncDecision.HIGH]:
        raise SyftException(public_message="Invalid sync decision")
    elif obj_diff_batch.is_unchanged:
        return SyftSuccess(message="No changes to sync")
    elif obj_diff_batch.decision is SyncDecision.IGNORE:
        raise SyftException(
            public_message="Attempted to sync an ignored object, please unignore first"
        )
    elif obj_diff_batch.decision is not None:
        raise SyftException(
            public_message="Attempted to sync an object that has already been synced"
        )

    src_client = obj_diff_batch.source_client
    tgt_client = obj_diff_batch.target_client
    src_resolved_state, tgt_resolved_state = obj_diff_batch.create_new_resolved_states()

    obj_diff_batch.decision = decision

    sync_instructions = []
    for diff in obj_diff_batch.get_dependencies(include_roots=True):
        # figure out the right verify key to share to
        # in case of a job with user code, share to user code owner
        # without user code, share to job owner
        share_to_user: SyftVerifyKey | None = (
            getattr(obj_diff_batch.user_code_high, "user_verify_key", None)
            or obj_diff_batch.user_verify_key_high
        )
        share_private_data_for_diff = share_private_data[diff.object_id]
        mockify_for_diff = mockify[diff.object_id]
        instruction = SyncInstruction.from_batch_decision(
            diff=diff,
            share_private_data=share_private_data_for_diff,
            mockify=mockify_for_diff,
            sync_direction=sync_direction,
            decision=decision,
            share_to_user=share_to_user,
        )
        sync_instructions.append(instruction)

    logger.debug(f"Decision: Syncing {len(sync_instructions)} objects")

    # Apply sync instructions to target side
    for sync_instruction in sync_instructions:
        tgt_resolved_state.add_sync_instruction(sync_instruction)
        src_resolved_state.add_sync_instruction(sync_instruction)
    # Apply empty state to source side to signal that we are done syncing
    # We also add permissions for users from the low side to mark L0 request as approved
    src_client.apply_state(src_resolved_state)
    return tgt_client.apply_state(tgt_resolved_state)


def handle_ignore_batch(
    obj_diff_batch: ObjectDiffBatch,
    all_batches: list[ObjectDiffBatch],
) -> SyftSuccess:
    if obj_diff_batch.decision is SyncDecision.IGNORE:
        return SyftSuccess(message="This batch is already ignored")
    elif obj_diff_batch.decision is not None:
        raise SyftException(
            message="Attempted to sync an object that has already been synced"
        )

    obj_diff_batch.decision = SyncDecision.IGNORE
    other_batches = [b for b in all_batches if b is not obj_diff_batch]
    other_ignore_batches = get_other_ignore_batches(obj_diff_batch, other_batches)

    for other_batch in other_ignore_batches:
        other_batch.decision = SyncDecision.IGNORE
        logger.debug(f"Ignoring other batch with root {other_batch.root_type.__name__}")

    src_client = obj_diff_batch.source_client
    tgt_client = obj_diff_batch.target_client
    src_resolved_state, tgt_resolved_state = obj_diff_batch.create_new_resolved_states()

    for batch in [obj_diff_batch] + other_ignore_batches:
        src_resolved_state.add_ignored(batch)
        tgt_resolved_state.add_ignored(batch)

    src_client.apply_state(src_resolved_state)
    return tgt_client.apply_state(tgt_resolved_state)


def handle_unignore_batch(
    obj_diff_batch: ObjectDiffBatch,
    all_batches: list[ObjectDiffBatch],
) -> SyftSuccess:
    src_client = obj_diff_batch.source_client
    tgt_client = obj_diff_batch.target_client
    src_resolved_state, tgt_resolved_state = obj_diff_batch.create_new_resolved_states()

    obj_diff_batch.decision = None
    src_resolved_state.add_unignored(obj_diff_batch.root_id)
    tgt_resolved_state.add_unignored(obj_diff_batch.root_id)

    # Unignore dependencies
    other_batches = [b for b in all_batches if b is not obj_diff_batch]
    other_unignore_batches = get_other_unignore_batches(obj_diff_batch, other_batches)
    for other_batch in other_unignore_batches:
        logger.debug(f"Ignoring other batch with root {other_batch.root_type.__name__}")
        other_batch.decision = None
        src_resolved_state.add_unignored(other_batch.root_id)
        tgt_resolved_state.add_unignored(other_batch.root_id)

    src_client.apply_state(src_resolved_state)
    return tgt_client.apply_state(tgt_resolved_state)


def get_other_unignore_batches(
    batch: ObjectDiffBatch,
    other_batches: list[ObjectDiffBatch],
) -> list[ObjectDiffBatch]:
    if batch.decision is not None:
        return []

    other_unignore_batches = []
    required_dependencies = {
        d.object_id for d in batch.get_dependencies(include_roots=True)
    }

    for other_batch in other_batches:
        if other_batch == batch:
            continue
        elif (
            other_batch.decision == SyncDecision.IGNORE
            and other_batch.root_id in required_dependencies
        ):
            other_unignore_batches.append(other_batch)
    return other_unignore_batches


def get_other_ignore_batches(
    batch: ObjectDiffBatch,
    other_batches: list[ObjectDiffBatch],
) -> list[ObjectDiffBatch]:
    if batch.decision != SyncDecision.IGNORE:
        return []

    other_ignore_batches = []
    ignored_ids = {x.object_id for x in batch.get_dependents(include_roots=False)}
    for other_batch in other_batches:
        if other_batch.decision != SyncDecision.IGNORE:
            # Currently, this is not recursive, in the future it might be
            other_batch_ids = {
                d.object_id for d in other_batch.get_dependencies(include_roots=True)
            }
            if len(other_batch_ids & ignored_ids) != 0:
                other_ignore_batches.append(other_batch)
                ignored_ids.update(other_batch_ids)

    return other_ignore_batches
