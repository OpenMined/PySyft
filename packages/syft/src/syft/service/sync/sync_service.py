# stdlib
from typing import Optional

# relative
from ...store.document_store import DocumentStore
from ..context import AuthedServiceContext
from ..service import AbstractService
from ..service import service_method
from ..user.user_roles import ADMIN_ROLE_LEVEL
from .sync_stash import SyncStash
from .sync_state import SyncState


class SyncService(AbstractService):
    store: DocumentStore
    stash: SyncStash

    def __init__(self, store: DocumentStore):
        self.store = store
        self.stash = SyncStash(store=store)

    @service_method(
        path="sync.get_state",
        name="get_state",
        roles=ADMIN_ROLE_LEVEL,
    )
    def get_state(
        self, context: AuthedServiceContext, add_to_store: bool = False
    ) -> Optional[SyncState]:
        new_state = SyncState()

        node = context.node

        projects = node.get_service("projectservice").get_all(context)
        new_state.add_objects(projects)

        requests = node.get_service("requestservice").get_all(context)
        new_state.add_objects(requests)

        user_codes = node.get_service("usercodeservice").get_all(context)
        new_state.add_objects(user_codes)

        jobs = node.get_service("jobservice").get_all(context)
        new_state.add_objects(jobs)

        logs = node.get_service("logservice").get_all(context)
        new_state.add_objects(logs)

        # TODO workaround, we only need action objects from output policies for now
        action_objects = []
        for code in user_codes:
            action_objects.extend(code.get_all_output_action_objects())
        for job in jobs:
            if job.result is not None:
                action_objects.append(job.result)
        new_state.add_objects(action_objects)

        new_state._build_dependencies()

        # TODO
        # previous_state = self.stash.get_latest(context=context)
        # if previous_state is not None:
        #     new_state.previous_state_link = LinkedObject.from_obj(
        #         obj=previous_state,
        #         service_type=SyncService,
        #         node_uid=context.node.id,
        #     )

        # if add_to_store:
        #     self.stash.add(new_state, context=context)

        return new_state
