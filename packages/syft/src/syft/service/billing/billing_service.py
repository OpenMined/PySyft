from typing import Union

from .billing_object import BillingResourceUsage
from ...util.telemetry import instrument
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash

@instrument
@serializable()
class BillingResourceUsageStash(BaseUIDStoreStash):
    object_type = BillingResourceUsage
    settings: PartitionSettings = PartitionSettings(
        name=BillingResourceUsage.__canonical_name__, object_type=BillingResourceUsage
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)


@instrument
@serializable()
class BillingService(AbstractService):
    store: DocumentStore
    stash: BillingResourceUsageStash
    command_stash: ContainerCommandStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = ContainerImageStash(store=store)

    @service_method(path="billing.add_billing", name="add_billing", autosplat=["image"])
    def add_billing(
        self, context: AuthedServiceContext, billing_object: BillingResourceUsage
    ) -> Union[SyftSuccess, SyftError]:
        """Add a Billing Object"""

        result = self.stash.set(context.credentials, billing_object=billing_object)
        if result.is_ok():
            return SyftSuccess(message=f"ContainerImage added: {image}")
        return SyftError(
            message=f"Failed to add ContainerImage {image}. {result.err()}"
        )

    