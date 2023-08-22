from typing import Union

from .billing_object import BillingResourceUsage, BillingOverviewObject, ComputeResource
from ...util.telemetry import instrument
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import PartitionSettings
from ...store.document_store import DocumentStore
from ..context import AuthedServiceContext
from ..service import AbstractService
from ..service import service_method
from ..response import SyftError
from ..response import SyftSuccess

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

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = BillingResourceUsageStash(store=store)

    @service_method(path="billing.add_billing", name="add_billing", autosplat=["billing"])
    def add_billing(
        self, context: AuthedServiceContext, billing_object: BillingResourceUsage
    ) -> Union[SyftSuccess, SyftError]:
        """Add a Billing Object"""

        result = self.stash.set(context.credentials, billing_object=billing_object)
        if result.is_ok():
            return SyftSuccess(message=f"BillingObject added: {billing}")
        return SyftError(
            message=f"Failed to add BillingObject {billing}. {result.err()}"
        )

    @service_method(path="billing.get_billing_overview", name="get_billing_overview")
    def get_billing_overview(self):
        result = self.stash.get_all(context.credentials)

        if result.is_ok():
            billing_usages = result.ok()
            return BillingOverviewObject(billing_objects=billing_usages)
        return SyftError(
            message=f"Failed to get Billing Objects {billing}. {result.err()}"
        )
    # def start_billing_item(self, name, price, time_units) -> None:
    #     self.current_billing_item = { # TODO: Use fresh billing class
    #         'name': name,
    #         'price': price,
    #         'time_units': time_units,
    #         'start_time': datetime.now()
    #     }

    # def end_billing_item(self) -> None:
    #     if self.current_billing_item is not None:
    #         self.current_billing_item['end_time'] = datetime.now()

    #         self.current_billing_item = None

    # def get_billing_item(self) -> dict:
    #     return self.current_billing_item