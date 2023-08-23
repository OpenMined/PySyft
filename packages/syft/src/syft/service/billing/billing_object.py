# stdlib
from datetime import datetime
from typing import List
from typing import Optional

# third party
from pydantic import BaseModel

# relative
from ...serde.serializable import serializable
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject


@serializable()
class ComputeResource(BaseModel):
    name: str
    cloud: str
    instance_type: Optional[str]
    accelerator: Optional[str]
    region: Optional[str]
    disk_size: Optional[int] = 256
    price_unit_cents: int
    time_unit_secs: int = 3600


@serializable()
class BillingResourceUsage(SyftObject):
    # version
    __canonical_name__ = "BillingResourceUsage"
    __version__ = SYFT_OBJECT_VERSION_1

    resource: ComputeResource
    start_time: datetime
    end_time: Optional[datetime] = None


@serializable()
class BillingOverviewObject(BaseModel):
    billing_objects: List[BillingResourceUsage]

    def __repr__(self) -> str:
        return ""

    def _repr_html_(self):
        """Returns a table of different computing resources used"""
        # name, accelerator, cost
        # third party
        import itables
        import pandas as pd

        df = pd.DataFrame(
            [
                {"name": x.name, "accelerator": x.accelerator}
                for x in self.billings_objects
            ]
        )
        return itables.to_html_datatable(df)
