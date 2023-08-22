# third party
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

# relative
from ...serde.serializable import serializable
from ...types.syft_object import SyftObject
from ...types.syft_object import SYFT_OBJECT_VERSION_1

@serializable()
class ComputeResource(BaseModel):
    # version
    # __canonical_name__ = "ComputeResource"
    # __version__ = SYFT_OBJECT_VERSION_1

    # name: str
    # cloud: str
    # instance_type: Optional[str]
    # accelerator: Optional[str]
    # region: Optional[str]
    # disk_size: Optional[int] = 256
    # price_unit_cents: int
    # time_unit_secs: int = 3600

    name: str
    cloud: str
    instance_type: str = ""
    accelerator: str = ""
    region: str = ""
    disk_size: int = 256
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
    # __canonical_name__ = "BillingOverviewObject"
    # __version__ = SYFT_OBJECT_VERSION_1

    billing_objects: List[BillingResourceUsage]

    def __repr__(self) -> str:
        return ''

    def _repr_html_(self):
        """Returns a table of different computing resources used"""
        # name, accelerator, cost
        import pandas as pd
        import itables
        

        df = pd.DataFrame([{"name": x.name,"accelerator": x.accelerator} for x in self.billings_objects])
        return itables.to_html_datatable(df)