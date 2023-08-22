# third party
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

# relative
from ...types.syft_object import SyftObject
from ...types.syft_object import SYFT_OBJECT_VERSION_1

class ComputeResource(SyftObject):
    # version
    __canonical_name__ = "ComputeResource"
    __version__ = SYFT_OBJECT_VERSION_1

    name: str
    cloud: str
    instance_type: Optional[str]
    accelerator: Optional[str]
    region: Optional[str]
    disk_size: Optional[int] = 256
    price_unit_cents: int
    time_unit_secs: int = 3600


class BillingResourceUsage(SyftObject):
    # version
    __canonical_name__ = "B"
    __version__ = SYFT_OBJECT_VERSION_1

    resource: ComputeResource
    start_time: datetime
    end_time: Optional[datetime]

