from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class ComputeResource(BaseModel):
    name: str
    cloud: str
    instance_type: Optional[str]
    accelerator: Optional[str]
    region: Optional[str]
    disk_size: Optional[int] = 256
    price_unit_cents: int
    time_unit_secs: int = 3600
    
class BillingResourceUsage(BaseModel):
    pass


class BillingResourceUsage(BaseModel):
    resource: ComputeResource
    start_time: datetime
    end_time: Optional[datetime]

