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
from ...util import options
from ...util.colors import ON_SURFACE_HIGHEST
from ...util.colors import SURFACE_SURFACE


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

    def get_costs(self, start_time, end_time=None):
        end_time = end_time or datetime.now()
        usage_units = (end_time - start_time).seconds / self.time_unit_secs
        return round((usage_units * self.price_unit_cents) / 100, 2)


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
        # TODO: fix this
        itables_css = f"""
        .itables table {{
            margin: 0 auto;
            float: left;
            color: {ON_SURFACE_HIGHEST[options.color_theme]};
        }}
        .itables table th {{color: {SURFACE_SURFACE[options.color_theme]};}}
        """
        # third party
        import itables
        import pandas as pd

        df = pd.DataFrame(
            [
                {
                    "name": x.resource.name,
                    "accelerator": x.resource.accelerator,
                    "costs": x.resource.get_costs(x.start_time, x.end_time),
                }
                for x in self.billing_objects
            ]
        )
        return itables.to_html_datatable(df, css=itables_css)
