# stdlib
from typing import Optional
from typing import Union

# third party
from pydantic import BaseModel


class AssociationRequestModel(BaseModel):
    association_id: str
    requested_date: str
    name: str
    email: str
    status: str
    reason: Optional[str] = None
    node_address: str
    node_name: str
    node_id: str
