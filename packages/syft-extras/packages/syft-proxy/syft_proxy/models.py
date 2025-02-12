from enum import Enum

from pydantic import BaseModel, Field
from syft_rpc import SyftRequest, SyftResponse
from syft_rpc.rpc import DEFAULT_EXPIRY
from typing_extensions import Any, List, Optional, Union


class RPCRequestBase(BaseModel):
    body: Any
    headers: dict[str, str] = Field(default_factory=dict)
    expiry: str = Field(default=DEFAULT_EXPIRY)
    cache: bool = Field(default=False)


class RPCSendRequest(RPCRequestBase):
    app_name: str = Field(..., min_length=3)
    url: str

    # @field_validator("url", mode="after")
    # def validate_url(cls, v):
    #     """
    #     Validates the URL to ensure it starts with the required scheme.

    #     Args:
    #         cls: The class that this method belongs to.
    #         v: The URL string to validate.

    #     Raises:
    #         ValueError: If the URL does not start with "syft://".

    #     Returns:
    #         The validated URL string if it is valid.
    #     """
    #     if not v.startswith("syft://"):
    #         raise ValueError('URL must start with "syft://"')
    #     return v


class RPCBroadcastRequest(RPCRequestBase):
    urls: List[str]

    # @field_validator("urls", mode="after")
    # def validate_urls(cls, v):
    #     """
    #     Validates the list of URLs to ensure it is not empty and that each URL starts with the required scheme.

    #     Args:
    #         cls: The class that this method belongs to.
    #         v: The list of URL strings to validate.

    #     Raises:
    #         ValueError: If the list of URLs is empty or if any URL does not start with "syft://".

    #     Returns:
    #         The validated list of URL strings if all are valid.
    #     """
    #     if not v:
    #         raise ValueError("The list of URLs must not be empty")
    #     for url in v:
    #         if not url.startswith("syft://"):
    #             raise ValueError(f'URL "{url}" must start with "syft://"')
    #     return v


class RPCBroadcastResult(BaseModel):
    id: str
    requests: List[SyftRequest]


class RPCStatusCode(Enum):
    NOT_FOUND = "RPC_NOT_FOUND"
    PENDING = "RPC_PENDING"
    COMPLETED = "RPC_COMPLETED"
    ERROR = "RPC_ERROR"


class RPCStatus(BaseModel):
    id: str
    status: RPCStatusCode
    request: Optional[Union[SyftRequest, List[SyftRequest]]]
    response: Optional[Union[SyftResponse, List[SyftResponse]]]
