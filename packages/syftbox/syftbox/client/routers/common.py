from fastapi import Depends, Request
from typing_extensions import Annotated

from syftbox.client.base import SyftBoxContextInterface

__all__ = ["APIContext"]


# Create a dependency for typed access to the client
async def get_context(request: Request) -> SyftBoxContextInterface:
    return request.app.state.context


APIContext = Annotated[SyftBoxContextInterface, Depends(get_context)]
