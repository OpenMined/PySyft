from __future__ import annotations

from datetime import datetime, timezone

from loguru import logger
from pydantic import BaseModel, Field
from syft_event import SyftEvents
from syft_event.types import Request

box = SyftEvents("pingpong")


class PingRequest(BaseModel):
    """Request to send a ping."""

    msg: str = Field(description="Ping request string")
    ts: datetime = Field(description="Timestamp of the ping request.")


class PongResponse(BaseModel):
    """Response to a ping request."""

    msg: str = Field(description="Ping response string")
    ts: datetime = Field(description="Timestamp of the pong response.")


@box.on_request("/ping")
def pong(ping: PingRequest, ctx: Request) -> PongResponse:
    """Respond to a ping request."""

    logger.info(f"Got ping request - {ping}")
    return PongResponse(
        msg=f"Pong from {box.client.email}",
        ts=datetime.now(timezone.utc),
    )


if __name__ == "__main__":
    try:
        print("Running rpc server for", box.app_rpc_dir)
        box.run_forever()
    except Exception as e:
        print(e)
