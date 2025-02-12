from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from loguru import logger
from pydantic import BaseModel
from syft_core import Client
from syft_rpc import rpc


@dataclass
class PingRequest:
    msg: str
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PongResponse(BaseModel):
    msg: str
    ts: datetime


def send_ping():
    client = Client.load()
    start = time.time()
    future = rpc.send(
        url=f"syft://{client.email}/api_data/pingpong/rpc/ping",
        body=PingRequest(msg="hello!"),
        expiry="5m",
        cache=True,
    )
    logger.debug(f"Request: {future.request}")

    try:
        response = future.wait(timeout=300)
        response.raise_for_status()
        pong_response = response.model(PongResponse)
        logger.info(f"Response: {pong_response}. Time taken: {time.time() - start}")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    send_ping()
