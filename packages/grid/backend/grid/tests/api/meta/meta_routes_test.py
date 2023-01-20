# third party
from fastapi import FastAPI
from httpx import AsyncClient
import pytest


class TestUsersRoutes:
    @pytest.mark.asyncio
    async def test_ping_status(self, app: FastAPI, client: AsyncClient) -> None:
        """Test grid client status API."""

        res = await client.get(app.url_path_for("ping"))
        assert res is not None
        assert res.status_code == 200
        assert res.json() == "pong"
