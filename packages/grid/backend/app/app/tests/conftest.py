# third party
from asgi_lifespan import LifespanManager
from httpx import AsyncClient
import pytest


@pytest.fixture
async def app():
    # grid absolute
    from app.main import app
    async with LifespanManager(app):
        yield app

@pytest.fixture
async def client(app) -> AsyncClient:
    async with AsyncClient(
        app=app,
        base_url="http://localhost", # TODO: Use base_url from config 
        headers={"Content-Type": "application/json"},
    ) as client:
        yield client


