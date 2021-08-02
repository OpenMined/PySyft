# stdlib
from typing import Dict

# third party
from asgi_lifespan import LifespanManager
from httpx import AsyncClient
import pytest

# grid absolute
from app.users.models import UserCreate


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
        base_url="http://localhost",
        headers={"Content-Type": "application/json"},
    ) as client:
        yield client

@pytest.fixture
async def test_user_creation() -> UserCreate:
    return UserCreate(email="test@openmined.org", name="Container Tester", password="changethis", role="Administrator")

@pytest.fixture()
async def logged_in_owner_auth_token_header(app, client) -> Dict[str, str]:
    res = await client.post(app.url_path_for('login'), json={"email": "info@openmined.org", "password": "changethis"})
    res = res.json()
    return {"Authorization": f"Bearer {res['access_token']}"}

