# stdlib
from typing import Dict

# third party
from fastapi import FastAPI
from httpx import AsyncClient
import pytest
from starlette import status

# grid absolute
from app.users.models import UserCreate

class TestUsersRoutes:
    @pytest.mark.asyncio
    async def test_routes_exist(self, app: FastAPI) -> None:
        for route_name in ['users:me', 'users:create', 'users:read_all']:
            assert app.url_path_for(route_name) is not None
        for route_name in ['users:read_one', 'users:update', 'users:delete']:
            assert app.url_path_for(route_name, **{"user_id": 1}) is not None
    
    @pytest.mark.asyncio
    async def test_unauthenticated_user_cannot_create_user(self, app: FastAPI, client: AsyncClient, create_test_user: UserCreate) -> None:
        res = await client.post(app.url_path_for("users:create"), json=dict(create_test_user))
        assert res.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_successfully_create_user(self, app: FastAPI, client: AsyncClient, create_test_user: UserCreate, logged_in_owner_auth_token_header: Dict[str, str]) -> None:
        res = await client.post(app.url_path_for("users:create"), json=dict(create_test_user), headers=logged_in_owner_auth_token_header)
        assert res.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_list_users(self, app: FastAPI, client: AsyncClient, logged_in_owner_auth_token_header: Dict[str, str]) -> None:
        res = await client.get(app.url_path_for("users:read_all"), headers=logged_in_owner_auth_token_header)
        assert res.status_code == status.HTTP_200_OK
    
    @pytest.mark.asyncio
    async def test_get_specific_user(self, app: FastAPI, client: AsyncClient, logged_in_owner_auth_token_header: Dict[str, str]) -> None:
        res = await client.get(app.url_path_for('users:read_one', **{"user_id": 1}), headers=logged_in_owner_auth_token_header)
        assert res.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_invalid_user_raises_error(self, app: FastAPI, client: AsyncClient, logged_in_owner_auth_token_header: Dict[str, str]) -> None:
        res = await client.post(app.url_path_for("users:create"), data={}, headers=logged_in_owner_auth_token_header)
        assert res.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

