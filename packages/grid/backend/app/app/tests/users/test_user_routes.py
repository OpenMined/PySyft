# third party
from fastapi import FastAPI
from httpx import AsyncClient
import pytest
from starlette import status

# grid absolute
from app.tests.utils.auth import authenticate_owner
from app.tests.utils.user import create_user


class TestUsersRoutes:
    @pytest.mark.asyncio
    async def test_routes_exist(self, app: FastAPI) -> None:
        for route_name in ["users:me", "users:create", "users:read_all"]:
            assert app.url_path_for(route_name) is not None
        for route_name in ["users:read_one", "users:update", "users:delete"]:
            assert app.url_path_for(route_name, **{"user_id": 1}) is not None

    @pytest.mark.asyncio
    async def test_unauthenticated_user_cannot_create_user(
        self, app: FastAPI, client: AsyncClient
    ) -> None:
        user = create_user()
        res = await client.post(app.url_path_for("users:create"), json=dict(user))
        assert "Authorization" not in res.request.headers.keys()
        assert res.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_successfully_create_user(
        self, app: FastAPI, client: AsyncClient
    ) -> None:
        user = create_user()
        headers = await authenticate_owner(app, client)
        res = await client.post(
            app.url_path_for("users:create"), json=dict(user), headers=headers
        )
        assert res.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_list_users(self, app: FastAPI, client: AsyncClient) -> None:
        headers = await authenticate_owner(app, client)
        res = await client.get(app.url_path_for("users:read_all"), headers=headers)
        assert res.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_get_specific_user(self, app: FastAPI, client: AsyncClient) -> None:
        headers = await authenticate_owner(app, client)
        res = await client.get(
            app.url_path_for("users:read_one", **{"user_id": 1}), headers=headers
        )
        assert res.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_invalid_user_raises_errors(
        self, app: FastAPI, client: AsyncClient
    ) -> None:
        headers = await authenticate_owner(app, client)
        res = await client.post(
            app.url_path_for("users:create"), data={}, headers=headers
        )
        assert res.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        invalid_user = create_user(email="info@openmined.org")
        res = await client.post(
            app.url_path_for("users:create"), json=dict(invalid_user), headers=headers
        )
        assert res.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        invalid_user = create_user()
        invalid_user = invalid_user.dict(exclude={"role"})
        invalid_user["role"] = 1
        res = await client.post(
            app.url_path_for("users:create"), json=invalid_user, headers=headers
        )
        assert res.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
