# stdlib
from random import random

# third party
from fastapi import FastAPI
from httpx import AsyncClient
import pytest
from starlette import status

# grid absolute
from grid.tests.utils.auth import authenticate_owner
from grid.tests.utils.user import create_user


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
        user = dict(user)  # type: ignore
        user["daa_pdf"] = ""  # type: ignore
        res = await client.post(app.url_path_for("users:create"), json=user)
        assert "Authorization" not in res.request.headers.keys()
        assert res.status_code == status.HTTP_401_UNAUTHORIZED

    """
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
    """

    @pytest.mark.asyncio
    async def test_successfully_update_user(
        self, app: FastAPI, client: AsyncClient
    ) -> None:
        user = create_user(budget=random() * 100)
        headers = await authenticate_owner(app, client)
        data = {
            "name": user.name,
            "budget": user.budget,
            "institution": user.institution,
            "website": f"www.{user.institution}.com",
        }

        # Update the information for the given user
        res = await client.patch(
            app.url_path_for("users:update", **{"user_id": 1}),
            json=data,
            headers=headers,
        )

        # Check if the request was successful
        assert res.status_code == status.HTTP_204_NO_CONTENT

        # Get the details of the update user
        res = await client.get(
            app.url_path_for("users:read_one", **{"user_id": 1}), headers=headers
        )

        assert res.status_code == status.HTTP_200_OK
        res_json = res.json()

        # Check if the user details were updated correctly
        for key, val in data.items():
            assert res_json.get(key) == val

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
    async def test_user_cannot_delete_itself(
        self, app: FastAPI, client: AsyncClient
    ) -> None:
        headers = await authenticate_owner(app, client)

        res = await client.delete(
            app.url_path_for("users:delete", **{"user_id": 1}), headers=headers
        )
        assert res.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert res.json() == {"detail": "There was an error processing your request."}

    """
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
        invalid_user = invalid_user.dict(exclude={"role"})  # type: ignore
        invalid_user["role"] = 1  # type: ignore
        res = await client.post(
            app.url_path_for("users:create"), json=invalid_user, headers=headers
        )
        assert res.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    """
