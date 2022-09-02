# stdlib
import json
from random import random

# third party
from fastapi import FastAPI
from httpx import AsyncClient
import pytest
from sqlalchemy.orm import Session
from starlette import status

# syft absolute
from syft.core.node.common.node_manager.user_manager import NoSQLUserManager

# grid absolute
from grid.tests.utils.auth import OWNER_EMAIL
from grid.tests.utils.auth import OWNER_PWD
from grid.tests.utils.auth import authenticate_owner
from grid.tests.utils.auth import authenticate_user
from grid.tests.utils.common import random_lower_string
from grid.tests.utils.user import create_user


# TODO:  Ionesio and Rasswanth, fix after adding pymongo inmemory db.
@pytest.mark.skip
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

    @pytest.mark.asyncio
    async def test_successfully_create_user(
        self, app: FastAPI, client: AsyncClient
    ) -> None:

        # Create dummy user
        user = create_user(budget=random() * 100)
        headers = await authenticate_owner(app, client)
        new_user = {
            "name": user.name,
            "email": user.email,
            "password": user.password,
            "confirm_password": user.password,
            "role": 1,
            "budget": user.budget,
        }
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        data = {"new_user": json.dumps(new_user), "file": None}

        res = await client.post(
            app.url_path_for("users:create"), headers=headers, data=data
        )

        assert res.status_code == status.HTTP_201_CREATED
        assert res.json() == "User created successfully!"

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
    async def test_fail_change_user_password(
        self, app: FastAPI, client: AsyncClient
    ) -> None:
        headers = await authenticate_owner(app, client)
        new_pwd = random_lower_string()
        data = {
            "password": "wrong_pwd",  # wrong password
            "new_password": new_pwd,
        }

        # Update the information for the given user
        res = await client.patch(
            app.url_path_for("users:update", **{"user_id": 1}),
            json=data,
            headers=headers,
        )

        # Check if it raises unknown private exception triggered by invalid credentials.
        assert res.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

        # Check if the user credentails remains the same
        headers = await authenticate_owner(app, client)

    @pytest.mark.asyncio
    async def test_successfully_change_user_password(
        self, app: FastAPI, client: AsyncClient
    ) -> None:
        headers = await authenticate_owner(app, client)
        new_pwd = random_lower_string()
        data = {
            "password": OWNER_PWD,
            "new_password": new_pwd,
        }

        # Update the information for the given user
        res = await client.patch(
            app.url_path_for("users:update", **{"user_id": 1}),
            json=data,
            headers=headers,
        )

        # Check if the request was successful
        assert res.status_code == status.HTTP_204_NO_CONTENT

        # Check if the user details were updated correctly
        headers = await authenticate_user(
            app, client, email=OWNER_EMAIL, password=new_pwd
        )
        assert "Authorization" in headers.keys()

        # Return to the standard password
        data = {
            "new_password": OWNER_PWD,
            "password": new_pwd,
        }
        # Update the information for the given user
        res = await client.patch(
            app.url_path_for("users:update", **{"user_id": 1}),
            json=data,
            headers=headers,
        )

        # Check if the request was successful
        assert res.status_code == status.HTTP_204_NO_CONTENT

    @pytest.mark.asyncio
    async def test_fail_change_owner_role(
        self, app: FastAPI, client: AsyncClient
    ) -> None:
        headers = await authenticate_owner(app, client)

        # Update the information for the given user
        res = await client.patch(
            app.url_path_for("users:update", **{"user_id": 1}),
            json={"role": "Data Scientist"},
            headers=headers,
        )
        # Inside of the node it should raise an UnauthorizedException,
        # then UnknownPrivateException and finally return 500 as a http response code.
        assert res.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

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

    @pytest.mark.asyncio
    async def test_delete_user(
        self, app: FastAPI, client: AsyncClient, db: Session
    ) -> None:
        # Create a data for dummy user
        user = create_user(budget=random() * 100)
        headers = await authenticate_owner(app, client)

        db_engine = db.get_bind()
        # FIXME: modify to nosql in memory db engine
        user_manager = NoSQLUserManager(db_engine)

        # create new user
        user_manager.signup(
            name=user.name,
            email=user.email,
            password=user.password,
            budget=user.budget,
            role=1,  # Assign Data Scientist as Role
            verify_key="",
            private_key="",
        )

        # retrieve the new user
        new_user = user_manager.last(**{"email": user.email})
        assert new_user is not None

        # Deleting the new user
        res = await client.delete(
            app.url_path_for("users:delete", **{"user_id": int(new_user.id)}),
            headers=headers,
        )
        assert res.status_code == status.HTTP_204_NO_CONTENT

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
