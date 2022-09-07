# stdlib
import json

# third party
from faker import Faker
from fastapi import FastAPI
from httpx import AsyncClient
import pytest
from starlette import status


# TODO:  Ionesio and Rasswanth, fix after adding pymongo inmemory db.
@pytest.mark.skip
class TestAuthRoutes:
    @pytest.mark.asyncio
    async def test_user_register(
        self, app: FastAPI, client: AsyncClient, faker: Faker
    ) -> None:

        user = {
            "email": faker.email(),
            "password": faker.password(),
            "name": faker.name(),
            "budget": 0.0,
        }
        res = await client.post(app.url_path_for("register"), data=json.dumps(user))
        assert res.status_code == status.HTTP_200_OK

        with pytest.raises(Exception):
            # TODO: Send a valid message instead of raising exception in case of duplicate entry
            res = await client.post(app.url_path_for("register"), data=json.dumps(user))
