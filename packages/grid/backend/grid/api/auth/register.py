# stdlib
from typing import Any

# third party
from fastapi import APIRouter
from fastapi import Body
from fastapi.responses import JSONResponse

# grid absolute
from grid.core.config import settings
from grid.core.node import node

router = APIRouter()


@router.post("/register", name="register", status_code=200, response_class=JSONResponse)
def register(data: dict = Body(..., example="sheldon@caltech.edu")) -> Any:

    if not settings.OPEN_REGISTRATION:
        return {
            "error": "This node doesn't allow for anyone to register a user. Please contact the domain"
            + " administrator who can setup a user account for you."
        }

    if "name" not in data:
        return {"error": "Missing 'name' attribute. Please try again"}

    if "email" not in data:
        return {"error": "Missing 'email' attribute. Please try again"}

    if "password" not in data:
        return {"error": "Missing 'password' attribute. Please try again"}

    node.get_root_client().users.create(
        **{
            "name": data["name"],
            "email": data["email"],
            "password": data["password"],
            "budget": 0,
        }
    )

    return {"message": "Success! You've now created a new user!"}
