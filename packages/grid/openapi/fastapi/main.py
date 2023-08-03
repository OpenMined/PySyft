# stdlib
from typing import Annotated
from typing import Dict
from typing import List
from typing import Optional

# third party
from fastapi import Depends
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

app = FastAPI(title="Blue Book", version="0.2.0")


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


class UserOut(BaseModel):
    username: str


class User(UserOut):
    password: str


fake_user = User(username="johndoe", password="secret")


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> User:
    if token != fake_user.username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return fake_user


class ResearchModel(BaseModel):
    name: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str


api_state: Dict[int, ResearchModel] = {7: ResearchModel(name="Ava")}


@app.post("/login", operation_id="login", summary="Login to the Blue Book API")
# application/x-www-form-urlencoded not supported by openapi3
# => User used instead of OAuth2PasswordRequestForm as form_data
# File "openapi3/paths.py", line 284, in _request_handle_body
#    raise NotImplementedError()
# async def login(form_data: OAuth2PasswordRequestForm = Depends()):
async def login(form_data: Annotated[User, Depends()]) -> LoginResponse:
    if (
        form_data.username != fake_user.username
        or form_data.password != fake_user.password
    ):
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    response = {"access_token": fake_user.username, "token_type": "bearer"}
    return LoginResponse(**response)


@app.get("/users/me", operation_id="get_me", summary="Get the current user")
async def read_users_me(
    current_user: Annotated[User, Depends(get_current_user)]
) -> UserOut:
    return current_user


@app.get("/", operation_id="home", summary="Home Page")
def read_root() -> Dict:
    return {"Caleb": "Smith"}


@app.get("/models/", operation_id="get_all", summary="Get all the Models")
def get_all() -> List[ResearchModel]:
    return list(api_state.values())


@app.get("/models/{model_id}", operation_id="get_model", summary="Get a Model by index")
def get_model(model_id: int) -> Optional[ResearchModel]:
    model = api_state.get(model_id, None)
    return model


@app.put("/models/{model_id}", operation_id="set_model", summary="Set a Model by index")
def set_model(model_id: int, model: ResearchModel) -> ResearchModel:
    api_state[model_id] = model
    return model
