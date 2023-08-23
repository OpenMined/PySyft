# stdlib
from typing import Annotated
from typing import Dict
from typing import List

# third party
from fastapi import Depends
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi import status
from fastapi.security import HTTPBearer
from pydantic import BaseModel

# syft absolute
from syft.service.billing.billing_object import ComputeResource

app = FastAPI(title="Blue Book", version="0.2.0")
# force to the older version for now
app.openapi_version = "3.0.2"

# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
oauth2_scheme = HTTPBearer()


class UserView(BaseModel):
    username: str


class User(UserView):
    password: str


secret_token = "letmein"
allowed_user = User(username="caleb.smith@bluebook.ai", password="secret")


async def get_current_user(
    request: Request, token: Annotated[str, Depends(oauth2_scheme)]
) -> User:
    # show headers during auth requests
    print(request.headers)
    if token.credentials != secret_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authentication Credentials",
            headers={"www-authenticate": "Bearer"},
        )
    return allowed_user


class ResearchModel(BaseModel):
    name: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str


azure_cpu = ComputeResource(name="azure_cpu", cloud="azure", price_unit_cents=30)

gcp_t4 = ComputeResource(
    name="gcp_t4", cloud="gcp", accelerator="T4", price_unit_cents=70
)

azure_a100 = ComputeResource(
    name="azure_a100",
    cloud="azure",
    region="eastus2",
    instance_type="Standard_NCC24ads_A100_v4",
    accelerator="A100-80GB",
    disk_size=512,
    price_unit_cents=370,
)

all_compute: Dict = {"azure_cpu": azure_cpu, "gcp_t4": gcp_t4, "azure_a100": azure_a100}


api_state: Dict[int, ResearchModel] = {7: ResearchModel(name="Ava")}


@app.post("/login", operation_id="login", summary="Login to the Blue Book API")
# application/x-www-form-urlencoded not supported by openapi3
# => User used instead of OAuth2PasswordRequestForm as form_data
# File "openapi3/paths.py", line 284, in _request_handle_body
#    raise NotImplementedError()
# async def login(form_data: OAuth2PasswordRequestForm = Depends()):
async def login(form_data: Annotated[User, Depends()]) -> LoginResponse:
    if (
        form_data.username != allowed_user.username
        or form_data.password != allowed_user.password
    ):
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    response = {"access_token": secret_token, "token_type": "Bearer"}
    return LoginResponse(**response)


@app.get("/users/me", operation_id="get_me", summary="Get the current user")
async def read_users_me(
    current_user: Annotated[User, Depends(get_current_user)]
) -> UserView:
    return current_user


@app.get("/", operation_id="home", summary="Home Page")
def read_root(request: Request) -> Dict:
    print("headers", request.headers)
    return {}


@app.get("/models/", operation_id="get_all", summary="Get all the Models")
def get_all(
    current_user: Annotated[User, Depends(get_current_user)]
) -> List[ResearchModel]:
    return list(api_state.values())


@app.get("/models/{model_id}", operation_id="get_model", summary="Get a Model by index")
def get_model(
    current_user: Annotated[User, Depends(get_current_user)], model_id: int
) -> ResearchModel:
    # do not return Optional[ResearchModel] as it breaks the openapiv3 library
    model = api_state.get(model_id, None)
    return model  # type: ignore


@app.put("/models/{model_id}", operation_id="set_model", summary="Set a Model by index")
def set_model(
    current_user: Annotated[User, Depends(get_current_user)],
    model_id: int,
    model: ResearchModel,
) -> ResearchModel:
    api_state[model_id] = model
    return model


@app.get(
    "/compute/", operation_id="get_all_compute", summary="Get all the Compute Options"
)
def get_all_compute(
    current_user: Annotated[User, Depends(get_current_user)]
) -> List[ComputeResource]:
    return list(all_compute.values())


@app.get(
    "/compute/{compute_name}",
    operation_id="get_compute_config",
    summary="Get Compute Config",
)
def get_compute(
    current_user: Annotated[User, Depends(get_current_user)], compute_name: str
) -> ComputeResource:
    compute = all_compute.get(compute_name, None)
    return compute  # type: ignore
