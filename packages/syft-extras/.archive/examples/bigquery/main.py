import json
import os
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

import uvicorn
from authlib.integrations.starlette_client import OAuth
from dynaconf import Dynaconf
from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.security import OAuth2AuthorizationCodeBearer
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from google.cloud import bigquery  # noqa: F811
from google.oauth2 import service_account
from jose import JWTError, jwt
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from data import schema_dict
from syftbox.lib import Client, SyftPermission

client_config = Client.load()
datasite = "bigquery@openmined.org"


def get_settings():
    current_dir = Path(__file__).parent
    config_file = current_dir / "settings.yaml"
    settings = Dynaconf(
        settings_files=[config_file],
        environments=True,
    )
    return settings


settings = get_settings()
app = FastAPI()

# Add Session Middleware (Required for OAuth)
app.add_middleware(SessionMiddleware, secret_key="!secret")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# SERVER_URL = "http://bigquery-.madhava-openmined-org.syftbox.localhost/"
# SERVER_URL = "http://localhost/bigquery/auth/google/callback"
# SERVER_URL = "http://bigquery-openmined-org.syftbox.openmined.dev/bigquery/"
# SERVER_URL = "http://localhost/bigquery"
# SERVER_URL = "http://localhost:9081/bigquery"
DOMAIN = "bigquery-openmined-org.syftbox.openmined.dev"
SERVER_URL = f"https://{DOMAIN}/bigquery"


ACCESS_TOKEN_EXPIRE_MINUTES = 30
GOOGLE_CALLBACK_URL = f"{SERVER_URL}/auth/google/callback"


oauth2_google = OAuth2AuthorizationCodeBearer(
    authorizationUrl="https://accounts.google.com/o/oauth2/v2/auth",
    tokenUrl="https://accounts.google.com/o/oauth2/v2/token",
)

# Google OAuth configuration
oauth = OAuth()
oauth.register(
    name="google",
    client_id=settings.google_auth_client_id,
    client_secret=settings.google_auth_client_secret,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)


# User model
class User(BaseModel):
    email: str
    name: str


# JWT secret and algorithm
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"


# Helper function to create JWT tokens
def create_access_token(data: dict):
    to_encode = data.copy()
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_current_user_from_cookie(request: Request) -> dict | None:
    token = request.cookies.get("access_token")
    if not token:
        return None

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        name: str = payload.get("name")
        if email is None or name is None:
            return None
        return {"email": email, "name": name}
    except JWTError:
        return None


def validate_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        name: str = payload.get("name")
        if email is None or name is None:
            raise HTTPException(status_code=403, detail="Invalid token")
        return {"email": email, "name": name}
    except JWTError:
        raise HTTPException(status_code=403, detail="Invalid token or token expired")


@app.get("/", include_in_schema=False)
async def home(request: Request, user: dict = Depends(get_current_user_from_cookie)):
    logged_in_html = ""
    if user:
        logged_in_html = f'Hello {user["name"]}, <a href="/bigquery/logout">Logout</a>'
    else:
        logged_in_html = '<a href="/bigquery/auth/google">Login</a>'

    output = f"""
<h1>Big Query</h1>
{logged_in_html}
<br />
<a href="/bigquery/users/me">Account</a>
</br>
<a href="/bigquery/sql">Submit SQL</a>
<br />Client ID: {settings.google_auth_client_id}
<br />CALLBACK URL: {GOOGLE_CALLBACK_URL}
"""

    return HTMLResponse(output)


@app.get("/sql", response_class=HTMLResponse, include_in_schema=False)
async def get_sql_page(request: Request):
    return templates.TemplateResponse("submit_sql.html", {"request": request})


def get_submission_dir(email: str) -> Path:
    sync_folder = Path(client_config.sync_folder)
    submissions = sync_folder / "submissions"
    os.makedirs(submissions, exist_ok=True)
    perm = SyftPermission.mine_no_permission(email=datasite)
    perm.ensure(submissions)
    user_submissions = submissions / email
    os.makedirs(user_submissions, exist_ok=True)
    return user_submissions


# Custom Pydantic model to represent the schema
class SubmitSQL(BaseModel):
    sql_query: str
    token: str


class DownloadSQL(BaseModel):
    sql_query: str
    token: str
    syft_link: str


class SQLJob(BaseModel):
    sql_query: str
    email: str
    created_time: float


@app.post(
    "/submit-sql",
    operation_id="submit_sql",
    summary="Submit SQL",
    description="Submits SQL to BigQuery",
    tags=["MyBigQuery"],
)
async def submit_sql(form: SubmitSQL = Form(...)):
    user = validate_token(form.token)
    email = user["email"]
    user_submissions = get_submission_dir(email)

    now = datetime.now().timestamp()
    job = SQLJob(created_time=now, sql_query=form.sql_query, email=email)

    with open(user_submissions / f"{int(now)}_sql.json", "w") as f:
        f.write(job.model_dump_json())

    return {"message": "SQL query submitted", "sql_query": form.sql_query}


@app.get(
    "/list-sql",
    operation_id="list_sql",
    summary="List SQL",
    description="Lists SQL Submissions",
    tags=["MyBigQuery"],
)
async def list_sql(token: str) -> list[SQLJob]:
    user = validate_token(token)
    email = user["email"]
    user_submissions = get_submission_dir(email)

    jobs = []
    for submission in os.listdir(user_submissions):
        try:
            with open(user_submissions / submission, "r") as f:
                data = f.read()
                job = SQLJob(**json.loads(data))
                jobs.append(job)
        except Exception as e:
            print(f"Failed to load: {submission}. {e}")
    return jobs


# Google login route to redirect the user to Google's OAuth login page
@app.get(
    "/auth/google",
    include_in_schema=False,
)
async def google_login(request: Request):
    redirect_uri = GOOGLE_CALLBACK_URL
    return await oauth.google.authorize_redirect(request, redirect_uri)


# Callback route to handle Google OAuth callback and get user info
@app.get(
    "/auth/google/callback",
    include_in_schema=False,
)
async def google_callback(request: Request):
    token = await oauth.google.authorize_access_token(request)
    userinfo = token["userinfo"]
    email = userinfo["email"]
    name = userinfo["name"]

    if not email:
        raise HTTPException(status_code=400, detail="Failed to fetch email from Google")

    # Generate JWT token for the user
    access_token = create_access_token({"sub": email, "name": name})

    response = RedirectResponse(url="/bigquery")
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        domain=DOMAIN,
    )

    return response


@app.get("/logout")
async def logout(response: Response):
    response = RedirectResponse(url="/bigquery")
    # Clear the cookie by setting it with an empty value and expiry in the past
    response.delete_cookie(key="access_token")
    return response


@app.get("/users/me")
async def get_current_user(
    request: Request, user: dict = Depends(get_current_user_from_cookie)
):
    token = request.cookies.get("access_token")
    return {"email": user["email"], "name": user["name"], "token": token}


@app.get("/healthcheck", include_in_schema=False)
async def healthcheck():
    return {"status": "healthy"}


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="mybigquery",
        version="1.0.0",
        summary="This is a very custom OpenAPI schema",
        description="Here's a longer description of the custom **OpenAPI** schema",
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.get(
    "/schema_live",
    operation_id="schema_live",
    tags=["MyBigQuery"],
)
async def schema_live(token: str) -> list | dict:
    user = validate_token(token)
    email = user["email"]
    print("user", email)

    print("settings", settings)

    # Auth for Bigquer based on the workload identity

    creds = settings["gce_service_account"]
    credentials = service_account.Credentials.from_service_account_info(creds)
    scoped_credentials = credentials.with_scopes(
        ["https://www.googleapis.com/auth/cloud-platform"]
    )

    client = bigquery.Client(
        credentials=scoped_credentials,
        location=settings["gce_region"],
    )

    try:
        # Formats the data schema in a data frame format
        # Warning: the only supported format types are primitives, np.ndarrays and pd.DataFrames

        data_schema = []
        for table_id in [settings.table_1, settings.table_2]:
            project_id = settings.gce_project_id
            dataset = settings.dataset_2
            table = client.get_table(f"{project_id}.{dataset}.{table_id}")
            for schema in table.schema:
                data_schema.append(
                    {
                        "project": str(table.project),
                        "dataset_id": str(table.dataset_id),
                        "table_id": str(table.table_id),
                        "schema_name": str(schema.name),
                        "schema_field": str(schema.field_type),
                        "description": str(table.description),
                        "num_rows": str(table.num_rows),
                    }
                )
        return data_schema

    except Exception as e:
        # not a bigquery exception
        print(e)
        return {"message": "error"}


@app.get(
    "/schema_mock",
    operation_id="schema_mock",
    tags=["MyBigQuery"],
)
async def schema_mock() -> list | dict:
    return [schema_dict]


def to_path(client_config, syft_path: str):
    if not syft_path.startswith("syft://"):
        raise Exception(f"Wrong link. {syft_path}")
    end = syft_path.split("syft://")[-1]
    full = client_config.sync_folder + "/" + end
    return full


def to_syft_link(path):
    syft_path = f"syft://{str(path).split('/sync/')[-1]}"
    return syft_path


@app.get(
    "/save_query_mock",
    operation_id="save_query_mock",
    tags=["MyBigQuery"],
)
async def save_query_mock(form: DownloadSQL = Form(...)) -> str:
    user = validate_token(form.token)
    email = user["email"]
    print("user", email)
    print("settings", settings)

    path = to_path(client_config, form.syft_link)
    print("got path", path)
    os.makedirs(path, exist_ok=True)

    # Auth for Bigquer based on the workload identity

    creds = settings["gce_service_account"]
    credentials = service_account.Credentials.from_service_account_info(creds)
    scoped_credentials = credentials.with_scopes(
        ["https://www.googleapis.com/auth/cloud-platform"]
    )

    client = bigquery.Client(
        credentials=scoped_credentials,
        location=settings["gce_region"],
    )
    try:
        rows = client.query_and_wait(
            form.sql_query,
            project=settings.gce_project_id,
        )

        if rows.total_rows > 1_000_000:
            raise Exception(
                public_message="Please only write queries that gather aggregate statistics"
            )

        now = datetime.now().timestamp()
        full_path = path + "/" + f"{int(now)}.csv"
        rows.to_dataframe().to_csv(path + "/" + f"{int(now)}.csv")
        return to_syft_link(full_path)

    except Exception as e:
        output = f"got exception e: {type(e)} {str(e)}"
        raise Exception(
            public_message=f"An error occured executing the API call {output}"
        )


@app.get(
    "/download_query_mock",
    operation_id="download_query_mock",
    tags=["MyBigQuery"],
    response_class=FileResponse,
)
async def download_query_mock(form: SubmitSQL = Form(...)):
    user = validate_token(form.token)
    email = user["email"]
    print("user", email)
    print("settings", settings)

    # Auth for Bigquer based on the workload identity

    creds = settings["gce_service_account"]
    credentials = service_account.Credentials.from_service_account_info(creds)
    scoped_credentials = credentials.with_scopes(
        ["https://www.googleapis.com/auth/cloud-platform"]
    )

    client = bigquery.Client(
        credentials=scoped_credentials,
        location=settings["gce_region"],
    )
    try:
        rows = client.query_and_wait(
            form.sql_query,
            project=settings.gce_project_id,
        )

        if rows.total_rows > 1_000_000:
            raise Exception(
                public_message="Please only write queries that gather aggregate statistics"
            )

        now = int(datetime.now().timestamp())
        df = rows.to_dataframe()

        # Save the DataFrame to a temporary CSV file
        with NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            df.to_csv(tmp.name, index=False)
            tmp_path = tmp.name

        return FileResponse(
            tmp_path, media_type="application/octet-stream", filename=f"{int(now)}.csv"
        )

    except Exception as e:
        output = f"got exception e: {type(e)} {str(e)}"
        raise Exception(
            public_message=f"An error occured executing the API call {output}"
        )


@app.get("/form", response_class=HTMLResponse, include_in_schema=False)
async def get_form_page(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


# @app.get("/bigquery")
# async def bigquery(request: Request):
#     return "bigquery"


class Person(BaseModel):
    name: str
    age: int


@app.post(
    "/submit_form",
)
async def submit_form(form: Person = Form(...)):
    return {"name": form.name, "age": form.age}


main_app = FastAPI()
main_app.mount("/bigquery", app)


def main() -> None:
    debug = True
    uvicorn.run(
        "main:main_app" if debug else main_app,
        host="0.0.0.0",
        port=9081,
        log_level="debug" if debug else "info",
        reload=debug,
        reload_dirs="./",
    )


if __name__ == "__main__":
    main()
