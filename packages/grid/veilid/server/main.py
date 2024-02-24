# third party
from fastapi import FastAPI

app = FastAPI(title="Veilid")


@app.get("/")
async def read_root() -> dict[str, str]:
    return {"message": "Hello World"}
