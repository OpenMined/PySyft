# stdlib
import os

# third party
import uvicorn

PORT = int(os.environ.get("PORT", "8081"))
HOST = os.environ.get("HOST", "0.0.0.0")

if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT)
