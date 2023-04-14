# stdlib
import asyncio
import threading
from typing import Callable
from typing import Tuple

# third party
from fastapi import APIRouter
from fastapi import FastAPI
import uvicorn

# relative
from ..worker import Worker
from .routes import make_routes


def make_app(name: str, router: APIRouter) -> FastAPI:
    app = FastAPI(
        title=name,
    )

    api_router = APIRouter()
    api_router.include_router(router, prefix="/new", tags=["new"])

    app.include_router(api_router, prefix="/api/v1")
    return app


# shutdown_event = asyncio.Event()


def run_uvicorn(port: int, host: str):
    print("port and host", port, host)

    async def _run_uvicorn(port: int, host: str):
        worker = Worker.named("test", processes=1, reset=True)
        router = make_routes(worker=worker)
        app = make_app(worker.name, router=router)

        print("got app in async thread", app)
        # logging.getLogger("uvicorn").setLevel(logging.CRITICAL)
        # logging.getLogger("uvicorn.access").setLevel(logging.CRITICAL)
        config = uvicorn.Config(app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
        # await shutdown_event.wait()
        # await server.shutdown()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_run_uvicorn(port, host))
    loop.close()


def bind_worker(port: int = 8080, host: str = "0.0.0.0") -> Tuple[Callable, Callable]:
    server_thread = threading.Thread(target=run_uvicorn, args=(port, host))

    def start():
        server_thread.start()

    def stop():
        # asyncio.run(shutdown_event.set())
        server_thread.join()

    return start, stop
