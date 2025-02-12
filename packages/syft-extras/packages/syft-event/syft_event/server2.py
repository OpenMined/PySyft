from __future__ import annotations

import json
from pathlib import Path
from threading import Event

from loguru import logger
from pydantic import BaseModel
from syft_core import Client
from syft_rpc import rpc
from syft_rpc.protocol import SyftRequest, SyftStatus
from typing_extensions import Callable, List, Optional, Type, Union
from watchdog.events import FileCreatedEvent, FileModifiedEvent, FileSystemEvent
from watchdog.observers import Observer

from syft_event.deps import func_args_from_request
from syft_event.handlers import AnyPatternHandler, RpcRequestHandler
from syft_event.schema import generate_schema
from syft_event.types import Response

DEFAULT_WATCH_EVENTS: List[Type[FileSystemEvent]] = [
    FileCreatedEvent,
    FileModifiedEvent,
]
PERMS = """
- path: 'syftperm.yaml'
  user: '*'
  permissions:
    - read
- path: 'rpc.schema.json'
  user: '*'
  permissions:
    - read
- path: '**/*.request'
  user: '*'
  permissions:
    - admin
- path: '**/*.response'
  user: '*'
  permissions:
    - admin
"""


class SyftEvents:
    def __init__(
        self,
        app_name: str,
        publish_schema: bool = True,
        client: Optional[Client] = None,
    ):
        self.app_name = app_name
        self.schema = publish_schema
        self.client = client or Client.load()
        self.app_dir = self.client.api_data(self.app_name)
        self.app_rpc_dir = self.app_dir / "rpc"
        self.obs = Observer()
        self.__rpc: dict[Path, Callable] = {}
        self._stop_event = Event()

    def start(self) -> None:
        # setup dirs
        self.app_dir.mkdir(exist_ok=True, parents=True)
        self.app_rpc_dir.mkdir(exist_ok=True, parents=True)

        # write perms
        perms = self.app_rpc_dir / "syftperm.yaml"
        perms.write_text(PERMS)

        # publish schema
        if self.schema:
            self.publish_schema()

        # process pending requests
        try:
            self.process_pending_requests()
        except Exception as e:
            print("Error processing pending requests", e)
            raise

        # start Observer
        self.obs.start()

    def publish_schema(self) -> None:
        schema = {}
        for endpoint, handler in self.__rpc.items():
            handler_schema = generate_schema(handler)
            ep_name = endpoint.relative_to(self.app_rpc_dir)
            ep_name = "/" + str(ep_name).replace("\\", "/")
            schema[ep_name] = handler_schema

        schema_path = self.app_rpc_dir / "rpc.schema.json"
        schema_path.write_text(json.dumps(schema, indent=2))
        logger.info(f"Published schema to {schema_path}")

    def process_pending_requests(self) -> None:
        # process all pending requests
        for path in self.app_rpc_dir.glob("**/*.request"):
            if path.with_suffix(".response").exists():
                continue
            if path.parent in self.__rpc:
                handler = self.__rpc[path.parent]
                logger.debug(f"Processing pending request {path.name}")
                self.__handle_rpc(path, handler)

    def run_forever(self) -> None:
        logger.info(f"Started watching for files. RPC Directory = {self.app_rpc_dir}")
        self.start()
        try:
            while not self._stop_event.is_set():
                self._stop_event.wait(timeout=5)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.error(f"Error in event loop: {e}")
            raise
        finally:
            self.stop()

    def stop(self) -> None:
        logger.debug("Stopping event loop")
        self._stop_event.set()
        self.obs.stop()
        self.obs.join()

    def on_request(self, endpoint: str) -> Callable:
        """Bind function to RPC requests at an endpoint"""

        def register_rpc(func):
            epath = self.__to_endpoint_path(endpoint)
            self.__register_rpc(epath, func)
            logger.info(f"Register RPC: {endpoint}")
            return func

        return register_rpc

    def watch(
        self,
        glob_path: Union[str, List[str]],
        event_filter: List[Type[FileSystemEvent]] = DEFAULT_WATCH_EVENTS,
    ):
        """Invoke the handler if any file changes in the glob path"""

        if not isinstance(glob_path, list):
            glob_path = [glob_path]

        globs = [self.__format_glob(path) for path in glob_path]

        def register_watch(func):
            def watch_cb(event):
                return func(event)

            self.obs.schedule(
                # use raw path for glob which will be convert to path/*.request
                AnyPatternHandler(globs, watch_cb),
                path=str(self.client.datasites),
                recursive=True,
                event_filter=event_filter,
            )
            logger.info(f"Register Watch: {globs}")
            return watch_cb

        return register_watch

    def __handle_rpc(self, path: Path, func: Callable):
        try:
            # may happen =)
            if not path.exists():
                return
            try:
                req = SyftRequest.load(path)
            except Exception as e:
                logger.error(f"Error loading request {path}", e)
                rpc.write_response(
                    path,
                    body=f"Error loading request: {repr(e)}",
                    status_code=SyftStatus.SYFT_400_BAD_REQUEST,
                    client=self.client,
                )
                return

            if req.is_expired:
                logger.debug(f"Request expired: {req}")
                rpc.reply_to(
                    req,
                    body="Request expired",
                    status_code=SyftStatus.SYFT_419_EXPIRED,
                    client=self.client,
                )
                return

            try:
                kwargs = func_args_from_request(func, req)
            except Exception as e:
                logger.warning(f"Invalid request body schema {req.url}: {e}")
                rpc.reply_to(
                    req,
                    body=f"Invalid request schema: {str(e)}",
                    status_code=SyftStatus.SYFT_400_BAD_REQUEST,
                    client=self.client,
                )
                return

            # call the function
            resp = func(**kwargs)

            if isinstance(resp, Response):
                resp_data = resp.body
                resp_code = SyftStatus(resp.status_code)
                resp_headers = resp.headers
            else:
                resp_data = resp
                resp_code = SyftStatus.SYFT_200_OK
                resp_headers = {}

            rpc.reply_to(
                req,
                body=resp_data,
                headers=resp_headers,
                status_code=resp_code,
                client=self.client,
            )
        except Exception as e:
            logger.error(f"Error handling request {path}: {e}")
            raise

    def __register_rpc(self, endpoint: Path, handler: Callable) -> Callable:
        def rpc_callback(event: FileSystemEvent):
            return self.__handle_rpc(Path(event.src_path), handler)

        self.obs.schedule(
            RpcRequestHandler(rpc_callback),
            path=str(endpoint),
            recursive=True,
            event_filter=[FileCreatedEvent],
        )
        # this is used for processing pending requests + generating schema
        self.__rpc[endpoint] = handler
        return rpc_callback

    def __to_endpoint_path(self, endpoint: str) -> Path:
        if "*" in endpoint or "?" in endpoint:
            raise ValueError("wildcards are not allowed in path")

        # this path must exist so that watch can emit events
        epath = self.app_rpc_dir / endpoint.lstrip("/").rstrip("/")
        epath.mkdir(exist_ok=True, parents=True)
        return epath

    def __format_glob(self, path: str) -> str:
        # replace placeholders with actual values
        path = path.format(
            email=self.client.email,
            datasite=self.client.email,
            api_data=self.client.api_data(self.app_name),
        )
        if not path.startswith("**/"):
            path = f"**/{path}"
        return path


if __name__ == "__main__":
    box = SyftEvents("test_app")

    # requests are always bound to the app
    # root path = {datasite}/api_data/{app_name}/rpc
    @box.on_request("/endpoint")
    def endpoint_request(req):
        print("rpc /endpoint:", req)

    # requests are always bound to the app
    # root path = {datasite}/api_data/{app_name}/rpc
    @box.on_request("/another")
    def another_request(req):
        print("rpc /another: ", req)

    # root path = ~/SyftBox/datasites/
    @box.watch("{datasite}/**/*.json")
    def all_json_on_my_datasite(event):
        print("watch {datasite}/**/*.json:".format(datasite=box.client.email), event)

    # root path = ~/SyftBox/datasites/
    @box.watch("test@openined.org/*.json")
    def jsons_in_some_datasite(event):
        print("watch test@openined.org/*.json:", event)

    # root path = ~/SyftBox/datasites/
    @box.watch("**/*.json")
    def all_jsons_everywhere(event):
        print("watch **/*.json:", event)

    print("Running rpc server for", box.app_rpc_dir)
    box.publish_schema()
    box.run_forever()


# if __name__ == "__main__":
#     box = SyftEvents("vector_store")

#     # requests are always bound to the app
#     # root path = {datasite}/api_data/{app_name}/rpc
#     @box.on_request("/doc_query")
#     def query(query: str) -> list[str]:
#         """Return similar documents for a given query"""
#         return []

#     @box.on_request("/doc_similarity")
#     def query_embedding(embedding: np.array) -> np.array:
#         """Return similar documents for a given embedding"""
#         return []

#     print("Running rpc server for", box.app_rpc_dir)
#     box.publish_schema()
#     box.run_forever()
