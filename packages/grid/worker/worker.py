# stdlib

# third party
from fastapi import FastAPI
from new_routes import make_routes

# syft absolute
from syft.core.node.new.sqlite_document_store import SQLiteStoreClientConfig
from syft.core.node.new.sqlite_document_store import SQLiteStoreConfig
from syft.core.node.worker import Worker
from syft.core.node.worker import create_admin_new

API_PATH = "/api/v1/new"


client_config = SQLiteStoreClientConfig()
sql_store_config = SQLiteStoreConfig(client_config=client_config)
worker = Worker(
    action_store_config=sql_store_config, document_store_config=sql_store_config
)

router = make_routes(worker=worker)

create_admin_new(
    name="Jane Doe", email="info@openmined.org", password="changethis", node=worker
)
print("worker", worker.id, worker.signing_key)


app = FastAPI(title="Worker")
app.include_router(router, prefix=API_PATH)
