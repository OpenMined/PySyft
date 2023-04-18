# stdlib

# third party

# syft absolute
from syft.client.api import APIRegistry
from syft.client.api import SyftAPI
from syft.node.credentials import SyftSigningKey
from syft.node.worker import Worker
from syft.service.context import AuthedServiceContext


def setup_worker():
    test_signing_key = SyftSigningKey.generate()
    credentials = test_signing_key.verify_key
    worker = Worker(name="Test Worker", signing_key=test_signing_key.signing_key)
    context = AuthedServiceContext(node=worker, credentials=credentials)

    api = SyftAPI.for_user(node=worker)

    APIRegistry.set_api_for(node_uid=worker.id, api=api)

    return worker, context
