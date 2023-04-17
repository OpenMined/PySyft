# stdlib

# third party

# syft absolute
from syft.core.node.new.api import APIRegistry
from syft.core.node.new.api import SyftAPI
from syft.core.node.new.context import AuthedServiceContext
from syft.core.node.new.credentials import SyftSigningKey
from syft.core.node.worker import Worker


def setup_worker():
    test_signing_key = SyftSigningKey.generate()
    credentials = test_signing_key.verify_key
    worker = Worker(name="Test Worker", signing_key=test_signing_key.signing_key)
    context = AuthedServiceContext(node=worker, credentials=credentials)

    api = SyftAPI.for_user(node=worker)

    APIRegistry.set_api_for(node_uid=worker.id, api=api)

    return worker, context
