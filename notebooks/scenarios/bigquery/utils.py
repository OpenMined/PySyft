# third party
import docker

REGISTRY_PORT = 5678


class LocalRegistryContainer:
    def __init__(self):
        self.name = "local_registry"
        self.client = docker.from_env()

    def start(self, host_port=REGISTRY_PORT):
        existing = self.get()
        if existing:
            return existing

        result = self.client.containers.run(
            "registry:2",
            name=self.name,
            detach=True,
            ports={"5000/tcp": host_port},
            labels={"orgs.openmined.syft": "local-registry"},
        )

        return result

    def teardown(self):
        existing = self.get()
        if existing:
            existing.stop()
            existing.remove()

    def get(self):
        try:
            result = self.client.containers.get(self.name)
            if result.status == "running":
                return result
        except docker.errors.NotFound:
            return None


local_registry_container = LocalRegistryContainer()
