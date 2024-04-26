# third party
import pytest

# syft absolute
from syft.service.worker.image_identifier import SyftWorkerImageIdentifier
from syft.service.worker.image_registry import SyftImageRegistry


def test_image_id_str_no_host():
    tag = "openmined/test-nginx:0.7.8"
    image_id = SyftWorkerImageIdentifier.from_str(tag)
    assert image_id.registry_host == ""


def test_image_id_str_with_host():
    tag = "localhost:5678/openmined/test-nginx:0.7.8"
    image_id = SyftWorkerImageIdentifier.from_str(tag)
    assert image_id.registry_host == "localhost:5678"
    assert image_id.tag == "0.7.8"
    assert image_id.repo == "openmined/test-nginx"
    assert image_id.repo_with_tag == "openmined/test-nginx:0.7.8"
    assert image_id.full_name_with_tag == tag


def test_image_id_with_registry():
    tag = "docker.io/openmined/test-nginx:0.7.8"
    registry = SyftImageRegistry.from_url("docker.io")
    image_id = SyftWorkerImageIdentifier.with_registry(tag, registry)

    assert image_id.registry_host == "docker.io"
    assert image_id.repo == "openmined/test-nginx"
    assert image_id.tag == "0.7.8"
    assert image_id.repo_with_tag == "openmined/test-nginx:0.7.8"
    assert image_id.full_name_with_tag == tag


def test_image_id_with_gcp_registry():
    tag = "openmined/test-image:1.0"

    gcp_url = "us-central1-docker.pkg.dev/project-12345/registry-name"
    registry = SyftImageRegistry.from_url(gcp_url)
    image_id = SyftWorkerImageIdentifier.with_registry(tag, registry)

    assert (
        image_id.registry_host
        == "us-central1-docker.pkg.dev/project-12345/registry-name"
    )
    assert image_id.repo == "openmined/test-image"
    assert image_id.tag == "1.0"
    assert image_id.repo_with_tag == "openmined/test-image:1.0"
    assert image_id.full_name_with_tag == f"{gcp_url}/{tag}"


def test_image_id_with_incorrect_registry():
    with pytest.raises(ValueError):
        tag = "docker.io/openmined/test-nginx:0.7.8"
        registry = SyftImageRegistry.from_url("localhost:5678")
        _ = SyftWorkerImageIdentifier.with_registry(tag, registry)
