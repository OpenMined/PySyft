# stdlib
import os
import platform
import subprocess

# third party
import pytest

CONTAINER_HOST = os.environ.get("CONTAINER_HOST", "docker")
OS = platform.system().lower()
print("CONTAINER_HOST", CONTAINER_HOST)


def make_curl_cmd(url: str, key: str) -> str:
    add_curl = "apk -U upgrade && apk fix && apk add curl"
    run_curl = f"curl -v -X POST -H 'X-STACK-API-KEY: {key}' {url}"
    if CONTAINER_HOST == "docker":
        container_cmd = f"{add_curl} && {run_curl}"
        container = "test_network_1-tailscale-1"
        return f'echo "{container_cmd}" | docker exec -i {container} bash 2>&1'
    else:
        pod = "proxy-0"
        container = "container-1"
        context = "k3d-test-network-1"
        namespace = "test-network-1"
        kubectl_run = (
            f"kubectl exec -it {pod} -c {container}  --context {context} "
            + f"--namespace {namespace} -- "
        )
        return (
            f'{kubectl_run} bash -c "{add_curl}" && {kubectl_run} bash -c "{run_curl}"'
        )


def make_get_key_cmd() -> str:
    get_env = "env | grep STACK_API_KEY | cut -d'=' -f2"
    if CONTAINER_HOST == "docker":
        container = "test_network_1-backend-1"
        return f"docker exec {container} bash -c {get_env}"
    else:
        service_name = "service/backend"
        context = "k3d-test-network-1"
        namespace = "test-network-1"
        kubectl_run = (
            f"kubectl exec -it {service_name} --context {context} "
            + f"--namespace {namespace} -- "
        )
        return f"{kubectl_run} {get_env}"


def get_stack_key() -> str:
    try:
        cmd = make_get_key_cmd()
        output = subprocess.check_output(cmd, shell=True)
        return output.decode("utf-8")
    except Exception as e:
        print(f"Exception running: {cmd}. {e}")
        raise e


@pytest.mark.security
def test_api_auth() -> None:
    # TODO: make work on Windows
    if OS == "windows":
        return
    urls = [
        "http://headscale:4000/commands/generate_key",
        "http://proxy:4000/commands/status",
    ]

    old_stack_key = "hex_key_value"

    # try with a bad header
    try:
        for url in urls:
            cmd = make_curl_cmd(url=url, key=old_stack_key)
            output = subprocess.check_output(cmd, shell=True)
            output = output.decode("utf-8")
            assert "500 INTERNAL SERVER ERROR".lower() in output.lower()
    except Exception as e:
        print(f"Exception running: {cmd}. {e}")
        raise e

    # try with correct header
    try:
        for url in urls:
            stack_key = get_stack_key()
            cmd = make_curl_cmd(url=url, key=stack_key)
            output = subprocess.check_output(cmd, shell=True)
            output = output.decode("utf-8")
            assert "result_url" in output
    except Exception as e:
        print(f"Exception running: {cmd}. {e}")
        raise e
