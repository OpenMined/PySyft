# stdlib
import os
import subprocess

# third party
import pytest

CONTAINER_HOST = os.environ.get("CONTAINER_HOST", "docker")
print("CONTAINER_HOST", CONTAINER_HOST)


def make_curl_cmd(url: str, key: str) -> str:
    add_curl = "apk add curl"
    run_curl = f"curl -v -X POST -H 'X-STACK-API-KEY: {key}' {url}"
    if CONTAINER_HOST == "docker":
        container_cmd = f"{add_curl} && {run_curl}"
        container = "test_network_1-tailscale-1"
        return f'echo "{container_cmd}" | docker exec -i {container} ash 2>&1'
    else:
        pod = "tailscale-0"
        container = "container-1"
        context = "k3d-test-network-1"
        namespace = "test-network-1"
        kubectl_run = (
            f"kubectl exec -it {pod} -c {container}  --context {context} "
            + f"--namespace {namespace} -- "
        )
        return f"{kubectl_run} {add_curl} &&" f"{kubectl_run} {run_curl}"


@pytest.mark.security
def test_api_auth() -> None:
    urls = [
        "http://headscale:4000/commands/generate_key",
        "http://tailscale:4000/commands/status",
    ]

    stack_key = "hex_key_value"

    # try with a bad header
    try:
        for url in urls:
            cmd = make_curl_cmd(url=url, key="garbage")
            output = subprocess.check_output(cmd, shell=True)
            output = output.decode("utf-8")
            assert "500 INTERNAL SERVER ERROR".lower() in output.lower()
    except Exception as e:
        print(f"Exception running: {cmd}. {e}")
        raise e

    # try with correct header
    try:
        for url in urls:
            cmd = make_curl_cmd(url=url, key=stack_key)
            output = subprocess.check_output(cmd, shell=True)
            output = output.decode("utf-8")
            assert "result_url" in output
    except Exception as e:
        print(f"Exception running: {cmd}. {e}")
        raise e
