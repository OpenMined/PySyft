# stdlib
import subprocess

# third party
import pytest


def make_curl_cmd(url: str, key: str, container: str) -> str:
    return (
        f"echo \"apk add curl && curl -v -X POST -H 'X-STACK-API-KEY: {key}' {url}\" | "
        f"docker exec -i {container} ash 2>&1"
    )


@pytest.mark.security
def test_api_auth() -> None:
    container = "test_network_1_tailscale_1"
    urls = [
        "http://headscale:4000/commands/generate_key",
        "http://tailscale:4000/commands/status",
    ]

    stack_key = "hex_key_value"

    # try with a bad header
    try:
        for url in urls:
            cmd = make_curl_cmd(url=url, key="garbage", container=container)
            output = subprocess.check_output(cmd, shell=True)
            output = output.decode("utf-8")
            assert "500 INTERNAL SERVER ERROR" in output
    except Exception as e:
        print(f"Exception running: {cmd}. {e}")
        raise e

    # try with correct header
    try:
        for url in urls:
            cmd = make_curl_cmd(url=url, key=stack_key, container=container)
            output = subprocess.check_output(cmd, shell=True)
            output = output.decode("utf-8")
            assert "result_url" in output
    except Exception as e:
        print(f"Exception running: {cmd}. {e}")
        raise e
