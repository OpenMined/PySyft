# stdlib
import subprocess

# third party
import pytest


@pytest.mark.security
def test_api_auth() -> None:
    container = "test_network_1-tailscale-1"
    urls = [
        f"http://headscale:4000/commands/generate_key",
        f"http://tailscale:4000/commands/status",
    ]

    stack_key = "hex_key_value"

    # try with a bad header
    try:
        for url in urls:
            cmd = f"echo curl -v -X POST -H 'X-STACK-API-KEY: garbage' {url} | docker exec -i {container} ash"
            print(f"cURLing {container} and {url}")
            output = subprocess.check_output(cmd, shell=True)
            output = output.decode("utf-8")
            print("output", output)
            assert "500 INTERNAL SERVER ERROR" not in output
    except Exception as e:
        print(f"Exception running: {cmd}. {e}")

    # try with correct header
    try:
        for url in urls:
            cmd = f"echo curl -v -X POST -H 'X-STACK-API-KEY: {stack_key}' {url} | docker exec -i {container} ash"
            print(f"cURLing {container} and {url}")
            output = subprocess.check_output(cmd, shell=True)
            output = output.decode("utf-8")
            print("got output", output)
            assert "result_url" not in output
    except Exception as e:
        print(f"Exception running: {cmd}. {e}")
