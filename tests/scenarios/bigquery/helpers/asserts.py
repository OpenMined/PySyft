# stdlib
import importlib.util
import inspect
import subprocess
import sys

# third party
import anyio

# syft absolute
import syft as sy


class FailedAssert(Exception):
    pass


async def has(expr, expects="", timeout=10, retry=1):
    try:
        with anyio.fail_after(timeout):
            result = expr()
            while not result:
                print(f"> {expects} {expr}...not yet satisfied")
                await anyio.sleep(retry)
    except TimeoutError:
        lambda_source = inspect.getsource(expr)
        raise FailedAssert(f"{lambda_source} {expects}")


def check_import_exists(module_name: str):
    # can pass . paths like google.cloud.bigquery
    spec = importlib.util.find_spec(module_name)
    return spec is not None


def install_package(package_name: str):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
    except subprocess.CalledProcessError:
        print(f"pip failed to install {package_name}. Trying uv pip...")
        try:
            subprocess.check_call(["uv", "pip", "install", package_name])
        except subprocess.CalledProcessError as e:
            print(
                f"An error occurred while trying to install {package_name} with uv pip: {e}"
            )


def ensure_package_installed(package_name, module_name):
    if not check_import_exists(module_name):
        print(f"{module_name} not found. Installing...")
        install_package(package_name)
    else:
        print(f"{module_name} is already installed.")


async def result_is(
    events,
    expr,
    matches: bool | str | type | object,
    after: str | None = None,
    register: str | None = None,
):
    if after:
        await events.await_for(event_name=after)

    lambda_source = inspect.getsource(expr)
    try:
        result = None
        try:
            result = expr()
        except Exception as e:
            if isinstance(e, sy.SyftException):
                result = e
            else:
                raise e

        assertion = False
        if isinstance(matches, bool):
            assertion = result == matches
        elif isinstance(matches, type):
            assertion = isinstance(result, matches)
        elif isinstance(matches, str):
            message = matches.replace("*", "")
            assertion = message in str(result)
        else:
            type_matches = isinstance(result, type(matches))
            message_matches = True

            message = None
            if isinstance(matches, sy.service.response.SyftResponseMessage):
                message = matches.message.replace("*", "")
            elif isinstance(result, sy.SyftException):
                message = matches.public_message.replace("*", "")

            if message:
                if isinstance(result, sy.service.response.SyftResponseMessage):
                    message_matches = message in str(result)
                elif isinstance(result, sy.SyftException):
                    message_matches = message in result.public_message

            assertion = type_matches and message_matches
        if assertion and register:
            events.register(event_name=register)
        return assertion
    except Exception as e:
        print(f"insinstance({lambda_source}, {matches}). {e}")

    return False
