# stdlib
import importlib.util
import inspect
import subprocess
import sys

# third party
import anyio


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
