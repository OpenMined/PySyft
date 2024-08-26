# stdlib
import inspect

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
