# stdlib
import base64
from functools import lru_cache
import importlib.resources as pkg_resources

IMAGE_RESOURCES = "syft.static.img"
CSS_RESOURCES = "syft.static.css"
JS_RESOURCES = "syft.static.js"
SVG_RESOURCES = "syft.static.img.svg"


@lru_cache(maxsize=32)
def read_resource(package: str, resource: str, mode: str = "r") -> str | bytes:
    if mode == "r":
        return pkg_resources.read_text(package, resource)
    elif mode == "rb":
        return pkg_resources.read_binary(package, resource)
    else:
        raise ValueError("Invalid mode")


def read_png_base64(fname: str) -> str:
    b = read_resource(IMAGE_RESOURCES, fname, mode="rb")
    res = base64.b64encode(b)
    return f"data:image/png;base64,{res.decode('utf-8')}"


def read_css(fname: str) -> str:
    return read_resource(CSS_RESOURCES, fname, mode="r")  # type: ignore


def read_js(fname: str) -> str:
    return read_resource(JS_RESOURCES, fname, mode="r")  # type: ignore


def read_svg(fname: str) -> str:
    return read_resource(SVG_RESOURCES, fname, mode="r")  # type: ignore
