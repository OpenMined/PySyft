# stdlib
import base64
from functools import lru_cache
import importlib.resources

IMAGE_ASSETS = "syft.assets.img"
SVG_ASSETS = "syft.assets.svg"
CSS_ASSETS = "syft.assets.css"
JS_ASSETS = "syft.assets.js"


@lru_cache(maxsize=32)
def load_svg(fname: str) -> str:
    # TODO add resize support
    return importlib.resources.read_text(SVG_ASSETS, fname)


@lru_cache(maxsize=32)
def load_png_base64(fname: str) -> str:
    b = importlib.resources.read_binary(IMAGE_ASSETS, fname)
    res = base64.b64encode(b)
    return f"data:image/png;base64,{res.decode('utf-8')}"


@lru_cache(maxsize=32)
def load_css(fname: str) -> str:
    return importlib.resources.read_text(CSS_ASSETS, fname)


@lru_cache(maxsize=32)
def load_js(fname: str) -> str:
    return importlib.resources.read_text(JS_ASSETS, fname)
