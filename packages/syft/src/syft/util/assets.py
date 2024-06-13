# stdlib
import base64
from functools import lru_cache
from importlib.resources import files

IMAGE_ASSETS = "syft.assets.img"
SVG_ASSETS = "syft.assets.svg"
CSS_ASSETS = "syft.assets.css"
JS_ASSETS = "syft.assets.js"


@lru_cache(maxsize=32)
def load_svg(fname: str) -> str:
    # TODO add resize support
    return files(SVG_ASSETS).joinpath(fname).read_text()


@lru_cache(maxsize=32)
def load_png_base64(fname: str) -> str:
    b = files(IMAGE_ASSETS).joinpath(fname).read_bytes()
    res = base64.b64encode(b)
    return f"data:image/png;base64,{res.decode('utf-8')}"


@lru_cache(maxsize=32)
def load_css(fname: str) -> str:
    return files(CSS_ASSETS).joinpath(fname).read_text()


@lru_cache(maxsize=32)
def load_js(fname: str) -> str:
    return files(JS_ASSETS).joinpath(fname).read_text()
