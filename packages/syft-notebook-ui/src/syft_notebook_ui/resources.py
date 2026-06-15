import importlib.resources
from functools import lru_cache

ASSETS = "syft_notebook_ui.assets"
CSS_ASSETS = f"{ASSETS}.css"
JS_ASSETS = f"{ASSETS}.js"
SVG_ASSETS = f"{ASSETS}.svg"


def load_css(fname: str) -> str:
    return load_resource(fname, CSS_ASSETS)


def load_js(fname: str) -> str:
    return load_resource(fname, JS_ASSETS)


def load_svg(fname: str) -> str:
    return load_resource(fname, SVG_ASSETS)


@lru_cache(maxsize=64)
def load_resource(fname: str, module: str = ASSETS) -> str:
    return importlib.resources.read_text(module, fname)
