# stdlib
import base64
from functools import lru_cache
import importlib.resources

# third party
import lxml.etree

IMAGE_ASSETS = "syft.assets.img"
SVG_ASSETS = "syft.assets.svg"
CSS_ASSETS = "syft.assets.css"


def _cleanup_svg(svg: str) -> str:
    # notebook_addons table template requires SVGs with single quotes and no newlines
    parser = lxml.etree.XMLParser(remove_blank_text=True)
    elem = lxml.etree.XML(svg, parser=parser)
    parsed = lxml.etree.tostring(elem, encoding="unicode")
    # NOTE UNSAFE for non-SVG xml
    parsed = parsed.replace('"', "'")
    return parsed


@lru_cache(maxsize=32)
def load_svg(fname: str) -> str:
    res = importlib.resources.read_text(SVG_ASSETS, fname)
    return _cleanup_svg(res)


@lru_cache(maxsize=32)
def load_png_base64(fname: str) -> str:
    b = importlib.resources.read_binary(IMAGE_ASSETS, fname)
    res = base64.b64encode(b)
    return f"data:image/png;base64,{res.decode('utf-8')}"


@lru_cache(maxsize=32)
def load_css(fname: str) -> str:
    return importlib.resources.read_text(CSS_ASSETS, fname)
