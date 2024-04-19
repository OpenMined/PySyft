# third party
import pytest

# syft absolute
from syft.util.resources import read_css
from syft.util.resources import read_js
from syft.util.resources import read_png_base64
from syft.util.resources import read_svg


def test_read_png_base64():
    png = read_png_base64("logo.png")
    assert isinstance(png, str)

    with pytest.raises(FileNotFoundError):
        read_png_base64("non_existent.png")


def test_read_svg():
    svg = read_svg("copy.svg")
    assert isinstance(svg, str)
    assert svg.startswith("<svg")
    assert svg.endswith("</svg>")


def test_read_css():
    css = read_css("style.css")
    assert isinstance(css, str)


def test_read_js():
    js = read_js("copy.js")
    assert isinstance(js, str)
