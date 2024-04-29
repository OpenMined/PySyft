# third party
import pytest

# syft absolute
from syft.util.assets import load_css
from syft.util.assets import load_png_base64
from syft.util.assets import load_svg


def test_load_png_base64():
    png = load_png_base64("logo.png")
    assert isinstance(png, str)

    with pytest.raises(FileNotFoundError):
        load_png_base64("non_existent.png")


def test_load_svg():
    svg = load_svg("copy.svg").strip()
    assert isinstance(svg, str)

    assert svg.startswith("<svg")
    assert svg.endswith("</svg>")


def test_load_css():
    css = load_css("style.css")
    assert isinstance(css, str)
