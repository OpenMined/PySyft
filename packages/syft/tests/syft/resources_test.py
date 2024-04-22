# third party
import pytest

# syft absolute
from syft.util.resources import load_css
from syft.util.resources import load_png_base64
from syft.util.resources import load_svg


def test_load_png_base64():
    png = load_png_base64("logo.png")
    assert isinstance(png, str)

    with pytest.raises(FileNotFoundError):
        load_png_base64("non_existent.png")


def test_load_svg():
    svg = load_svg("copy.svg")
    assert isinstance(svg, str)

    assert svg.startswith("<svg")
    assert svg.endswith("</svg>")
    # Required for notebook_addons table
    assert "\n" not in svg
    assert '"' not in svg


def test_load_css():
    css = load_css("style.css")
    assert isinstance(css, str)
