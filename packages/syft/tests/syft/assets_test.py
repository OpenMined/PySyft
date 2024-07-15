# third party
import pytest

# syft absolute
from syft.util.assets import load_css
from syft.util.assets import load_png_base64
from syft.util.assets import load_svg
from syft.util.notebook_ui.icons import Icon


def test_load_assets():
    png = load_png_base64("small-syft-symbol-logo.png")
    assert isinstance(png, str)

    with pytest.raises(FileNotFoundError):
        load_png_base64("non_existent.png")

    svg = load_svg("copy.svg")
    assert isinstance(svg, str)

    css = load_css("style.css")
    assert isinstance(css, str)


def test_icons():
    for icon in Icon:
        assert isinstance(icon.svg, str)
        assert isinstance(icon.js_escaped_svg, str)
