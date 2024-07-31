SURFACE_DARK_BRIGHT = "#464158"
SURFACE_SURFACE_DARK = "#2E2B3B"
DK_ON_SURFACE_HIGHEST = "#534F64"

ON_SURFACE_HIGHEST = {"light": "#534F64", "dark": "#ffffff"}

SURFACE_SURFACE = {"light": "#2E2B3B", "dark": "#ffffff"}

SURFACE = {"light": "#464158", "dark": "#ffffff"}


def light_dark_css(cmap: dict[str, str]) -> str:
    return f"light-dark({cmap['light'], cmap['dark']})"
