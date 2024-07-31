SURFACE_DARK_BRIGHT = "#464158"
SURFACE_SURFACE_DARK = "#2E2B3B"
DK_ON_SURFACE_HIGHEST = "#534F64"

ON_SURFACE_HIGHEST = {"light": "#534F64", "dark": "#ffffff"}

SURFACE_SURFACE = {"light": "#2E2B3B", "dark": "#ffffff"}

SURFACE = {"light": "#464158", "dark": "#ffffff"}


def light_dark_css(cmap: dict[str, str]) -> str:
    """
    Generate a CSS light-dark function call for the light-dark color scheme.

    Args:
        cmap (dict[str, str]): A dictionary containing the light and dark colors.
            The keys should be "light" and "dark", and the values should be valid CSS color values.

    Returns:
        str: A CSS function call as a string representing the light-dark color scheme.
            The format is "light-dark(light_color, dark_color)".

    """
    return f"light-dark({cmap['light'], cmap['dark']})"
