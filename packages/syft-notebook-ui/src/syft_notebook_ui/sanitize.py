from copy import deepcopy

import nh3


def sanitize_html(html_str: str) -> str:
    policy = {
        "tags": ["svg", "strong", "rect", "path", "circle", "code", "pre"],
        "attributes": {
            "*": {"class", "style"},
            "svg": {
                "class",
                "style",
                "xmlns",
                "width",
                "height",
                "viewBox",
                "fill",
                "stroke",
                "stroke-width",
            },
            "path": {"d", "fill", "stroke", "stroke-width"},
            "rect": {"x", "y", "width", "height", "fill", "stroke", "stroke-width"},
            "circle": {"cx", "cy", "r", "fill", "stroke", "stroke-width"},
        },
        "remove": {"script", "style"},
    }

    tags = nh3.ALLOWED_TAGS
    for tag in policy["tags"]:
        tags.add(tag)

    _attributes = deepcopy(nh3.ALLOWED_ATTRIBUTES)
    attributes = {**_attributes, **policy["attributes"]}  # type: ignore

    return nh3.clean(
        html_str,
        tags=tags,
        clean_content_tags=policy["remove"],
        attributes=attributes,
    )
