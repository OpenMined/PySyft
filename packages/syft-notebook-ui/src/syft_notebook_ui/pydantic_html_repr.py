import enum
import html
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import jinja2
from pydantic import BaseModel

from syft_notebook_ui.resources import load_css
from syft_notebook_ui.utils import make_dirtree_string

jinja_env = jinja2.Environment(
    loader=jinja2.PackageLoader("syft_notebook_ui", "assets/jinja")
)  # nosec
PERM_FILE = "syftperm.yaml"


def format_field_value(value: Any) -> str:
    """Format various field types for display with multiline text for paths"""
    if value is None:
        return "None"
    elif isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    elif isinstance(value, Path):
        path_str = str(value)
        return f'<div class="path-text">{html.escape(path_str)}</div>'
    elif isinstance(value, enum.Enum):
        return str(value.name)
    elif hasattr(value, "uid"):
        return f"{type(value).__name__}({value.uid})"
    else:
        return f'<div class="path-text">{html.escape(str(value))}</div>'


def prepare_path_display(obj: Any, path_field: str, open_default: bool = False) -> str:
    """
    Generate HTML for displaying a path (either file content or directory tree)

    Args:
        obj: The object containing the path field
        path_field: The name of the field containing the path
        open_default: Whether the details element should be open by default

    Returns:
        str: HTML representation of the file content or directory tree
    """
    template = jinja_env.get_template("model_repr_file_section.jinja2")
    render_params = {"title": path_field, "open": open_default}

    # Check if field exists
    if not hasattr(obj, path_field):
        render_params["error"] = f"Field '{path_field}' does not exist on this object."
        render_params["error_class"] = "error-text"
        return template.render(**render_params)

    # Get path
    path = getattr(obj, path_field)
    if not isinstance(path, Path) or not path.exists():
        render_params["error"] = (
            "The path does not exist or is not a valid Path object."
        )
        render_params["error_class"] = "warning-text"
        return template.render(**render_params)

    try:
        if path.is_file():
            # Handle file display
            with open(path, "r") as f:
                file_content = f.read()

            render_params["title"] = f"{path_field} content ({path.name})"
            render_params["content"] = html.escape(file_content)
        elif path.is_dir():
            # Handle directory tree display
            tree_content = make_dirtree_string(path)
            render_params["title"] = f"{path_field} directory structure ({path.name})"
            render_params["content"] = html.escape(tree_content) if tree_content else ""
        else:
            render_params["error"] = "Path is neither a file nor a directory."
            render_params["error_class"] = "warning-text"
    except Exception as e:
        render_params["error"] = str(e)
        render_params["error_class"] = "error-text"

    return template.render(**render_params)


def create_html_repr(
    obj: BaseModel,
    fields: List[str],
    name_field: str = "name",
    display_paths: Optional[List[str]] = None,
    paths_open: bool = False,
) -> str:
    """
    Generate an HTML representation of a model with automatic path type detection

    Args:
        obj: A Pydantic BaseModel
        fields: List of fields to display
        display_paths: List of field names containing paths that should be displayed
            (automatically detects if it's a file or directory)
        paths_open: Whether path sections should be open by default

    Returns:
        str: HTML representation
    """
    display_paths = display_paths or []

    # Prepare field values
    field_values = {}
    for field in fields:
        if hasattr(obj, field):
            field_values[field] = format_field_value(getattr(obj, field))
        else:
            field_values[field] = '<span class="error-text">Field not found</span>'

    # Get object name
    obj_name = getattr(obj, name_field, None)

    # Prepare path divs (file content and directory trees)
    path_html_reprs = [
        prepare_path_display(obj, path_field, paths_open)
        for path_field in display_paths
    ]

    template = jinja_env.get_template("model_repr.jinja2")
    return template.render(
        obj=obj,
        obj_name=obj_name,
        display_fields=fields,
        field_values=field_values,
        path_displays=path_html_reprs,
        css=load_css("model_repr.css"),
    )
