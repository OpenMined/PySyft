import json
import logging
import re
import secrets
from typing import Any, Optional

import jinja2
from IPython.display import HTML, display

# from syft_notebook_ui.components import Badge, CopyButton, Label
from syft_notebook_ui.icons import Icon
from syft_notebook_ui.resources import load_css, load_js
from syft_notebook_ui.table_utils import (
    TABLE_INDEX_KEY,
    format_table_data,
    prepare_table_data,
)

logger = logging.getLogger(__name__)


def make_links(text: str) -> str:
    file_pattern = re.compile(r"([\w/.-]+\.py)\", line (\d+)")
    return file_pattern.sub(r'<a href="file://\1:\2">\1, line \2</a>', text)


DEFAULT_ID_WIDTH = 110
jinja_env = jinja2.Environment(
    loader=jinja2.PackageLoader("syft_notebook_ui", "assets/jinja")
)  # nosec
jinja_env.filters["make_links"] = make_links


def create_tabulator_columns(
    column_names: list[str],
    column_widths: dict | None = None,
    header_sort: bool = True,
) -> tuple[list[dict], dict | None]:
    """Returns tuple of (columns, row_header) for tabulator table"""
    if column_widths is None:
        column_widths = {}

    columns = []
    row_header = {}
    if TABLE_INDEX_KEY in column_names:
        row_header = {
            "field": TABLE_INDEX_KEY,
            "frozen": True,
            "widthGrow": 0.3,
            "minWidth": 60,
            "headerSort": header_sort,
        }

    for colname in column_names:
        if colname != TABLE_INDEX_KEY:
            column = {
                "title": colname,
                "field": colname,
                "formatter": "html",
                "resizable": True,
                "minWidth": 60,
                "maxInitialWidth": 500,
                "headerSort": header_sort,
            }
            if colname in column_widths:
                column["widthGrow"] = column_widths[colname]
            columns.append(column)

    return columns, row_header


def _build_table_html(
    table_data: list[dict],
    table_metadata: dict,
    max_height: Optional[int],
    pagination: bool,
    header_sort: bool,
    uid: Optional[str],
) -> str:
    # UID is used to identify the table in the DOM
    uid = uid if uid is not None else secrets.token_hex(8)

    table_template = jinja_env.get_template("table.jinja2")
    tabulator_js = load_js("tabulator.min.js")
    tabulator_css = load_css("tabulator_pysyft.min.css")
    js = load_js("table.js")
    css = load_css("style.css")

    # Add tabulator as a named module for VSCode compatibility
    tabulator_js = tabulator_js.replace(
        "define(t)", "define('tabulator-tables', [], t)"
    )

    icon = table_metadata.get("icon", None)
    if icon is None:
        icon = Icon.TABLE.svg

    column_data, row_header = create_tabulator_columns(
        table_metadata["columns"], header_sort=header_sort
    )
    table_data = format_table_data(table_data)
    table_html = table_template.render(
        uid=uid,
        columns=json.dumps(column_data),
        row_header=json.dumps(row_header),
        data=json.dumps(table_data),
        css=css,
        js=js,
        index_field_name=TABLE_INDEX_KEY,
        icon=icon,
        name=table_metadata["name"],
        tabulator_js=tabulator_js,
        tabulator_css=tabulator_css,
        max_height=json.dumps(max_height),
        pagination=json.dumps(pagination),
        header_sort=json.dumps(header_sort),
    )

    return table_html


def build_tabulator_table(
    obj: Any,
    uid: str | None = None,
    max_height: int | None = None,
    pagination: bool = True,
    header_sort: bool = True,
) -> str | None:
    """
    Builds a Tabulator table from the given object. If no table can be built, returns None.

    If the object cannot be represented as a table, returns None.

    Args:
        obj (Any): The object to build the table from.
        uid (str, optional): The unique identifier for the table. Defaults to None.
        max_height (int, optional): The maximum height of the table. Defaults to None.
        pagination (bool, optional): Whether to enable pagination. Defaults to True.
        header_sort (bool, optional): Whether to enable header sorting. Defaults to True.

    Returns:
        str | None: The HTML representation of the Tabulator table or None

    """
    table_data, table_metadata = prepare_table_data(obj)
    if len(table_data) == 0:
        if hasattr(obj, "__len__") and len(obj) == 0:
            return obj.__repr__()
        else:
            return None

    return _build_table_html(
        table_data=table_data,
        table_metadata=table_metadata,
        max_height=max_height,
        pagination=pagination,
        header_sort=header_sort,
        uid=uid,
    )


def show_table(obj: Any) -> None:
    """Utility function to display a Tabulator table in Jupyter, without overwriting `obj._html_repr_`."""
    table = build_tabulator_table(obj)
    if table is not None:
        display(HTML(table))
