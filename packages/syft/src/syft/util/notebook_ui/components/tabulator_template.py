# stdlib
import json
import logging
import re
import secrets
from typing import Any

# third party
from IPython.display import HTML
from IPython.display import display
import jinja2

# relative
from ....types.uid import UID
from ...assets import load_css
from ...assets import load_js
from ...table import TABLE_INDEX_KEY
from ...table import prepare_table_data
from ...util import sanitize_html
from ..icons import Icon

logger = logging.getLogger(__name__)


def make_links(text: str) -> str:
    file_pattern = re.compile(r"([\w/.-]+\.py)\", line (\d+)")
    return file_pattern.sub(r'<a href="file://\1:\2">\1, line \2</a>', text)


DEFAULT_ID_WIDTH = 110
jinja_env = jinja2.Environment(loader=jinja2.PackageLoader("syft", "assets/jinja"))  # nosec
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


def format_dict(data: Any) -> str:
    # relative
    from .sync import Badge
    from .sync import CopyButton
    from .sync import Label

    if not isinstance(data, dict):
        return data

    if set(data.keys()) != {"type", "value"}:
        return sanitize_html(str(data))
    if "badge" in data["type"]:
        return Badge(value=data["value"], badge_class=data["type"]).to_html()
    elif "label" in data["type"]:
        return Label(value=data["value"], label_class=data["type"]).to_html()
    if "clipboard" in data["type"]:
        return CopyButton(copy_text=data["value"]).to_html()

    return sanitize_html(str(data))


def format_uid(uid: UID) -> str:
    # relative
    from .sync import CopyButton

    return CopyButton(copy_text=uid.no_dash).to_html()


def format_table_data(table_data: list[dict[str, Any]]) -> list[dict[str, str]]:
    formatted: list[dict[str, str]] = []
    for row in table_data:
        row_formatted: dict[str, str] = {}
        for k, v in row.items():
            if isinstance(v, str):
                row_formatted[k] = sanitize_html(v.replace("\n", "<br>"))
                continue
            # make UID copyable and trimmed
            if isinstance(v, UID):
                v_formatted = format_uid(v)
            else:
                v_formatted = format_dict(v)
            row_formatted[k] = v_formatted
        formatted.append(row_formatted)
    return formatted


def _render_tabulator_table(
    uid: str,
    table_data: list[dict],
    table_metadata: dict,
    max_height: int | None,
    pagination: bool,
    header_sort: bool,
) -> str:
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


def build_tabulator_table_with_data(
    table_data: list[dict],
    table_metadata: dict,
    uid: str | None = None,
    max_height: int | None = None,
    pagination: bool = True,
    header_sort: bool = True,
) -> str:
    """
    Builds a Tabulator table for the provided data and metadata.

    Args:
        table_data (list[dict]): The data to populate the table.
        table_metadata (dict): The metadata for the table.
        uid (str, optional): The unique identifier for the table. Defaults to None.
        max_height (int, optional): The maximum height of the table. Defaults to None.
        pagination (bool, optional): Whether to enable pagination. Defaults to True.
        header_sort (bool, optional): Whether to enable header sorting. Defaults to True.

    Returns:
        str: The HTML representation of the Tabulator table.

    """
    uid = uid if uid is not None else secrets.token_hex(4)
    return _render_tabulator_table(
        uid, table_data, table_metadata, max_height, pagination, header_sort
    )


def build_tabulator_table(
    obj: Any,
    uid: str | None = None,
    max_height: int | None = None,
    pagination: bool = True,
    header_sort: bool = True,
) -> str | None:
    """
    Builds a Tabulator table from the given object if possible.

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

    return build_tabulator_table_with_data(
        table_data, table_metadata, uid, max_height, pagination, header_sort
    )


def show_table(obj: Any) -> None:
    table = build_tabulator_table(obj)
    if table is not None:
        display(HTML(table))


def highlight_single_row(
    table_uid: str,
    index: int | str | None = None,
    jump_to_row: bool = True,
) -> None:
    js_code = f"<script>highlightSingleRow('{table_uid}', {json.dumps(index)}, {json.dumps(jump_to_row)});</script>"
    display(HTML(js_code))


def update_table_cell(uid: str, index: int, field: str, value: str) -> None:
    js_code = f"""
    <script>
    updateTableCell('{uid}', {json.dumps(index)}, {json.dumps(field)}, {json.dumps(value)});
    </script>
    """
    display(HTML(js_code))
