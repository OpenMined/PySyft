# stdlib
import json
import secrets
from typing import Any

# third party
from IPython.display import HTML
from IPython.display import display
import jinja2
from loguru import logger

# relative
from ...assets import load_css
from ...assets import load_js
from ...table import TABLE_INDEX_KEY
from ...table import prepare_table_data
from ..icons import Icon

DEFAULT_ID_WIDTH = 110
env = jinja2.Environment(loader=jinja2.PackageLoader("syft", "assets/jinja"))  # nosec


def create_tabulator_columns(
    column_names: list[str],
    column_widths: dict | None = None,
) -> tuple[list[dict], dict | None]:
    """Returns tuple of (columns, row_header) for tabulator table"""
    if column_widths is None:
        column_widths = {}

    columns = []
    row_header = {}
    if TABLE_INDEX_KEY in column_names:
        row_header = {
            "field": TABLE_INDEX_KEY,
            "headerSort": True,
            "frozen": True,
            "widthGrow": 0.3,
            "minWidth": 60,
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
        return str(data)

    if "badge" in data["type"]:
        return Badge(value=data["value"], badge_class=data["type"]).to_html()
    elif "label" in data["type"]:
        return Label(value=data["value"], label_class=data["type"]).to_html()
    if "clipboard" in data["type"]:
        return CopyButton(copy_text=data["value"]).to_html()

    return str(data)


def format_table_data(table_data: list[dict[str, Any]]) -> list[dict[str, str]]:
    formatted: list[dict[str, str]] = []
    for row in table_data:
        row_formatted: dict[str, str] = {}
        for k, v in row.items():
            if isinstance(v, str):
                row_formatted[k] = v.replace("\n", "<br>")
                continue
            v_formatted = format_dict(v)
            row_formatted[k] = v_formatted
        formatted.append(row_formatted)
    return formatted


def build_tabulator_table(obj: Any) -> str | None:
    try:
        table_data, table_metadata = prepare_table_data(obj)
        if len(table_data) == 0:
            return obj.__repr__()

        table_template = env.get_template("table.jinja2")
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

        column_data, row_header = create_tabulator_columns(table_metadata["columns"])
        table_data = format_table_data(table_data)
        table_html = table_template.render(
            uid=secrets.token_hex(4),
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
        )

        return table_html
    except Exception as e:
        logger.debug("error building table", e)

    return None


def show_table(obj: Any) -> None:
    table = build_tabulator_table(obj)
    if table is not None:
        display(HTML(table))
