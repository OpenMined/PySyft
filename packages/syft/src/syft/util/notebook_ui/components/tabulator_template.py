import json
import secrets
from syft.util.assets import load_css, load_js
from .sync import Badge, Label, CopyButton
from ...table import TABLE_INDEX_KEY, prepare_table_data
import jinja2
from IPython.display import display, HTML

DEFAULT_ID_WIDTH = 110
DEFAULT_INDEX_WIDTH = 50
env = jinja2.Environment(loader=jinja2.PackageLoader("syft", "assets/jinja"))


def create_tabulator_columns(
    column_names: list[str],
    column_widths: dict | None = None,
) -> list[dict]:
    if column_widths is None:
        column_widths = {}

    if TABLE_INDEX_KEY in column_names:
        index_column = {
            "title": "",
            "field": TABLE_INDEX_KEY,
            "formatter": "plaintext",
            "width": DEFAULT_INDEX_WIDTH,
            "resizable": False,
        }

    columns = [
        {
            "title": colname,
            "field": colname,
            "formatter": "html",
            "resizable": True,
        }
        for colname in column_names
        if colname != TABLE_INDEX_KEY
    ]

    columns = [index_column] + columns

    # Prevent resizing out of bounds
    columns[0]["resizable"] = False
    columns[-1]["resizable"] = False

    for col in columns:
        if col["field"] == "id":
            col["width"] = DEFAULT_ID_WIDTH
        if col["field"] in column_widths:
            col["widthGrow"] = column_widths[col["field"]]

    return columns


def dict_to_html(data: dict) -> str:
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


def format_table_data(table_data: list[dict]) -> list[dict]:
    formatted = []
    for row in table_data:
        row_formatted = {}
        for k, v in row.items():
            row_formatted[k] = dict_to_html(v)
        formatted.append(row_formatted)
    return formatted


table_template = env.get_template("table.j2")
js = load_js("table.js")
css = load_css("style.css")


def show_table(obj):
    try:
        table_data, table_metadata = prepare_table_data(obj)
        if len(table_data) == 0:
            return obj.__repr__()
        colnames = list(table_data[0].keys())

        column_data = create_tabulator_columns(colnames)
        column_data[0]["resizable"] = True
        table_data = format_table_data(table_data)

        table_html = table_template.render(
            uid=secrets.token_hex(4),
            columns=json.dumps(column_data),
            data=json.dumps(table_data),
            css=css,
            js=js,
            index_field_name=TABLE_INDEX_KEY,
        )

        display(HTML(table_html))
    except Exception as e:
        print("error building table", e)

    return None
