import datetime
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Dict, List, Mapping, Optional, Union
from uuid import UUID

from loguru import logger

from syft_notebook_ui.icons import Icon
from syft_notebook_ui.sanitize import sanitize_html

TABLE_INDEX_KEY = "_table_repr_index"
TABLE_EXTRA_FIELDS = "__table_extra_fields__"
TABLE_COL_WIDTHS = "__table_col_widths__"
CUSTOM_ROW_REPR = "__table_row_repr__"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

ID_FIELD = "uid"
DATE_FIELD = "created_at"
TYPE_FIELD = "type"
MAPPING_KEY_FIELD = "key"
RESERVED_COLUMNS = {
    ID_FIELD,
    TYPE_FIELD,
    MAPPING_KEY_FIELD,
    TABLE_INDEX_KEY,
    DATE_FIELD,
}


def _get_values_for_table_repr(obj: Any) -> list:
    if isinstance(obj, Mapping):
        values = list(obj.values())
    elif isinstance(obj, list):
        values = obj

    return values


def _make_template_columns(first_value: Any) -> tuple[str | None, str | None]:
    grid_template_cols = getattr(first_value, TABLE_COL_WIDTHS, None)
    if isinstance(grid_template_cols, list):
        grid_template_columns = " ".join(grid_template_cols)
        grid_template_cell_columns = "unset"
    else:
        grid_template_columns = None
        grid_template_cell_columns = None
    return grid_template_columns, grid_template_cell_columns


def _get_extra_field_value(obj: Any, field: str) -> Any:
    """
    Retrieves the value of an extra field from an object.
    Extra fields can be nested attributes separated by '.'.

    Args:
        obj (Any): The object from which to retrieve the extra field.
        field (str): The name of the extra field to retrieve.

    Returns:
        Any: The value of the extra field.
    """
    attrs = field.split(".")
    value = obj
    for attr in attrs:
        value = getattr(value, attr, None)
        if value is None:
            break
    return value


def _create_table_rows(
    items: Union[Mapping, Iterable],
    is_homogenous: bool,
    extra_fields: Optional[List[str]] = None,
    add_index: bool = True,
) -> List[Dict[str, Any]]:
    """
    Creates row data for a table based on input object obj.

    If valid table data cannot be created, an empty list is returned.

    Args:
        items (Union[Mapping, Iterable]): The input data as a Mapping or Iterable.
        is_homogenous (bool): A boolean indicating whether the data is homogenous.
        extra_fields (Optional[List], optional): Additional fields to include in the table. Defaults to None.
        add_index (bool, optional): Whether to add an index column. Defaults to True.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries where each dictionary represents a row in the table.
    """
    extra_fields = extra_fields or []
    column_data = defaultdict(list)

    for item in iter(items.items() if isinstance(items, Mapping) else items):
        # Handle mapping items by extracting key and value
        if isinstance(items, Mapping):
            key, item = item
            column_data[MAPPING_KEY_FIELD].append(key)

        # Add ID field if present
        uid = getattr(item, ID_FIELD, None)
        if uid is not None:
            column_data[ID_FIELD].append(uid)

        # Add date_created field if present
        if hasattr(item, DATE_FIELD):
            column_data[DATE_FIELD].append(getattr(item, DATE_FIELD))

        # Add type information for heterogeneous collections
        if not is_homogenous:
            column_data["type"].append(item.__class__.__name__)

        # Process custom row representation if available
        if hasattr(item, CUSTOM_ROW_REPR):
            row_dict = getattr(item, CUSTOM_ROW_REPR)()
            for col_name, value in row_dict.keys():
                if col_name not in RESERVED_COLUMNS:
                    column_data[col_name].append(value)
        else:
            # Process extra fields
            for col_name in extra_fields:
                if col_name in RESERVED_COLUMNS:
                    continue

                try:
                    value = _get_extra_field_value(item, col_name)
                except Exception:
                    value = None

                column_data[col_name].append(value)

    # Validate that all columns have the same length
    unique_col_lengths = {len(column_data[col]) for col in column_data.keys()}
    if not column_data or len(unique_col_lengths) != 1:
        logger.debug("Cannot create table for items with different number of fields.")
        return []

    # Add index column if requested
    num_rows = unique_col_lengths.pop()
    if add_index and TABLE_INDEX_KEY not in column_data:
        column_data[TABLE_INDEX_KEY] = list(range(num_rows))

    # Transpose column data to row data
    row_data = []
    for i in range(num_rows):
        row = {}
        for col_name in column_data:
            row[col_name] = column_data[col_name][i]
        row_data.append(row)

    return row_data


def prepare_table_data(
    obj: Any,
    add_index: bool = True,
) -> tuple[list[dict], dict]:
    """
    Creates table data and metadata for a given object.

    If a tabular representation cannot be created, an empty list and empty dict are returned instead.

    Args:
        obj (Any): The input object for which table data is prepared.
        add_index (bool, optional): Whether to add an index column to the table. Defaults to True.

    Returns:
        tuple: A tuple (table_data, table_metadata) where table_data is a list of dictionaries
        where each dictionary represents a row in the table and table_metadata is a dictionary
        containing metadata about the table such as name, icon, etc.

    """

    values = _get_values_for_table_repr(obj)
    if len(values) == 0:
        return [], {}

    # check first value and obj itself to see if syft in mro. If not, don't create table
    first_value = values[0]

    extra_fields = getattr(first_value, TABLE_EXTRA_FIELDS, [])
    is_homogenous = len({type(x) for x in values}) == 1
    if is_homogenous:
        cls_name = first_value.__class__.__name__
        grid_template_columns, grid_template_cell_columns = _make_template_columns(
            first_value
        )
    else:
        cls_name = ""
        grid_template_columns = None
        grid_template_cell_columns = None

    table_data = _create_table_rows(
        items=obj,
        is_homogenous=is_homogenous,
        extra_fields=extra_fields,
        add_index=add_index,
    )
    # if empty result, collection objects have no table representation
    if not table_data:
        return [], {}

    table_metadata = {
        "name": f"{cls_name} {obj.__class__.__name__.capitalize()}",
        "columns": list(table_data[0].keys()),
        "icon": Icon.TABLE.svg,
        "grid_template_columns": grid_template_columns,
        "grid_template_cell_columns": grid_template_cell_columns,
    }

    return table_data, table_metadata


def format_dict(data: Any) -> str:
    if not isinstance(data, dict):
        return data

    # is_component_dict = set(data.keys()) == {"type", "value"}
    # if is_component_dict and "badge" in data["type"]:
    #     return Badge(value=data["value"], badge_class=data["type"]).to_html()
    # elif is_component_dict and "label" in data["type"]:
    #     return Label(value=data["value"], label_class=data["type"]).to_html()
    # if is_component_dict and "clipboard" in data["type"]:
    #     return CopyButton(copy_text=data["value"]).to_html()

    return sanitize_html(str(data))


def format_uid(uid: UUID) -> str:
    return str(uid)
    # return CopyButton(copy_text=str(uid)).to_html()


def format_table_value(value: Any) -> str:
    """
    Format a single cell value for display in a table.
    TODO add support for more complex types like components from PySyft.
    """
    if isinstance(value, UUID):
        return format_uid(value)
    elif isinstance(value, datetime.datetime):
        return value.strftime(DATETIME_FORMAT)
    elif isinstance(value, dict):
        return format_dict(value)
    else:
        return sanitize_html(str(value).replace("\n", "<br>"))


def format_table_data(table_data: list[dict[str, Any]]) -> list[dict[str, str]]:
    formatted: list[dict[str, str]] = []
    for row in table_data:
        row_formatted: dict[str, str] = {}
        for k, v in row.items():
            row_formatted[k] = format_table_value(v)
        formatted.append(row_formatted)
    return formatted
