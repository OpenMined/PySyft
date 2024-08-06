# stdlib
from collections import defaultdict
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Set
import json
import logging
import re
from typing import Any

# third party
import itables
import pandas as pd

# relative
from .util import full_name_with_qualname
from .util import sanitize_html

TABLE_INDEX_KEY = "_table_repr_index"

logger = logging.getLogger(__name__)


def _syft_in_mro(self: Any, item: Any) -> bool:
    if hasattr(type(item), "mro") and type(item) != type:
        # if unbound method, supply self
        if hasattr(type(item).mro, "__self__"):
            mro = type(item).mro()
        else:
            mro = type(item).mro(type(item))  # type: ignore

    elif hasattr(item, "mro") and type(item) != type:
        mro = item.mro()
    else:
        mro = str(self)  # type: ignore

    return "syft" in str(mro).lower()


def _get_values_for_table_repr(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        values = list(obj.values())
    elif isinstance(obj, Set):
        values = list(obj)
    else:
        values = obj

    return values


def _get_grid_template_columns(first_value: Any) -> tuple[str | None, str | None]:
    grid_template_cols = getattr(first_value, "__table_coll_widths__", None)
    if isinstance(grid_template_cols, list):
        grid_template_columns = " ".join(grid_template_cols)
        grid_template_cell_columns = "unset"
    else:
        grid_template_columns = None
        grid_template_cell_columns = None
    return grid_template_columns, grid_template_cell_columns


def _create_table_rows(
    _self: Mapping | Iterable,
    is_homogenous: bool,
    extra_fields: list | None = None,
    add_index: bool = True,
) -> list[dict[str, Any]]:
    """
    Creates row data for a table based on input object obj.

    If valid table data cannot be created, an empty list is returned.

    Args:
        _self (Mapping | Iterable): The input data as a Mapping or Iterable.
        is_homogenous (bool): A boolean indicating whether the data is homogenous.
        extra_fields (list | None, optional): Additional fields to include in the table. Defaults to None.
        add_index (bool, optional): Whether to add an index column. Defaults to True.

    Returns:
        list[dict[str, Any]]: A list of dictionaries where each dictionary represents a row in the table.

    """

    if extra_fields is None:
        extra_fields = []

    cols = defaultdict(list)

    for item in iter(_self.items() if isinstance(_self, Mapping) else _self):
        # unpack dict
        if isinstance(_self, Mapping):
            key, item = item
            cols["key"].append(key)

        # get id
        id_ = getattr(item, "id", None)
        include_id = getattr(item, "__syft_include_id_coll_repr__", True)
        if id_ is not None and include_id:
            cols["id"].append({"value": str(id_), "type": "clipboard"})

        if type(item) == type:
            t = full_name_with_qualname(item)
        else:
            try:
                t = item.__class__.__name__
            except Exception:
                t = item.__repr__()

        if not is_homogenous:
            cols["type"].append(t)

        # if has _coll_repr_

        if hasattr(item, "_coll_repr_"):
            ret_val = item._coll_repr_()
            if "id" in ret_val:
                del ret_val["id"]
            for key in ret_val.keys():
                cols[key].append(ret_val[key])
        else:
            for field in extra_fields:
                value = item
                try:
                    attrs = field.split(".")
                    for i, attr in enumerate(attrs):
                        # find indexing like abc[1]
                        res = re.search(r"\[[+-]?\d+\]", attr)
                        has_index = False
                        if res:
                            has_index = True
                            index_str = res.group()
                            index = int(index_str.replace("[", "").replace("]", ""))
                            attr = attr.replace(index_str, "")

                        value = getattr(value, attr, None)
                        if isinstance(value, list) and has_index:
                            value = value[index]
                        # If the object has a special representation when nested we will use that instead
                        if (
                            hasattr(value, "__repr_syft_nested__")
                            and i == len(attrs) - 1
                        ):
                            value = value.__repr_syft_nested__()
                        if (
                            isinstance(value, list)
                            and i == len(attrs) - 1
                            and len(value) > 0
                            and hasattr(value[0], "__repr_syft_nested__")
                        ):
                            value = [
                                (
                                    x.__repr_syft_nested__()
                                    if hasattr(x, "__repr_syft_nested__")
                                    else x
                                )
                                for x in value
                            ]
                    if value is None:
                        value = "n/a"

                except Exception as e:
                    print(e)
                    value = None
                cols[field].append(sanitize_html(str(value)))

    col_lengths = {len(cols[col]) for col in cols.keys()}
    if len(col_lengths) != 1:
        logger.debug("Cannot create table for items with different number of fields.")
        return []

    num_rows = col_lengths.pop()
    if add_index and TABLE_INDEX_KEY not in cols:
        cols[TABLE_INDEX_KEY] = list(range(num_rows))

    # NOTE cannot use Pandas, not all values can be in a DataFrame (dict/list/...)
    rows = []
    for i in range(num_rows):
        row = {}
        for col in cols.keys():
            row[col] = cols[col][i]
        rows.append(row)

    return rows


def _sort_table_rows(rows: list[dict[str, Any]], sort_key: str) -> list[dict[str, Any]]:
    try:
        sort_values = [row[sort_key] for row in rows]
    except KeyError:
        # Not all rows have the sort_key, do not sort
        return rows

    # relative
    from ..types.datetime import DateTime
    from ..types.datetime import str_is_datetime

    if all(isinstance(v, str) and str_is_datetime(v) for v in sort_values):
        sort_values = [DateTime.from_str(v) for v in sort_values]

    reverse_sort = False
    if isinstance(sort_values[0], DateTime):
        sort_values = [d.utc_timestamp for d in sort_values]
        reverse_sort = True

    rows_sorted = [
        row
        for _, row in sorted(
            zip(sort_values, rows),
            reverse=reverse_sort,
            key=lambda pair: pair[0],
        )
    ]

    return rows_sorted


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
    if not _syft_in_mro(obj, first_value):
        return [], {}

    extra_fields = getattr(first_value, "__repr_attrs__", [])
    is_homogenous = len({type(x) for x in values}) == 1
    if is_homogenous:
        sort_key = getattr(first_value, "__table_sort_attr__", None) or "created_at"
        cls_name = first_value.__class__.__name__
        grid_template_columns, grid_template_cell_columns = _get_grid_template_columns(
            first_value
        )
    else:
        sort_key = "created_at"
        cls_name = ""
        grid_template_columns = None
        grid_template_cell_columns = None

    table_data = _create_table_rows(
        _self=obj,
        is_homogenous=is_homogenous,
        extra_fields=extra_fields,
        add_index=add_index,
    )
    # if empty result, collection objects have no table representation
    if not table_data:
        return [], {}

    table_data = _sort_table_rows(table_data, sort_key)

    table_metadata = {
        "name": f"{cls_name} {obj.__class__.__name__.capitalize()}",
        "columns": list(table_data[0].keys()),
        "icon": getattr(first_value, "icon", None),
        "grid_template_columns": grid_template_columns,
        "grid_template_cell_columns": grid_template_cell_columns,
    }

    return table_data, table_metadata


def itable_template_from_df(df: pd.DataFrame, itable_css: str | None = None) -> str:
    """
    Generate an itable template from a pandas DataFrame.

    The itable template contains a JSON string that can be used to render an itable downstream
    by the patched ipython.

    Args:
        df (pd.DataFrame): The DataFrame to generate the template from.
        itable_css (str | None, optional): The CSS styles to apply to the itable template. Defaults to None.

    Returns:
        str: The generated itable template as a string.

    """
    itable_template = f"""<!-- Start itable_template -->
    {json.dumps({"columns": df.columns.tolist(),
                    "data": df.values.tolist(),
                    "itable_css": itable_css})}
    <!-- End itable_template -->"""
    return itable_template


def render_itable_template(itable_str: str) -> str:
    """
    Renders an itable template string into an HTML table using itables.to_html_datatable.

    Args:
        itable_str (str): The itable template string to render.

    Returns:
        str: The rendered HTML table.

    """

    df, itable_css = _extract_df_from_itable_template(itable_str)
    if itable_css:
        return itables.to_html_datatable(df=df, css=itable_css)
    else:
        return itables.to_html_datatable(df=df)


def _extract_df_from_itable_template(
    itable_str: str,
) -> tuple[pd.DataFrame, str | None]:
    """
    Extracts a DataFrame and CSS styles from an itable template string.

    Args:
        itable_str (str): The itable template string to extract from.

    Returns:
        tuple[pd.DataFrame, str | None]: A tuple containing the extracted DataFrame and the CSS styles.
            - The DataFrame is created using the columns and data from the itable template.
            - The CSS styles are extracted from the itable template, or None if not present.

    """
    json_data = json.loads(itable_str)
    extracted_df = pd.DataFrame(columns=json_data["columns"], data=json_data["data"])
    return extracted_df, json_data.get("itable_css", None)
