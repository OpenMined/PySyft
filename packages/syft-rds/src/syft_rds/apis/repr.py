"""Text and HTML views of apis, styled like the jobs table."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from syft_rds.apis.api import Api

_STYLE = """
<style>
    .syftapi { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin: 16px 0; font-size: 14px; }
    .syftapi-header { background: #1F2937; color: white; padding: 12px 16px;
        border: 2px solid #111827; margin-bottom: 12px; }
    .syftapi-header h3 { margin: 0 0 4px 0; font-size: 16px; font-weight: 700; }
    .syftapi-header p { margin: 0; font-size: 13px; font-weight: 500; }
    .syftapi-table { width: 100%; border-collapse: collapse; background: white;
        font-size: 13px; border: 2px solid #6B7280; color: #111827; }
    .syftapi-th { background: #E5E7EB; padding: 8px 12px; text-align: left;
        font-weight: 700; border-right: 2px solid #6B7280;
        border-bottom: 2px solid #6B7280; }
    .syftapi-td { padding: 8px 12px; border-right: 1px solid #9CA3AF;
        border-bottom: 1px solid #9CA3AF; vertical-align: middle; }
    .syftapi-table tr:nth-child(odd) td { background: #F9FAFB; }
    .syftapi-name { font-weight: 600; }
    .syftapi-code { font-family: 'Consolas', 'Courier New', monospace;
        font-size: 12px; }
    .syftapi-arg { background: #DBEAFE; color: #1E3A8A; padding: 2px 6px;
        border-radius: 3px; margin-right: 4px; font-family: monospace;
        font-size: 12px; }
    .syftapi-pre { background: #F3F4F6; border: 1px solid #D1D5DB; padding: 10px;
        font-family: 'Consolas', 'Courier New', monospace; font-size: 12px;
        white-space: pre; overflow-x: auto; color: #111827; }
    .syftapi-muted { color: #6B7280; }
</style>
"""

_EMPTY_HTML = (
    _STYLE
    + """
<div class="syftapi">
    <div class="syftapi-header">
        <h3>No apis found</h3>
        <p>Apis a datasite shares with you show up here after client.sync()</p>
    </div>
</div>
"""
)


def _args_html(api: Api) -> str:
    if not api.args:
        return '<span class="syftapi-muted">-</span>'
    return "".join(f'<span class="syftapi-arg">{escape(a)}</span>' for a in api.args)


def _row_html(index: int, api: Api) -> str:
    call = escape(api.call_signature) if api.is_callable else "not callable"
    code = escape(api.layout.python_file) if api.layout else "-"
    return (
        "<tr>"
        f'<td class="syftapi-td syftapi-code">{index}</td>'
        f'<td class="syftapi-td syftapi-name">{escape(api.name)}</td>'
        f'<td class="syftapi-td">{escape(api.datasite)}</td>'
        f'<td class="syftapi-td">{_args_html(api)}</td>'
        f'<td class="syftapi-td syftapi-code">{code}</td>'
        f'<td class="syftapi-td syftapi-code">{call}</td>'
        "</tr>"
    )


def api_collection_repr_html(apis: Sequence[Api]) -> str:
    if not apis:
        return _EMPTY_HTML
    headers = ["#", "Name", "Datasite", "Args", "Code", "Call"]
    head = "".join(f'<th class="syftapi-th">{h}</th>' for h in headers)
    rows = "".join(_row_html(i, api) for i, api in enumerate(apis))
    return f"""{_STYLE}
<div class="syftapi">
    <div class="syftapi-header">
        <h3>Apis</h3>
        <p>{len(apis)} api(s) shared with you</p>
    </div>
    <table class="syftapi-table"><thead><tr>{head}</tr></thead>
    <tbody>{rows}</tbody></table>
</div>
"""


def api_collection_repr_str(apis: Sequence[Api]) -> str:
    if not apis:
        return "No apis found. Apis shared with you show up after client.sync()."
    lines = [f"Apis ({len(apis)}):"]
    for i, api in enumerate(apis):
        call = api.call_signature if api.is_callable else "not callable"
        lines.append(f"  [{i}] {api.name} (from {api.datasite}) -> {call}")
    return "\n".join(lines)


def api_repr_html(api: Api) -> str:
    return f"""{_STYLE}
<div class="syftapi">
    <div class="syftapi-header">
        <h3>{escape(api.name)}</h3>
        <p>from {escape(api.datasite)}</p>
    </div>
    <p><b>Args:</b> {_args_html(api)}</p>
    {_api_body_html(api)}
</div>
"""


def _api_body_html(api: Api) -> str:
    if api.layout is None:
        files = ", ".join(e.relative_path for e in api.definition.file_contents)
        return (
            f"<p><b>Files:</b> {escape(files)}</p>"
            '<p class="syftapi-muted">This api can\'t be called with arguments.</p>'
        )
    return (
        f'<p><b>Call:</b> <span class="syftapi-code">'
        f"{escape(api.call_signature)}</span></p>"
        f"<p><b>Runs</b> {escape(api.layout.python_file)}:</p>"
        f'<div class="syftapi-pre">{escape(api.code or "")}</div>'
    )


def api_repr_str(api: Api) -> str:
    header = f"Api '{api.name}' from {api.datasite}\nArgs: {', '.join(api.args) or '-'}"
    if api.layout is None:
        return f"{header}\nThis api can't be called with arguments."
    return (
        f"{header}\nCall: {api.call_signature}\n"
        f"Runs {api.layout.python_file}:\n\n{api.code}"
    )
