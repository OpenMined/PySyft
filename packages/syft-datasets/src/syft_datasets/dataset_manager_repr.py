from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from .dataset import Dataset


_EMPTY_HTML = """
<style>
    .syft-datasets-empty {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 16px 20px;
        background: #fafafa;
        color: #1f2937;
        margin: 12px 0;
        max-width: 640px;
    }
    .syft-datasets-empty h4 {
        margin: 0 0 12px 0;
        font-size: 15px;
        font-weight: 700;
    }
    .syft-datasets-empty .hint {
        margin-top: 8px;
        font-weight: 600;
        color: #374151;
    }
    .syft-datasets-empty ol {
        margin: 6px 0 0 0;
        padding-left: 22px;
        color: #4b5563;
        font-size: 13px;
        line-height: 1.55;
    }
    .syft-datasets-empty code {
        background: #eef2ff;
        padding: 1px 5px;
        border-radius: 3px;
        font-family: 'Monaco', 'Menlo', 'SF Mono', monospace;
        font-size: 12px;
        color: #1e3a8a;
    }
    @media (prefers-color-scheme: dark) {
        .syft-datasets-empty { background: #1f2937; color: #e5e7eb; border-color: #374151; }
        .syft-datasets-empty .hint { color: #d1d5db; }
        .syft-datasets-empty ol { color: #cbd5e1; }
        .syft-datasets-empty code { background: #1e3a8a; color: #dbeafe; }
    }
</style>
<div class="syft-datasets-empty">
    <h4>📦 No datasets available yet.</h4>
    <div class="hint">💡 Possible reasons:</div>
    <ol>
        <li>Your peers haven't created any datasets yet.</li>
        <li>You need to sync first — try: <code>client.sync()</code></li>
        <li>You're not connected to any peers yet.</li>
    </ol>
</div>
"""


def _format_tags(tags: list[str]) -> str:
    if not tags:
        return ""
    return "[" + ", ".join(escape(t) for t in tags) + "]"


def _file_count_label(n: int) -> str:
    return f"{n} mock file" if n == 1 else f"{n} mock files"


def _dataset_row_html(dataset: "Dataset") -> str:
    name = escape(dataset.name)
    owner = escape(dataset.owner)
    file_count = _file_count_label(len(dataset.mock_files_urls))
    tags = _format_tags(dataset.tags)
    return (
        '<div class="syft-datasets-row">'
        f'<span class="syft-datasets-name">{name}</span>'
        f'<span class="syft-datasets-from">from: {owner}</span>'
        f'<span class="syft-datasets-files">{file_count}</span>'
        f'<span class="syft-datasets-tags">{tags}</span>'
        "</div>"
    )


def _access_snippet_html(first: "Dataset") -> str:
    name = escape(first.name)
    owner = escape(first.owner)
    return (
        '<div class="syft-datasets-hint">💡 Access a dataset with:</div>'
        '<pre class="syft-datasets-snippet">'
        f'dataset = client.datasets.get("{name}", datasite="{owner}")\n'
        "contents = dataset.mock_files[0].read_text()"
        "</pre>"
    )


def dataset_manager_repr_html(datasets: Sequence["Dataset"]) -> str:
    if not datasets:
        return _EMPTY_HTML

    rows = "\n".join(_dataset_row_html(d) for d in datasets)
    snippet = _access_snippet_html(datasets[0])
    count = len(datasets)
    return f"""
<style>
    .syft-datasets-summary {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 14px 18px;
        background: #ffffff;
        color: #111827;
        margin: 12px 0;
        max-width: 720px;
    }}
    .syft-datasets-summary h4 {{
        margin: 0 0 10px 0;
        font-size: 15px;
        font-weight: 700;
    }}
    .syft-datasets-rows {{
        font-family: 'Monaco', 'Menlo', 'SF Mono', monospace;
        font-size: 13px;
        margin-bottom: 12px;
    }}
    .syft-datasets-row {{
        display: grid;
        grid-template-columns: minmax(160px, 1.5fr) minmax(180px, 1.5fr) minmax(110px, 0.8fr) minmax(140px, 1fr);
        column-gap: 16px;
        padding: 4px 0;
        align-items: baseline;
    }}
    .syft-datasets-name {{ color: #1d4ed8; font-weight: 600; }}
    .syft-datasets-from {{ color: #374151; }}
    .syft-datasets-files {{ color: #6b7280; }}
    .syft-datasets-tags {{ color: #6d28d9; }}
    .syft-datasets-hint {{
        margin-top: 6px;
        font-weight: 600;
        color: #374151;
        font-size: 13px;
    }}
    .syft-datasets-snippet {{
        background: #f3f4f6;
        border: 1px solid #e5e7eb;
        border-radius: 4px;
        padding: 10px 12px;
        margin: 6px 0 0 0;
        font-family: 'Monaco', 'Menlo', 'SF Mono', monospace;
        font-size: 12px;
        color: #111827;
        white-space: pre-wrap;
        line-height: 1.5;
    }}
    @media (prefers-color-scheme: dark) {{
        .syft-datasets-summary {{ background: #111827; color: #e5e7eb; border-color: #374151; }}
        .syft-datasets-name {{ color: #93c5fd; }}
        .syft-datasets-from {{ color: #cbd5e1; }}
        .syft-datasets-files {{ color: #9ca3af; }}
        .syft-datasets-tags {{ color: #c4b5fd; }}
        .syft-datasets-hint {{ color: #d1d5db; }}
        .syft-datasets-snippet {{ background: #1f2937; color: #e5e7eb; border-color: #374151; }}
    }}
</style>
<div class="syft-datasets-summary">
    <h4>📦 Available datasets ({count}):</h4>
    <div class="syft-datasets-rows">
{rows}
    </div>
    {snippet}
</div>
"""
