from __future__ import annotations

import re
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .job import JobInfo


class StderrViewer:
    """A viewer for stderr content with scrollable display in Jupyter notebooks."""

    def __init__(self, job_info: "JobInfo"):
        self.job_info = job_info

    def _strip_ansi_codes(self, text: str) -> str:
        """Remove ANSI escape sequences from text."""
        ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
        return ansi_escape.sub("", text)

    def _convert_ansi_to_html(self, text: str) -> str:
        """Convert ANSI color codes to HTML spans."""
        ansi_colors = {
            "30": "color: #000000;",
            "31": "color: #cd3131;",
            "32": "color: #00bc00;",
            "33": "color: #e5e510;",
            "34": "color: #0451a5;",
            "35": "color: #bc05bc;",
            "36": "color: #0598bc;",
            "37": "color: #ffffff;",
            "90": "color: #666666;",
            "91": "color: #f14c4c;",
            "92": "color: #23d18b;",
            "93": "color: #f5f543;",
            "94": "color: #3b8eea;",
            "95": "color: #d670d6;",
            "96": "color: #29b8db;",
            "97": "color: #ffffff;",
            "1": "font-weight: bold;",
            "0": "",
        }

        result = text
        result = re.sub(r"\x1b\[0m", "</span>", result)

        for code, style in ansi_colors.items():
            if style:
                pattern = rf"\x1b\[{code}m"
                replacement = f'<span style="{style}">'
                result = re.sub(pattern, replacement, result)

        if "<span" in result and result.count("<span") > result.count("</span>"):
            result += "</span>"

        return result

    def __str__(self) -> str:
        """Return the stderr content with ANSI codes stripped."""
        if self.job_info.status not in ("done", "failed"):
            return "No stderr available - job not completed yet"

        stderr_file = self.job_info.job_review_path / "stderr.txt"

        if not stderr_file.exists():
            return "No stderr file found"

        try:
            with open(stderr_file, "r") as f:
                content = f.read()
                return self._strip_ansi_codes(content)
        except Exception as e:
            return f"Error reading stderr file: {e}"

    def __repr__(self) -> str:
        """Return a brief representation."""
        content = str(self)
        if content.startswith("No stderr") or content.startswith("Error"):
            return content

        lines = content.split("\n")
        if len(lines) <= 3:
            return content
        else:
            return f"StderrViewer({len(lines)} lines, {len(content)} chars)"

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks with scrollable view."""
        if self.job_info.status not in ("done", "failed"):
            error_msg = "No stderr available - job not completed yet"
        else:
            stderr_file = self.job_info.job_review_path / "stderr.txt"

            if not stderr_file.exists():
                error_msg = "No stderr file found"
            else:
                try:
                    with open(stderr_file, "r") as f:
                        raw_content = f.read()
                    error_msg = None
                except Exception as e:
                    error_msg = f"Error reading stderr file: {e}"

        if error_msg:
            return f"""
            <style>
                .syftjob-stderr-empty {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    padding: 20px;
                    text-align: center;
                    border-radius: 8px;
                    background: #f8f9fa;
                    border: 2px dashed #dee2e6;
                    color: #6c757d;
                    font-style: italic;
                }}
            </style>
            <div class="syftjob-stderr-empty">
                📄 {error_msg}
            </div>
            """

        html_content = self._convert_ansi_to_html(raw_content)
        html_content = (
            html_content.replace("&", "&amp;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;")
        )

        clean_content = self._strip_ansi_codes(raw_content)
        lines = clean_content.split("\n")
        char_count = len(clean_content)
        line_count = len(lines)

        return f"""
        <style>
            .syftjob-stderr-container {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                overflow: hidden;
                background: white;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                max-width: 100%;
                margin: 16px 0;
            }}

            .syftjob-stderr-header {{
                background: linear-gradient(135deg, #e53e3e 0%, #c53030 100%);
                color: white;
                padding: 12px 16px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-weight: 600;
            }}

            .syftjob-stderr-title {{
                display: flex;
                align-items: center;
                gap: 8px;
                font-size: 14px;
            }}

            .syftjob-stderr-stats {{
                font-size: 12px;
                opacity: 0.9;
                display: flex;
                gap: 16px;
            }}

            .syftjob-stderr-content {{
                background: #f7fafc;
                border: 1px solid #e2e8f0;
                font-family: 'Monaco', 'Menlo', 'SF Mono', monospace;
                font-size: 12px;
                color: #2d3748;
                padding: 16px;
                overflow: auto;
                white-space: pre-wrap;
                word-wrap: break-word;
                max-height: 400px;
                line-height: 1.5;
                margin: 0;
            }}

            .syftjob-stderr-content::-webkit-scrollbar {{
                width: 8px;
                height: 8px;
            }}

            .syftjob-stderr-content::-webkit-scrollbar-track {{
                background: #f1f1f1;
                border-radius: 4px;
            }}

            .syftjob-stderr-content::-webkit-scrollbar-thumb {{
                background: #c1c1c1;
                border-radius: 4px;
            }}

            .syftjob-stderr-content::-webkit-scrollbar-thumb:hover {{
                background: #a1a1a1;
            }}

            /* Dark theme */
            @media (prefers-color-scheme: dark) {{
                .syftjob-stderr-container {{
                    background: #1a202c;
                    border-color: #4a5568;
                }}

                .syftjob-stderr-header {{
                    background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
                }}

                .syftjob-stderr-content {{
                    background: #2d3748;
                    border-color: #4a5568;
                    color: #e2e8f0;
                }}

                .syftjob-stderr-content::-webkit-scrollbar-track {{
                    background: #2d3748;
                }}

                .syftjob-stderr-content::-webkit-scrollbar-thumb {{
                    background: #4a5568;
                }}

                .syftjob-stderr-content::-webkit-scrollbar-thumb:hover {{
                    background: #718096;
                }}
            }}

            /* Jupyter dark theme */
            .jp-RenderedHTMLCommon[data-jp-theme-light="false"] .syftjob-stderr-container,
            body[data-jp-theme-light="false"] .syftjob-stderr-container {{
                background: #1a202c;
                border-color: #4a5568;
            }}

            .jp-RenderedHTMLCommon[data-jp-theme-light="false"] .syftjob-stderr-header,
            body[data-jp-theme-light="false"] .syftjob-stderr-header {{
                background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
            }}

            .jp-RenderedHTMLCommon[data-jp-theme-light="false"] .syftjob-stderr-content,
            body[data-jp-theme-light="false"] .syftjob-stderr-content {{
                background: #2d3748;
                border-color: #4a5568;
                color: #e2e8f0;
            }}
        </style>

        <div class="syftjob-stderr-container">
            <div class="syftjob-stderr-header">
                <div class="syftjob-stderr-title">
                    🚨 stderr.txt
                </div>
                <div class="syftjob-stderr-stats">
                    <span>{line_count} lines</span>
                    <span>{char_count:,} chars</span>
                </div>
            </div>
            <pre class="syftjob-stderr-content">{html_content}</pre>
        </div>
        """


def _get_python_syntax_highlighted_html(code: str) -> str:
    """Convert Python code to syntax-highlighted HTML."""
    keywords = [
        "and",
        "as",
        "assert",
        "break",
        "class",
        "continue",
        "def",
        "del",
        "elif",
        "else",
        "except",
        "exec",
        "finally",
        "for",
        "from",
        "global",
        "if",
        "import",
        "in",
        "is",
        "lambda",
        "not",
        "or",
        "pass",
        "print",
        "raise",
        "return",
        "try",
        "while",
        "with",
        "yield",
        "True",
        "False",
        "None",
    ]

    code = (
        code.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )

    code = re.sub(
        r"(#.*)",
        r'<span style="color: #6a737d; font-style: italic;">\1</span>',
        code,
    )
    code = re.sub(
        r"(&quot;[^&]*?&quot;)", r'<span style="color: #032f62;">\1</span>', code
    )
    code = re.sub(
        r"(&#x27;[^&]*?&#x27;)", r'<span style="color: #032f62;">\1</span>', code
    )

    for keyword in keywords:
        code = re.sub(
            rf"\b({keyword})\b",
            r'<span style="color: #d73a49; font-weight: bold;">\1</span>',
            code,
        )

    code = re.sub(
        r"\b(def)\s+(\w+)",
        r'<span style="color: #d73a49; font-weight: bold;">\1</span> <span style="color: #6f42c1; font-weight: bold;">\2</span>',
        code,
    )
    code = re.sub(
        r"\b(class)\s+(\w+)",
        r'<span style="color: #d73a49; font-weight: bold;">\1</span> <span style="color: #6f42c1; font-weight: bold;">\2</span>',
        code,
    )

    return code


def job_info_repr_html(job: "JobInfo") -> str:
    """HTML representation for individual job display in Jupyter."""
    submitted_time = "Unknown"
    job_type = "bash"
    try:
        config_file = job.job_submission_path / "config.yaml"
        if config_file.exists():
            from datetime import datetime

            import yaml

            with open(config_file, "r") as f:
                config_data = yaml.safe_load(f)
                job_type = config_data.get("type", "bash")
                submitted_at = config_data.get("submitted_at")

                if submitted_at:
                    try:
                        dt = datetime.fromisoformat(submitted_at.replace("Z", "+00:00"))
                        submitted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        submitted_time = str(submitted_at)
                else:
                    import os

                    mtime = os.path.getmtime(config_file)
                    dt = datetime.fromtimestamp(mtime)
                    submitted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            import os
            from datetime import datetime

            if job.job_submission_path.exists():
                mtime = os.path.getmtime(job.job_submission_path)
                dt = datetime.fromtimestamp(mtime)
                submitted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        submitted_time = "Unknown"

    script_content = "No script available"
    try:
        script_file = job.job_submission_path / "run.sh"
        if script_file.exists():
            with open(script_file, "r") as f:
                script_content = f.read().strip()
                if len(script_content) > 500:
                    script_content = script_content[:500] + "..."
                script_content = (
                    script_content.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace('"', "&quot;")
                    .replace("'", "&#x27;")
                )
    except Exception:
        pass

    code_section = ""
    if job_type == "python":
        try:
            python_files = [
                f for f in job.code_dir.iterdir() if f.suffix == ".py" and f.is_file()
            ]
            if python_files:
                py_file = python_files[0]
                with open(py_file, "r") as f:
                    py_content = f.read()

                highlighted_content = _get_python_syntax_highlighted_html(py_content)

                code_section = f"""
                <div class="syftjob-single-section">
                    <h4>🐍 Code</h4>
                    <div class="syftjob-single-filename">{py_file.name}</div>
                    <div class="syftjob-single-code">{highlighted_content}</div>
                </div>"""
        except Exception:
            pass

    status_emoji_map = {
        "received": "📨",
        "pending": "📥",
        "approved": "✅",
        "rejected": "❌",
        "running": "🔄",
        "done": "🎉",
        "failed": "💥",
    }
    status_emoji = status_emoji_map.get(job.status, "❓")

    outputs_section = ""
    if job.status in ("done", "failed"):
        output_files = job.output_paths
        if output_files:
            outputs_items = "\n".join(
                [
                    f'                        <div class="syftjob-single-outputs-item">📄 {path.name}</div>'
                    for path in output_files
                ]
            )
            outputs_section = f"""
                <div class="syftjob-single-section">
                    <h4>📁 Outputs ({len(output_files)} files)</h4>
                    <div class="syftjob-single-outputs-list">
{outputs_items}
                    </div>
                </div>"""

    return f"""
        <style>
            .syftjob-single {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                border: 2px solid #9CA3AF;
                margin: 16px 0;
                background: white;
                font-size: 14px;
                max-width: 100%;
            }}

            .syftjob-single-header {{
                background: #1F2937;
                color: white;
                padding: 12px 16px;
                border-bottom: 2px solid #111827;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}

            .syftjob-single-title {{
                display: flex;
                align-items: center;
                gap: 12px;
                font-size: 16px;
                font-weight: 700;
                margin: 0;
            }}

            .syftjob-single-status-received {{
                background: #D1D5DB;
                color: #374151;
                padding: 4px 8px;
                border: 2px solid #9CA3AF;
                border-radius: 3px;
                font-size: 11px;
                font-weight: 700;
                display: inline-block;
            }}

            .syftjob-single-status-pending {{
                background: #FBBF24;
                color: #451A03;
                padding: 4px 8px;
                border: 2px solid #B45309;
                border-radius: 3px;
                font-size: 11px;
                font-weight: 700;
                display: inline-block;
            }}

            .syftjob-single-status-approved {{
                background: #60A5FA;
                color: #1E3A8A;
                padding: 4px 8px;
                border: 2px solid #1D4ED8;
                border-radius: 3px;
                font-size: 11px;
                font-weight: 700;
                display: inline-block;
            }}

            .syftjob-single-status-rejected {{
                background: #F87171;
                color: #7F1D1D;
                padding: 4px 8px;
                border: 2px solid #DC2626;
                border-radius: 3px;
                font-size: 11px;
                font-weight: 700;
                display: inline-block;
            }}

            .syftjob-single-status-running {{
                background: #FB923C;
                color: #7C2D12;
                padding: 4px 8px;
                border: 2px solid #EA580C;
                border-radius: 3px;
                font-size: 11px;
                font-weight: 700;
                display: inline-block;
            }}

            .syftjob-single-status-done {{
                background: #34D399;
                color: #064E3B;
                padding: 4px 8px;
                border: 2px solid #047857;
                border-radius: 3px;
                font-size: 11px;
                font-weight: 700;
                display: inline-block;
            }}

            .syftjob-single-status-failed {{
                background: #F87171;
                color: #7F1D1D;
                padding: 4px 8px;
                border: 2px solid #DC2626;
                border-radius: 3px;
                font-size: 11px;
                font-weight: 700;
                display: inline-block;
            }}

            .syftjob-single-content {{
                padding: 16px;
                background: white;
            }}

            .syftjob-single-details {{
                display: grid;
                gap: 12px;
                margin-bottom: 16px;
            }}

            .syftjob-single-detail {{
                display: flex;
                align-items: flex-start;
                gap: 12px;
                padding: 8px;
                background: #F9FAFB;
                border: 1px solid #E5E7EB;
                border-radius: 4px;
            }}

            .syftjob-single-detail-label {{
                color: #374151;
                font-weight: 700;
                min-width: 120px;
                font-size: 13px;
            }}

            .syftjob-single-detail-value {{
                color: #111827;
                font-size: 13px;
                flex: 1;
            }}

            .syftjob-single-section {{
                margin-top: 20px;
                border: 2px solid #E5E7EB;
                border-radius: 4px;
                overflow: hidden;
            }}

            .syftjob-single-section h4 {{
                background: #F3F4F6;
                margin: 0;
                padding: 8px 12px;
                font-size: 14px;
                color: #111827;
                font-weight: 700;
                border-bottom: 2px solid #E5E7EB;
            }}

            .syftjob-single-script {{
                background: #f8f9fa;
                padding: 12px;
                font-family: 'Monaco', 'Menlo', 'SF Mono', monospace;
                font-size: 12px;
                color: #2d3748;
                overflow: auto;
                white-space: pre-wrap;
                word-wrap: break-word;
                max-height: 200px;
                line-height: 1.4;
                margin: 0;
            }}

            .syftjob-single-filename {{
                background: #E5E7EB;
                padding: 6px 12px;
                font-family: 'Monaco', 'Menlo', 'SF Mono', monospace;
                font-size: 11px;
                color: #374151;
                font-weight: 600;
                border-bottom: 1px solid #D1D5DB;
            }}

            .syftjob-single-code {{
                background: #f8f9fa;
                padding: 16px;
                font-family: 'Monaco', 'Menlo', 'SF Mono', monospace;
                font-size: 12px;
                color: #2d3748;
                overflow: auto;
                white-space: pre-wrap;
                word-wrap: break-word;
                max-height: 400px;
                line-height: 1.5;
                margin: 0;
            }}

            .syftjob-single-outputs-list {{
                padding: 12px;
                background: white;
            }}

            .syftjob-single-outputs-item {{
                padding: 4px 0;
                font-family: 'Monaco', 'Menlo', monospace;
                font-size: 12px;
                color: #4a5568;
            }}

            .syftjob-single-code::-webkit-scrollbar,
            .syftjob-single-script::-webkit-scrollbar {{
                width: 8px;
                height: 8px;
            }}

            .syftjob-single-code::-webkit-scrollbar-track,
            .syftjob-single-script::-webkit-scrollbar-track {{
                background: #f1f1f1;
                border-radius: 4px;
            }}

            .syftjob-single-code::-webkit-scrollbar-thumb,
            .syftjob-single-script::-webkit-scrollbar-thumb {{
                background: #c1c1c1;
                border-radius: 4px;
            }}

            .syftjob-single-code::-webkit-scrollbar-thumb:hover,
            .syftjob-single-script::-webkit-scrollbar-thumb:hover {{
                background: #a1a1a1;
            }}

        </style>
        <div class="syftjob-single">
            <div class="syftjob-single-header">
                <h3 class="syftjob-single-title">📋 {job.name}</h3>
                <span class="syftjob-single-status-{job.status}">
                    {status_emoji} {job.status.upper()}
                </span>
            </div>
            <div class="syftjob-single-content">
                <div class="syftjob-single-details">
                    <div class="syftjob-single-detail">
                        <div class="syftjob-single-detail-label">User:</div>
                        <div class="syftjob-single-detail-value">{job.datasite_owner_email}</div>
                    </div>
                    <div class="syftjob-single-detail">
                        <div class="syftjob-single-detail-label">Submitted by:</div>
                        <div class="syftjob-single-detail-value">{job.submitted_by}</div>
                    </div>
                    <div class="syftjob-single-detail">
                        <div class="syftjob-single-detail-label">Location:</div>
                        <div class="syftjob-single-detail-value">{job.job_submission_path}</div>
                    </div>
                    <div class="syftjob-single-detail">
                        <div class="syftjob-single-detail-label">Submitted:</div>
                        <div class="syftjob-single-detail-value">{submitted_time}</div>
                    </div>
                    <div class="syftjob-single-detail">
                        <div class="syftjob-single-detail-label">Approval:</div>
                        <div class="syftjob-single-detail-value">{job.approval_method or "—"}</div>
                    </div>
                    <div class="syftjob-single-detail">
                        <div class="syftjob-single-detail-label">Review reason:</div>
                        <div class="syftjob-single-detail-value">{job.review_reason or ""}</div>
                    </div>
                </div>
                <div class="syftjob-single-section">
                    <h4>📜 Script</h4>
                    <div class="syftjob-single-script">{script_content}</div>
                </div>{code_section}{outputs_section}
            </div>
        </div>
        """


def jobs_list_str(
    jobs: List["JobInfo"], root_email: str, has_do_role: bool = False
) -> str:
    """Format jobs list as separate tables grouped by user."""
    if not jobs:
        return "📭 No jobs found.\n"

    jobs_by_user: dict[str, list["JobInfo"]] = {}
    for job in jobs:
        if job.datasite_owner_email not in jobs_by_user:
            jobs_by_user[job.datasite_owner_email] = []
        jobs_by_user[job.datasite_owner_email].append(job)

    status_emojis = {
        "received": "📨",
        "pending": "📥",
        "approved": "✅",
        "rejected": "❌",
        "running": "🔄",
        "done": "🎉",
        "failed": "💥",
    }

    lines = []
    lines.append("📊 Jobs Overview")
    lines.append("=" * 50)

    total_jobs = 0
    global_status_counts: dict[str, int] = {}

    def user_sort_key(item):
        user_email, _user_jobs = item
        if user_email == root_email:
            return (0, user_email)
        return (1, user_email)

    sorted_users = sorted(jobs_by_user.items(), key=user_sort_key)

    job_index = 0

    for user_email, user_jobs in sorted_users:
        if not user_jobs:
            continue

        total_jobs += len(user_jobs)

        lines.append("")
        lines.append(f"👤 {user_email}")
        lines.append("-" * 60)

        name_width = max(len(job.name) for job in user_jobs) + 2
        status_width = max(len(job.status) for job in user_jobs) + 2
        submitted_width = max(len(job.submitted_by) for job in user_jobs) + 2

        name_width = max(name_width, 15)
        status_width = max(status_width, 12)
        submitted_width = max(submitted_width, 15)

        approval_width = 10

        header = f"{'Index':<6} {'Job Name':<{name_width}} {'Submitted By':<{submitted_width}} {'Status':<{status_width}} {'Approval':<{approval_width}}"
        lines.append(header)
        lines.append("-" * len(header))

        sorted_jobs = user_jobs

        for job in sorted_jobs:
            emoji = status_emojis.get(job.status, "❓")
            status_display = f"{emoji} {job.status}"
            approval_display = job.approval_method or "—"
            line = f"[{job_index:<4}] {job.name:<{name_width}} {job.submitted_by:<{submitted_width}} {status_display:<{status_width}} {approval_display:<{approval_width}}"
            lines.append(line)
            job_index += 1

            global_status_counts[job.status] = (
                global_status_counts.get(job.status, 0) + 1
            )

        user_status_counts: dict[str, int] = {}
        for job in user_jobs:
            user_status_counts[job.status] = user_status_counts.get(job.status, 0) + 1

        summary_parts = []
        for status, count in user_status_counts.items():
            emoji = status_emojis.get(status, "❓")
            summary_parts.append(f"{emoji} {count} {status}")

        lines.append(
            f"📋 {user_email}: {len(user_jobs)} jobs - " + " | ".join(summary_parts)
        )

    lines.append("")
    lines.append("=" * 50)
    lines.append(f"📈 Total: {total_jobs} jobs across {len(jobs_by_user)} users")

    global_summary_parts = []
    for status, count in global_status_counts.items():
        emoji = status_emojis.get(status, "❓")
        global_summary_parts.append(f"{emoji} {count} {status}")

    if global_summary_parts:
        lines.append("📋 Global: " + " | ".join(global_summary_parts))

    if has_do_role:
        lines.append("")
        lines.append(
            "💡 Use job_client.jobs[0].approve() to approve jobs or job_client.jobs[0].accept_by_depositing_result('file_or_folder') to complete jobs"
        )

    return "\n".join(lines)


def jobs_list_repr_html(
    jobs: List["JobInfo"], root_email: str, has_do_role: bool = False
) -> str:
    """HTML representation for Jupyter notebooks with enhanced visual appeal."""
    if not jobs:
        return """
            <style>

                .syftjob-empty {
                    padding: 30px 20px;
                    text-align: center;
                    border-radius: 8px;
                    background: linear-gradient(135deg, #f8c073 0%, #f79763 50%, #cc677b 100%);
                    border: 1px solid rgba(248,192,115,0.2);
                    color: white;
                }


                .syftjob-empty h3 {
                    margin: 0 0 12px 0;
                    font-size: 18px;
                    color: white;
                    font-weight: 600;
                }

                .syftjob-empty p {
                    margin: 0;
                    color: rgba(255,255,255,0.9);
                    font-size: 16px;
                    opacity: 0.95;
                }

                .syftjob-empty-icon {
                    font-size: 24px;
                    margin-bottom: 12px;
                    display: block;
                }

                /* Dark theme */
                @media (prefers-color-scheme: dark) {
                    .syftjob-empty {
                        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
                        border-color: rgba(74,85,104,0.2);
                    }
                    .syftjob-empty h3 {
                        color: white;
                    }
                    .syftjob-empty p {
                        color: rgba(255,255,255,0.95);
                        opacity: 0.95;
                    }
                }

                /* Jupyter dark theme detection */
                .jp-RenderedHTMLCommon[data-jp-theme-light="false"] .syftjob-empty,
                body[data-jp-theme-light="false"] .syftjob-empty {
                    background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
                    border-color: rgba(74,85,104,0.2);
                }
                .jp-RenderedHTMLCommon[data-jp-theme-light="false"] .syftjob-empty h3,
                body[data-jp-theme-light="false"] .syftjob-empty h3 {
                    color: white;
                }
                .jp-RenderedHTMLCommon[data-jp-theme-light="false"] .syftjob-empty p,
                body[data-jp-theme-light="false"] .syftjob-empty p {
                    color: rgba(255,255,255,0.95);
                    opacity: 0.95;
                }
            </style>
            <div class="syftjob-empty">
                <span class="syftjob-empty-icon">📭</span>
                <h3>No jobs found</h3>
                <p>Submit jobs to see them here</p>
            </div>
            """

    jobs_by_user: dict[str, list["JobInfo"]] = {}
    for job in jobs:
        if job.datasite_owner_email not in jobs_by_user:
            jobs_by_user[job.datasite_owner_email] = []
        jobs_by_user[job.datasite_owner_email].append(job)

    status_styles = {
        "received": {
            "emoji": "📨",
            "light": {"color": "#888888", "bg": "#f0f0f0"},
            "dark": {"color": "#aaaaaa", "bg": "#444444"},
        },
        "pending": {
            "emoji": "📥",
            "light": {"color": "#6976ae", "bg": "#e8f2ff"},
            "dark": {"color": "#96d195", "bg": "#52a8c5"},
        },
        "approved": {
            "emoji": "✅",
            "light": {"color": "#53bea9", "bg": "#e6f9f4"},
            "dark": {"color": "#53bea9", "bg": "#2a5d52"},
        },
        "rejected": {
            "emoji": "❌",
            "light": {"color": "#cc3333", "bg": "#ffe8e8"},
            "dark": {"color": "#ff6666", "bg": "#5a2222"},
        },
        "running": {
            "emoji": "🔄",
            "light": {"color": "#cc8800", "bg": "#fff3e0"},
            "dark": {"color": "#ffcc66", "bg": "#5a4422"},
        },
        "done": {
            "emoji": "🎉",
            "light": {"color": "#937098", "bg": "#f3e5f5"},
            "dark": {"color": "#f2d98c", "bg": "#cc677b"},
        },
        "failed": {
            "emoji": "💥",
            "light": {"color": "#cc3333", "bg": "#ffe8e8"},
            "dark": {"color": "#ff6666", "bg": "#5a2222"},
        },
    }

    total_jobs = len(jobs)
    global_status_counts: dict[str, int] = {}
    for job in jobs:
        global_status_counts[job.status] = global_status_counts.get(job.status, 0) + 1

    html = f"""
        <style>
            .syftjob-overview {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 16px 0;
                font-size: 14px;
            }}

            .syftjob-global-header {{
                background: #1F2937;
                color: white;
                padding: 12px 16px;
                border: 2px solid #111827;
                text-align: center;
                margin-bottom: 16px;
            }}

            .syftjob-global-header h3 {{
                margin: 0 0 4px 0;
                font-size: 16px;
                font-weight: 700;
            }}
            .syftjob-global-header p {{
                margin: 0;
                font-size: 13px;
                font-weight: 500;
            }}

            .syftjob-user-section {{
                margin-bottom: 24px;
                border: 2px solid #9CA3AF;
            }}

            .syftjob-user-header {{
                background: #F3F4F6;
                border-bottom: 2px solid #9CA3AF;
                padding: 8px 12px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}

            .syftjob-user-header h4 {{
                margin: 0;
                font-size: 14px;
                font-weight: 700;
                color: #111827;
            }}

            .syftjob-user-summary {{
                font-size: 12px;
                color: #374151;
                font-weight: 600;
            }}

            .syftjob-table {{
                width: 100%;
                border-collapse: collapse;
                background: white;
                font-size: 13px;
                border: 2px solid #6B7280;
            }}

            .syftjob-thead {{
                background: #E5E7EB;
            }}
            .syftjob-th {{
                padding: 8px 12px;
                text-align: left;
                font-weight: 700;
                color: #111827;
                border-right: 2px solid #6B7280;
                border-bottom: 2px solid #6B7280;
            }}
            .syftjob-th:last-child {{ border-right: none; }}

            .syftjob-row-even {{
                background: #FFFFFF;
            }}
            .syftjob-row-odd {{
                background: #F9FAFB;
            }}
            .syftjob-row {{
                border-bottom: 1px solid #9CA3AF;
            }}
            .syftjob-row:hover {{
                background: #DBEAFE !important;
            }}

            .syftjob-td {{
                padding: 8px 12px;
                border-right: 1px solid #9CA3AF;
                vertical-align: middle;
            }}
            .syftjob-td:last-child {{ border-right: none; }}

            .syftjob-index {{
                background: #D1D5DB;
                padding: 4px 8px;
                border-radius: 3px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                font-weight: 700;
                color: #111827;
                border: 2px solid #6B7280;
                display: inline-block;
                min-width: 24px;
                text-align: center;
            }}

            .syftjob-job-name {{
                font-weight: 600;
                color: #111827;
            }}

            .syftjob-status-received {{
                background: #D1D5DB;
                color: #374151;
                padding: 3px 8px;
                border: 2px solid #9CA3AF;
                border-radius: 3px;
                font-size: 11px;
                font-weight: 700;
                display: inline-block;
            }}

            .syftjob-status-pending {{
                background: #FBBF24;
                color: #451A03;
                padding: 3px 8px;
                border: 2px solid #B45309;
                border-radius: 3px;
                font-size: 11px;
                font-weight: 700;
                display: inline-block;
            }}

            .syftjob-status-approved {{
                background: #60A5FA;
                color: #1E3A8A;
                padding: 3px 8px;
                border: 2px solid #1D4ED8;
                border-radius: 3px;
                font-size: 11px;
                font-weight: 700;
                display: inline-block;
            }}

            .syftjob-status-rejected {{
                background: #F87171;
                color: #7F1D1D;
                padding: 3px 8px;
                border: 2px solid #DC2626;
                border-radius: 3px;
                font-size: 11px;
                font-weight: 700;
                display: inline-block;
            }}

            .syftjob-status-running {{
                background: #FB923C;
                color: #7C2D12;
                padding: 3px 8px;
                border: 2px solid #EA580C;
                border-radius: 3px;
                font-size: 11px;
                font-weight: 700;
                display: inline-block;
            }}

            .syftjob-status-done {{
                background: #34D399;
                color: #064E3B;
                padding: 3px 8px;
                border: 2px solid #047857;
                border-radius: 3px;
                font-size: 11px;
                font-weight: 700;
                display: inline-block;
            }}

            .syftjob-status-failed {{
                background: #F87171;
                color: #7F1D1D;
                padding: 3px 8px;
                border: 2px solid #DC2626;
                border-radius: 3px;
                font-size: 11px;
                font-weight: 700;
                display: inline-block;
            }}

            .syftjob-submitted {{
                color: #374151;
                font-size: 12px;
                font-weight: 600;
            }}

            .syftjob-global-footer {{
                background: #F3F4F6;
                padding: 12px 16px;
                text-align: center;
                border: 2px solid #9CA3AF;
                margin-top: 16px;
            }}

            .syftjob-global-summary {{
                display: flex;
                justify-content: center;
                gap: 16px;
                margin-bottom: 12px;
                flex-wrap: wrap;
            }}

            .syftjob-summary-item {{
                display: inline-block;
                font-size: 12px;
                color: #111827;
                padding: 4px 8px;
                background: white;
                border: 2px solid #6B7280;
                border-radius: 3px;
                font-weight: 600;
            }}

            .syftjob-hint {{
                font-size: 12px;
                color: #374151;
                line-height: 1.4;
                margin-top: 8px;
                font-weight: 500;
            }}

            .syftjob-code {{
                background: #E5E7EB;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                border: 2px solid #6B7280;
                font-weight: 600;
                color: #111827;
            }}

        </style>

        <div class="syftjob-overview">
            <div class="syftjob-global-header">
                <h3>📊 Jobs Overview</h3>
                <p>Total: {total_jobs} jobs across {len(jobs_by_user)} users</p>
            </div>
        """

    def user_sort_key(item):
        user_email, _user_jobs = item
        if user_email == root_email:
            return (0, user_email)
        return (1, user_email)

    sorted_users = sorted(jobs_by_user.items(), key=user_sort_key)

    job_index = 0
    for user_email, user_jobs in sorted_users:
        if not user_jobs:
            continue

        sorted_user_jobs = user_jobs

        user_status_counts: dict[str, int] = {}
        for job in user_jobs:
            user_status_counts[job.status] = user_status_counts.get(job.status, 0) + 1

        user_summary_parts = []
        for status, count in user_status_counts.items():
            emoji = status_styles.get(status, {}).get("emoji", "❓")
            user_summary_parts.append(f"{emoji} {count} {status}")

        html += f"""
            <div class="syftjob-user-section">
                <div class="syftjob-user-header">
                    <h4>👤 {user_email}</h4>
                    <div class="syftjob-user-summary">{len(user_jobs)} jobs - {" | ".join(user_summary_parts)}</div>
                </div>
                <table class="syftjob-table">
                    <thead class="syftjob-thead">
                        <tr>
                            <th class="syftjob-th">Index</th>
                            <th class="syftjob-th">Job Name</th>
                            <th class="syftjob-th">Submitted By</th>
                            <th class="syftjob-th">Status</th>
                            <th class="syftjob-th">Approval</th>
                        </tr>
                    </thead>
                    <tbody>
            """

        for i, job in enumerate(sorted_user_jobs):
            style_info = status_styles.get(job.status, {"emoji": "❓"})
            row_class = "syftjob-row-even" if i % 2 == 0 else "syftjob-row-odd"

            approval_display = job.approval_method or "—"
            html += f"""
                        <tr class="{row_class} syftjob-row">
                            <td class="syftjob-td">
                                <span class="syftjob-index">[{job_index}]</span>
                            </td>
                            <td class="syftjob-td syftjob-job-name">
                                {job.name}
                            </td>
                            <td class="syftjob-td syftjob-submitted">
                                {job.submitted_by}
                            </td>
                            <td class="syftjob-td">
                                <span class="syftjob-status-{job.status}">
                                    {style_info["emoji"]} {job.status.upper()}
                                </span>
                            </td>
                            <td class="syftjob-td">
                                {approval_display}
                            </td>
                        </tr>
                """
            job_index += 1

        html += """
                    </tbody>
                </table>
            </div>
            """

    html += """
            <div class="syftjob-global-footer">
                <div class="syftjob-global-summary">
        """

    for status, count in global_status_counts.items():
        style_info = status_styles.get(status, {"emoji": "❓"})
        html += f"""
                    <span class="syftjob-summary-item">
                        {style_info["emoji"]} {count} {status}
                    </span>
            """

    html += """
                </div>
        """
    if has_do_role:
        html += """
                <div class="syftjob-hint">
                    💡 Use <code class="syftjob-code">jobs[0].approve()</code> to approve jobs or <code class="syftjob-code">jobs[0].accept_by_depositing_result('file_or_folder')</code> to complete jobs
                </div>
        """
    html += """
            </div>
        </div>
        """

    return html
