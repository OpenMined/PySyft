from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .job import JobInfo


class StdoutViewer:
    """A viewer for stdout content with scrollable display in Jupyter notebooks."""

    def __init__(self, job_info: "JobInfo"):
        self.job_info = job_info

    def _strip_ansi_codes(self, text: str) -> str:
        """Remove ANSI escape sequences from text."""
        # Pattern to match ANSI escape sequences
        ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
        return ansi_escape.sub("", text)

    def _convert_ansi_to_html(self, text: str) -> str:
        """Convert ANSI color codes to HTML spans."""
        # Basic ANSI color code mapping
        ansi_colors = {
            "30": "color: #000000;",  # black
            "31": "color: #cd3131;",  # red
            "32": "color: #00bc00;",  # green
            "33": "color: #e5e510;",  # yellow
            "34": "color: #0451a5;",  # blue
            "35": "color: #bc05bc;",  # magenta
            "36": "color: #0598bc;",  # cyan
            "37": "color: #ffffff;",  # white
            "90": "color: #666666;",  # bright black (gray)
            "91": "color: #f14c4c;",  # bright red
            "92": "color: #23d18b;",  # bright green
            "93": "color: #f5f543;",  # bright yellow
            "94": "color: #3b8eea;",  # bright blue
            "95": "color: #d670d6;",  # bright magenta
            "96": "color: #29b8db;",  # bright cyan
            "97": "color: #ffffff;",  # bright white
            "1": "font-weight: bold;",  # bold
            "0": "",  # reset
        }

        # Replace ANSI codes with HTML
        result = text

        # Handle reset codes first
        result = re.sub(r"\x1b\[0m", "</span>", result)

        # Handle color codes
        for code, style in ansi_colors.items():
            if style:  # Skip empty styles (like reset)
                pattern = rf"\x1b\[{code}m"
                replacement = f'<span style="{style}">'
                result = re.sub(pattern, replacement, result)

        # Handle any remaining unclosed spans by adding a closing span at the end
        if "<span" in result and result.count("<span") > result.count("</span>"):
            result += "</span>"

        return result

    def __str__(self) -> str:
        """Return the stdout content with ANSI codes stripped."""
        if self.job_info.status != "done":
            return "No stdout available - job not completed yet"

        stdout_file = self.job_info.job_review_path / "stdout.txt"

        if not stdout_file.exists():
            return "No stdout file found"

        try:
            with open(stdout_file, "r") as f:
                content = f.read()
                return self._strip_ansi_codes(content)
        except Exception as e:
            return f"Error reading stdout file: {e}"

    def __repr__(self) -> str:
        """Return a brief representation."""
        content = str(self)
        if content.startswith("No stdout") or content.startswith("Error"):
            return content

        lines = content.split("\n")
        if len(lines) <= 3:
            return content
        else:
            return f"StdoutViewer({len(lines)} lines, {len(content)} chars)"

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks with scrollable view."""
        # Get raw content first to check for errors
        if self.job_info.status != "done":
            error_msg = "No stdout available - job not completed yet"
        else:
            stdout_file = self.job_info.job_review_path / "stdout.txt"

            if not stdout_file.exists():
                error_msg = "No stdout file found"
            else:
                try:
                    with open(stdout_file, "r") as f:
                        raw_content = f.read()
                    error_msg = None
                except Exception as e:
                    error_msg = f"Error reading stdout file: {e}"

        # If no content or error, show a simple message
        if error_msg:
            return f"""
            <style>
                .syftjob-stdout-empty {{
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
            <div class="syftjob-stdout-empty">
                📄 {error_msg}
            </div>
            """

        # Convert ANSI codes to HTML for display
        html_content = self._convert_ansi_to_html(raw_content)

        # Escape any remaining HTML characters that aren't our color spans
        # We need to be careful not to escape our intentional HTML
        html_content = (
            html_content.replace("&", "&amp;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;")
        )
        # Don't escape < and > since we want our HTML spans to work

        # Count lines and characters (use clean content for stats)
        clean_content = self._strip_ansi_codes(raw_content)
        lines = clean_content.split("\n")
        char_count = len(clean_content)
        line_count = len(lines)

        return f"""
        <style>
            .syftjob-stdout-container {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                overflow: hidden;
                background: white;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                max-width: 100%;
                margin: 16px 0;
            }}

            .syftjob-stdout-header {{
                background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
                color: white;
                padding: 12px 16px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-weight: 600;
            }}

            .syftjob-stdout-title {{
                display: flex;
                align-items: center;
                gap: 8px;
                font-size: 14px;
            }}

            .syftjob-stdout-stats {{
                font-size: 12px;
                opacity: 0.9;
                display: flex;
                gap: 16px;
            }}

            .syftjob-stdout-content {{
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

            .syftjob-stdout-content::-webkit-scrollbar {{
                width: 8px;
                height: 8px;
            }}

            .syftjob-stdout-content::-webkit-scrollbar-track {{
                background: #f1f1f1;
                border-radius: 4px;
            }}

            .syftjob-stdout-content::-webkit-scrollbar-thumb {{
                background: #c1c1c1;
                border-radius: 4px;
            }}

            .syftjob-stdout-content::-webkit-scrollbar-thumb:hover {{
                background: #a1a1a1;
            }}

            /* Dark theme */
            @media (prefers-color-scheme: dark) {{
                .syftjob-stdout-container {{
                    background: #1a202c;
                    border-color: #4a5568;
                }}

                .syftjob-stdout-header {{
                    background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
                }}

                .syftjob-stdout-content {{
                    background: #2d3748;
                    border-color: #4a5568;
                    color: #e2e8f0;
                }}

                .syftjob-stdout-content::-webkit-scrollbar-track {{
                    background: #2d3748;
                }}

                .syftjob-stdout-content::-webkit-scrollbar-thumb {{
                    background: #4a5568;
                }}

                .syftjob-stdout-content::-webkit-scrollbar-thumb:hover {{
                    background: #718096;
                }}
            }}

            /* Jupyter dark theme */
            .jp-RenderedHTMLCommon[data-jp-theme-light="false"] .syftjob-stdout-container,
            body[data-jp-theme-light="false"] .syftjob-stdout-container {{
                background: #1a202c;
                border-color: #4a5568;
            }}

            .jp-RenderedHTMLCommon[data-jp-theme-light="false"] .syftjob-stdout-header,
            body[data-jp-theme-light="false"] .syftjob-stdout-header {{
                background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
            }}

            .jp-RenderedHTMLCommon[data-jp-theme-light="false"] .syftjob-stdout-content,
            body[data-jp-theme-light="false"] .syftjob-stdout-content {{
                background: #2d3748;
                border-color: #4a5568;
                color: #e2e8f0;
            }}
        </style>

        <div class="syftjob-stdout-container">
            <div class="syftjob-stdout-header">
                <div class="syftjob-stdout-title">
                    📄 stdout.txt
                </div>
                <div class="syftjob-stdout-stats">
                    <span>{line_count} lines</span>
                    <span>{char_count:,} chars</span>
                </div>
            </div>
            <pre class="syftjob-stdout-content">{html_content}</pre>
        </div>
        """
