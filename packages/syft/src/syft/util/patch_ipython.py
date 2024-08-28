# stdlib
import html
import re
from types import MethodType
from typing import Any

# relative
from ..service.response import SyftResponseMessage
from ..types.dicttuple import DictTuple
from ..types.syft_object import SyftObject
from .table import render_itable_template
from .util import sanitize_html


def _patch_ipython_sanitization() -> None:
    try:
        # third party
        from IPython import get_ipython
    except ImportError:
        return

    ip = get_ipython()
    if ip is None:
        return

    # stdlib
    from importlib import resources

    # relative
    from .assets import load_css
    from .assets import load_js
    from .notebook_ui.components.sync import ALERT_CSS
    from .notebook_ui.components.sync import COPY_CSS
    from .notebook_ui.styles import CSS_CODE
    from .notebook_ui.styles import FONT_CSS
    from .notebook_ui.styles import JS_DOWNLOAD_FONTS

    tabulator_js = load_js("tabulator.min.js")
    tabulator_js = tabulator_js.replace(
        "define(t)", "define('tabulator-tables', [], t)"
    )

    SKIP_SANITIZE = [
        FONT_CSS,
        CSS_CODE,
        JS_DOWNLOAD_FONTS,
        tabulator_js,
        load_css("tabulator_pysyft.min.css"),
        load_js("table.js"),
    ]

    css_reinsert = f"""
<style>{FONT_CSS}</style>
{JS_DOWNLOAD_FONTS}
{CSS_CODE}
<style>{ALERT_CSS}</style>
<style>{COPY_CSS}</style>
"""

    escaped_js_css = re.compile(
        "|".join(re.escape(substr) for substr in SKIP_SANITIZE),
        re.IGNORECASE | re.MULTILINE,
    )

    table_template = (
        resources.files("syft.assets.jinja").joinpath("table.jinja2").read_text()
    )
    table_template = table_template.strip()
    table_template = re.sub(r"\\{\\{.*?\\}\\}", ".*?", re.escape(table_template))
    escaped_template = re.compile(table_template, re.DOTALL | re.VERBOSE)

    jobs_repr_template = (
        r"<!-- Start job_repr_template -->(.*?)<!-- End job_repr_template -->"
    )
    jobs_pattern = re.compile(jobs_repr_template, re.DOTALL)

    itable_template = (
        r"<!-- Start itable_template -->\s*(.*?)\s*<!-- End itable_template -->"
    )
    escaped_itable_template = re.compile(itable_template, re.DOTALL)

    def display_sanitized_html(obj: SyftObject | DictTuple) -> str | None:
        if callable(obj_repr_html_ := getattr(obj, "_repr_html_", None)):
            html_str = obj_repr_html_()
            if html_str is not None:
                # find matching table and jobs
                matching_table = escaped_template.findall(html_str)
                matching_jobs = jobs_pattern.findall(html_str)
                matching_itables = escaped_itable_template.findall(html_str)
                template = "\n".join(matching_table + matching_jobs)

                # remove escaped tables from sanitized html
                sanitized_str = escaped_template.sub("", html_str)
                # remove escaped js/css from sanitized html
                sanitized_str = escaped_js_css.sub("", sanitized_str)

                # remove jobs from sanitized html
                sanitized_str = jobs_pattern.sub("", sanitized_str)

                # remove escaped itables from sanitized html
                sanitized_str = escaped_itable_template.sub(
                    "SYFT_PLACEHOLDER_ITABLE", sanitized_str
                )
                sanitized_str = sanitize_html(sanitized_str)

                # add back css / js that skips sanitization

                for matching_itable in matching_itables:
                    sanitized_str = sanitized_str.replace(
                        "SYFT_PLACEHOLDER_ITABLE",
                        render_itable_template(matching_itable),
                        1,
                    )
                return f"{css_reinsert} {sanitized_str} {template}"
        return None

    def display_sanitized_md(obj: SyftObject) -> str | None:
        if callable(getattr(obj, "_repr_markdown_", None)):
            md = obj._repr_markdown_()
            if md is not None:
                md_sanitized = sanitize_html(md)
                return html.unescape(md_sanitized)
        return None

    ip.display_formatter.formatters["text/html"].for_type(
        SyftObject, display_sanitized_html
    )
    ip.display_formatter.formatters["text/html"].for_type(
        DictTuple, display_sanitized_html
    )
    ip.display_formatter.formatters["text/markdown"].for_type(
        SyftObject, display_sanitized_md
    )
    ip.display_formatter.formatters["text/html"].for_type(
        SyftResponseMessage, display_sanitized_html
    )


def _patch_ipython_autocompletion() -> None:
    try:
        # third party
        from IPython import get_ipython
        from IPython.core.guarded_eval import EVALUATION_POLICIES
    except ImportError:
        return

    ipython = get_ipython()
    if ipython is None:
        return

    try:
        # this allows property getters to be used in nested autocomplete
        ipython.Completer.evaluation = "limited"
        ipython.Completer.use_jedi = False
        policy = EVALUATION_POLICIES["limited"]

        policy.allowed_getattr_external.update(
            [
                ("syft.client.api", "APIModule"),
                ("syft.client.api", "SyftAPI"),
            ]
        )
        original_can_get_attr = policy.can_get_attr

        def patched_can_get_attr(value: Any, attr: str) -> bool:
            attr_name = "__syft_allow_autocomplete__"
            # first check if exist to prevent side effects
            if hasattr(value, attr_name) and attr in getattr(value, attr_name, []):
                if attr in dir(value):
                    return True
                else:
                    return False
            else:
                return original_can_get_attr(value, attr)

        policy.can_get_attr = patched_can_get_attr
    except Exception:
        print("Failed to patch ipython autocompletion for syft property getters")

    try:
        # this constraints the completions for autocomplete.
        # if __syft_dir__ is defined we only autocomplete those properties
        original_attr_matches = ipython.Completer.attr_matches

        def patched_attr_matches(self, text: str) -> list[str]:  # type: ignore
            res = original_attr_matches(text)
            m2 = re.match(r"(.+)\.(\w*)$", self.line_buffer)
            if not m2:
                return res
            expr, _ = m2.group(1, 2)
            obj = self._evaluate_expr(expr)
            if isinstance(obj, SyftObject) and hasattr(obj, "__syft_dir__"):
                # here we filter all autocomplete results to only contain those
                # defined in __syft_dir__, however the original autocomplete prefixes
                # have the full path, while __syft_dir__ only defines the attr
                attrs = set(obj.__syft_dir__())
                new_res = []
                for r in res:
                    splitted = r.split(".")
                    if len(splitted) > 1:
                        attr_name = splitted[-1]
                        if attr_name in attrs:
                            new_res.append(r)
                return new_res
            else:
                return res

        ipython.Completer.attr_matches = MethodType(
            patched_attr_matches, ipython.Completer
        )
    except Exception:
        print("Failed to patch syft autocompletion for __syft_dir__")


def patch_ipython() -> None:
    _patch_ipython_sanitization()
    _patch_ipython_autocompletion()
