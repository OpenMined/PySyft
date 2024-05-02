# relative
from ...util.notebook_ui.styles import CSS_CODE
from ...util.notebook_ui.styles import JS_DOWNLOAD_FONTS

type_html = """
            <div class="label label-light-blue"
            style="display: flex; align-items:center; justify-content: center; width: 34px; height:21px; radius:4px;
                padding: 2px, 6px, 2px, 6px">
            <span style="font-family: DejaVu Sans Mono, sans-serif;
                font-size: 12px; font-weight: 400; line-height:16.8px">
            ${job_type}</span>
            </div>
"""

header_line_html = (
    """
    <div style="height:16px;"></div>
    <div style="gap: 12px; height: 20 px; font-family: DejaVu Sans Mono, sans-serif; font-size: 14px; font-weight: 400;
        line-height:16.8px; color: #4392C5">${api_header}</div>
    <div style="height:16px;"></div>
    <div style="display: flex; gap: 12px; justify-content: start; width: 100%; overflow:
        hidden; align-items: center;">
        <div style="display: flex; gap: 12px; justify-content: start; align-items: center;
            border: 0px, 0px, 2px, 0px; padding: 0px, 0px, 16px, 0px">"""
    + type_html
    + """
        <span class='jobs-title'>${user_code_name}</span>
        </div>
        ${button_html}
    </div>
    <div style="height:16px;"></div>
"""
)  # noqa: E501

attrs_html = """<div style="display: table-row; padding: 0px, 0px, 12px, 0px; gap:8px">
    <div style="margin-top: 6px; margin-bottom: 6px;">
    <span style="font-weight: 700; line-weight: 19.6px; font-size: 14px; font: 'Open Sans'">UserCode:</span>
        ${user_code_name}
    </div>
    <div style="margin-top: 6px; margin-bottom: 6px;">
    <span style="font-weight: 700; line-weight: 19.6px; font-size: 14px; font: 'Open Sans'">Status:</span>
        ${status}
    </div>
    <div style="margin-top: 6px; margin-bottom: 6px;">
        <span style="font-weight: 700; line-weight: 19.6px; font-size: 14px; font: 'Open Sans'">
            Started At:</span>
        ${creation_time} by ${user_repr}
    </div>
    <div style="margin-top: 6px; margin-bottom: 6px;">
        <span style="font-weight: 700; line-weight: 19.6px; font-size: 14px; font: 'Open Sans'">
        Updated At:</span>
        ${updated_at}
    </div>
    ${worker_attr}
    <div style="margin-top: 6px; margin-bottom: 6px;">
    <span style="font-weight: 700; line-weight: 19.6px; font-size: 14px; font: 'Open Sans'">Subjobs:</span>
        ${no_subjobs}
    </div>
</div>
<div style="height:16px;"></div>
"""

logs_html = """
<style>
pre {
    counter-reset: line -1;
    background-color: transparent;
    color: black;
}

pre code {
    display: block;
    counter-increment: line;
    background-color: transparent;
    color: black;
    font-family: DejaVu Sans Mono, sans-serif;
    font-size: 12px;
    font-weight: 400;
    line-height: 16.8px;
}

pre code::before {
    content: counter(line);
    display: inline-block;
    width: 2em;
    padding-right: 1.5em;
    margin-right: 1.5em;
    text-align: right;
}

pre code:first-of-type::before {
    content: "#";
    font-weight: bold;
}

.logsTab {
    color: #000000;
    background: #F4F3F6;
    border-color: #CFCDD6;
    border-width: 0.5px;
    border-style: solid;
    padding: 24px;
    gap: 8px;
    margin-top: 24px;
    display: none;
    align-items: left;

}

</style>

<div id="${logs_tab_id}" class="tab-${identifier} logsTab">
    <pre>
        ${logs_lines_html}
    </pre>
</div>
"""

# TODO: add style change for selected tab
onclick_html = """<script>
function onClick_${identifier}(evt, tabname) {
    existing_tabs = document.getElementsByClassName("tab-${identifier}");
    for (i = 0; i < existing_tabs.length; i++) {
        existing_tabs[i].style.display = "none";
    }
    tablinks = document.getElementsByClassName("tablink-${identifier}");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
    tablinks = document.getElementsByClassName("tablink-border-${identifier}");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active-border", "");
    }
    document.getElementById(tabname).style.display = "block";
    evt.currentTarget.className += " active";
    evt.currentTarget.parentNode.className += " active-border";
}
</script>
"""


tabs_html = """
    <div style="margin-top: 8px; padding: 8px, 0px, 8px, 0px; gap: 16px">
    <div style="display: flex;">
        <div class="tablink-border-${identifier} log-tab-header active-border">
            <a  onclick="onClick_${identifier}(event, '${result_tab_id}')" class='tablink-${identifier} active'>
                Result
            </a>
        </div>
        <div class="tablink-border-${identifier} log-tab-header">
            <a onclick="onClick_${identifier}(event, '${logs_tab_id}')" class='tablink-${identifier}'>Logs</a>
        </div>
    </div>
</div>
"""

result_html = """<div id="${result_tab_id}" class="tab-${identifier}" style="background: #F4F3F6;
    border-color: #CFCDD6; border-width: 0.5px; border-style: solid; padding: 24px; gap: 8px; margin-top: 24px">
    <div style="font-size: 12px; font-weight: 400; font: DejaVu Sans Mono, sans-serif; line-height: 16.8px">
        ${result}
    </div>
</div>
"""

job_repr_template = f"""
<!-- Start job_repr_template -->
<div class="syft-widget">

<!-- Start JS_DOWNLOAD_FONTS -->
{JS_DOWNLOAD_FONTS}
<!-- End JS_DOWNLOAD_FONTS -->

<!-- Start CSS_CODE -->
{CSS_CODE}
<!-- End CSS_CODE -->

<!-- Start header_line_html -->
{header_line_html}
<!-- End header_line_html -->

<!-- Start attrs_html -->
{attrs_html}
<!-- End attrs_html -->

<!-- Start tabs_html -->
{tabs_html}
<!-- End tabs_html -->

<!-- Start result_html -->
{result_html}
<!-- End result_html -->

<!-- Start logs_html -->
{logs_html}
<!-- End logs_html -->

<!-- Start onclick_html -->
{onclick_html}
<!-- End onclick_html -->

<div style='height: 16px;'></div>
</div>
<!-- End job_repr_template -->
"""
