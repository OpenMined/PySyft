# stdlib
from typing import Any

# third party
from pydantic import model_validator

# relative
from ....client.sync_decision import SyncDirection
from ....service.code.user_code import UserCode
from ....service.job.job_stash import Job
from ....service.request.request import Request
from ....types.syft_object import SYFT_OBJECT_VERSION_1
from ....types.syft_object import SyftObject
from ..notebook_addons import CSS_CODE
from .base import HTMLComponentBase

COPY_ICON = (
    '<svg width="13" height="13" viewBox="0 0 13 13" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M12 0.5H4C3.86739 0.5 3.74021 0.552679 3.64645 0.646447C3.55268 0.740215 3.5 0.867392 '
    "3.5 1V3.5H1C0.867392 3.5 0.740215 3.55268 0.646447 3.64645C0.552679 3.74021 0.5 3.86739 0.5 "
    "4V12C0.5 12.1326 0.552679 12.2598 0.646447 12.3536C0.740215 12.4473 0.867392 12.5 1 12.5H9C9.13261 "
    "12.5 9.25979 12.4473 9.35355 12.3536C9.44732 12.2598 9.5 12.1326 9.5 12V9.5H12C12.1326 9.5 12.2598 "
    "9.44732 12.3536 9.35355C12.4473 9.25979 12.5 9.13261 12.5 9V1C12.5 0.867392 12.4473 0.740215 12.3536 "
    "0.646447C12.2598 0.552679 12.1326 0.5 12 0.5ZM8.5 11.5H1.5V4.5H8.5V11.5ZM11.5 8.5H9.5V4C9.5 3.86739 "
    '9.44732 3.74021 9.35355 3.64645C9.25979 3.55268 9.13261 3.5 9 3.5H4.5V1.5H11.5V8.5Z" fill="#B4B0BF"/>'
    "</svg>"
)

INFO_ICON = """<svg width="14" height="14" viewBox="0 0 14 14" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M7 1.3125C5.87512 1.3125 4.7755 1.64607 3.8402 2.27102C2.90489 2.89597 2.17591 3.78423 1.74544 4.82349C1.31496 5.86274 1.20233 7.00631 1.42179 8.10958C1.64124 9.21284 2.18292 10.2263 2.97833 11.0217C3.77374 11.8171 4.78716 12.3588 5.89043 12.5782C6.99369 12.7977 8.13726 12.685 9.17651 12.2546C10.2158 11.8241 11.104 11.0951 11.729 10.1598C12.3539 9.2245 12.6875 8.12488 12.6875 7C12.6859 5.49207 12.0862 4.04636 11.0199 2.98009C9.95365 1.91382 8.50793 1.31409 7 1.3125ZM7 11.8125C6.04818 11.8125 5.11773 11.5303 4.32632 11.0014C3.53491 10.4726 2.91808 9.72103 2.55383 8.84166C2.18959 7.9623 2.09428 6.99466 2.27997 6.06113C2.46566 5.12759 2.92401 4.27009 3.59705 3.59705C4.27009 2.92401 5.1276 2.46566 6.06113 2.27997C6.99466 2.09428 7.9623 2.18958 8.84167 2.55383C9.72104 2.91808 10.4726 3.53491 11.0014 4.32632C11.5303 5.11773 11.8125 6.04818 11.8125 7C11.8111 8.27591 11.3036 9.49915 10.4014 10.4014C9.49915 11.3036 8.27591 11.8111 7 11.8125ZM7.875 9.625C7.875 9.74103 7.82891 9.85231 7.74686 9.93436C7.66481 10.0164 7.55353 10.0625 7.4375 10.0625C7.20544 10.0625 6.98288 9.97031 6.81878 9.80622C6.65469 9.64212 6.5625 9.41956 6.5625 9.1875V7C6.44647 7 6.33519 6.95391 6.25314 6.87186C6.1711 6.78981 6.125 6.67853 6.125 6.5625C6.125 6.44647 6.1711 6.33519 6.25314 6.25314C6.33519 6.17109 6.44647 6.125 6.5625 6.125C6.79457 6.125 7.01713 6.21719 7.18122 6.38128C7.34531 6.54538 7.4375 6.76794 7.4375 7V9.1875C7.55353 9.1875 7.66481 9.23359 7.74686 9.31564C7.82891 9.39769 7.875 9.50897 7.875 9.625ZM6.125 4.59375C6.125 4.46396 6.16349 4.33708 6.2356 4.22916C6.30771 4.12124 6.4102 4.03712 6.53012 3.98745C6.65003 3.93778 6.78198 3.92479 6.90928 3.95011C7.03658 3.97543 7.15351 4.03793 7.24529 4.12971C7.33707 4.22149 7.39957 4.33842 7.42489 4.46572C7.45021 4.59302 7.43722 4.72497 7.38755 4.84489C7.33788 4.9648 7.25377 5.06729 7.14585 5.1394C7.03793 5.21151 6.91105 5.25 6.78125 5.25C6.6072 5.25 6.44028 5.18086 6.31721 5.05779C6.19414 4.93472 6.125 4.7678 6.125 4.59375Z" fill="#1F567A"/>
</svg>
"""  # noqa: E501

COPY_CSS = """
.copy-container {
  cursor: pointer;
  border-radius: 3px;
  padding: 0px 3px;
  display: inline-block;
  transition: background-color 0.3s;
  user-select: none;
  color: #B4B0BF;
  overflow: hidden;
  white-space: nowrap;
  vertical-align: middle;
}

.copy-container:hover {
  background-color: #f5f5f5;
}

.copy-container:active {
  background-color: #ebebeb;
}

.copy-text-display {
  display: inline-block;
  max-width: 50px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  vertical-align: bottom;
}
"""


class CopyIDButton(HTMLComponentBase):
    __canonical_name__ = "CopyButton"
    __version__ = SYFT_OBJECT_VERSION_1
    copy_text: str
    max_width: int = 50

    def to_html(self) -> str:
        copy_js = f"event.stopPropagation(); navigator.clipboard.writeText('{self.copy_text}');"
        button_html = f"""
        <style>{COPY_CSS}</style>
        <div class="copy-container" onclick="{copy_js}">
            <span class="copy-text-display" style="max-width: {self.max_width}px;">#{self.copy_text}</span>{COPY_ICON}
        </div>
        """
        return button_html


class SyncTableObject(HTMLComponentBase):
    __canonical_name__ = "SyncTableObject"
    __version__ = SYFT_OBJECT_VERSION_1

    object: SyftObject

    def get_status_str(self) -> str:
        if isinstance(self.object, UserCode):
            return ""
        elif isinstance(self.object, Job):  # type: ignore
            return f"Status: {self.object.status.value}"
        elif isinstance(self.object, Request):
            code = self.object.code
            statusses = list(code.status.status_dict.values())
            if len(statusses) != 1:
                raise ValueError("Request code should have exactly one status")
            status_tuple = statusses[0]
            status, _ = status_tuple
            return status.value
        return ""  # type: ignore

    def to_html(self) -> str:
        type_html = Badge(object=self.object).to_html()

        type_html = Badge(object=self.object).to_html()
        description_html = MainDescription(object=self.object).to_html()
        copy_id_button = CopyIDButton(
            copy_text=str(self.object.id.id), max_width=60
        ).to_html()

        updated_delta_str = "29m ago"
        updated_by = "john@doe.org"
        status_str = self.get_status_str()
        status_seperator = " • " if len(status_str) else ""
        summary_html = f"""
            <div style="display: flex; gap: 8px; justify-content: space-between; width: 100%; overflow: hidden; align-items: center;">
            <div style="display: flex; gap: 8px; justify-content: start; align-items: center;">
            {type_html} {description_html}
            </div>
            {copy_id_button}
            </div>
            <div style="display: table-row">
            <span class='syncstate-col-footer'>
            {status_str}{status_seperator}Updated by {updated_by} {updated_delta_str}
            </span>
            </div>
        """  # noqa: E501
        summary_html = summary_html.replace("\n", "").replace("    ", "")
        return summary_html


ALERT_CSS = """
.syft-alert-container {
    padding: 4px;
    display: flex;
    justify-content: center;
}

.syft-alert-info {
    display: flex;
    align-items: center;
    width: 100%;
    padding: 8px 10px;
    gap: 8px;
    border-radius: 4px;
    background: #C2DEF0;
    color: #1F567A;
    line-height: 1.4;
    font-size: 12px;
    font-family: 'Open Sans';
}
"""


class Alert(HTMLComponentBase):
    __canonical_name__ = "Alert"
    __version__ = SYFT_OBJECT_VERSION_1  # Ensure the version constant is correctly defined elsewhere
    message: str

    def to_html(self) -> str:
        full_message = f"{INFO_ICON} {self.message}"
        return f"""
            <style>{ALERT_CSS}</style>
            <div class="syft-alert-container">
            <div class="syft-alert-info">
            {full_message}
            </div>
            </div>
            """


class Badge(HTMLComponentBase):
    __canonical_name__ = "CopyButton"
    __version__ = SYFT_OBJECT_VERSION_1
    object: SyftObject

    def type_badge_class(self) -> str:
        if isinstance(self.object, UserCode):
            return "label-light-blue"
        elif isinstance(self.object, Job):  # type: ignore
            return "label-light-blue"
        elif isinstance(self.object, Request):  # type: ignore
            # TODO: handle other requests
            return "label-light-purple"
        return "label-light-blue"  # type: ignore

    def to_html(self) -> str:
        badge_class = self.type_badge_class()
        object_type = type(self.object).__name__.upper()
        return f'<div class="label {badge_class}">{object_type}</div>'


class MainDescription(HTMLComponentBase):
    __canonical_name__ = "CopyButton"
    __version__ = SYFT_OBJECT_VERSION_1
    object: SyftObject

    def main_object_description_str(self) -> str:
        if isinstance(self.object, UserCode):
            return self.object.service_func_name
        elif isinstance(self.object, Job):  # type: ignore
            return self.object.user_code_name or ""
        elif isinstance(self.object, Request):  # type: ignore
            # TODO: handle other requests
            return f"Execute {self.object.code.service_func_name}"
        # SyftLog
        # ExecutionOutput
        # ActionObject
        # UserCodeStatusCollection

        return ""  # type: ignore

    def to_html(self) -> str:
        return f'<span class="syncstate-description">{self.main_object_description_str()}</span>'


class SyncWidgetHeader(SyncTableObject):
    diff_batch: Any

    @model_validator(mode="before")
    @classmethod
    def add_object(cls, values: dict) -> dict:
        if "diff_batch" not in values:
            raise ValueError("diff_batch is required")
        diff_batch = values["diff_batch"]
        values["object"] = diff_batch.root_diff.non_empty_object
        return values

    def to_html(self) -> str:
        # CSS Styles
        style = CSS_CODE

        first_line_html = "<span style='color: #B4B0BF;'>Syncing changes on</span>"

        type_html = Badge(object=self.object).to_html()
        description_html = MainDescription(object=self.object).to_html()
        copy_id_button = CopyIDButton(
            copy_text=str(self.object.id.id), max_width=60
        ).to_html()

        second_line_html = f"""
            <div class="widget-header2">
            <div class="widget-header2-2">
            {type_html} {description_html}
            </div>
            {copy_id_button}
            </div>
        """  # noqa: E501

        num_diffs = len(self.diff_batch.get_dependencies(include_roots=True))
        if self.diff_batch.sync_direction == SyncDirection.HIGH_TO_LOW:
            source_side = "High"
            target_side = "Low"
        else:
            source_side = "Low"
            target_side = "High"

        # Third line HTML
        third_line_html = f"<span style='color: #5E5A72;'>This would sync <span style='color: #B8520A'>{num_diffs} changes </span> from <i>{source_side} Node</i> to <i>{target_side} Node</i></span>"  # noqa: E501

        header_html = f"""
        {style}
        {first_line_html}
        {second_line_html}
        {third_line_html}
        <div style='height: 16px;'></div>
        """

        return header_html
