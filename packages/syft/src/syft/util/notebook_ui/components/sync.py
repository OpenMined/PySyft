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
from ..icons import Icon
from ..notebook_addons import CSS_CODE
from .base import HTMLComponentBase


class CopyIDButton(HTMLComponentBase):
    __canonical_name__ = "CopyButton"
    __version__ = SYFT_OBJECT_VERSION_1
    copy_text: str
    max_width: int = 50

    def to_html(self) -> str:
        copy_js = f"event.stopPropagation(); navigator.clipboard.writeText('{self.copy_text}');"
        button_html = f"""
        <div class="copy-container" onclick="{copy_js}">
            <span class="copy-text-display"
                  style="max-width: {self.max_width}px;">
                #{self.copy_text}
            </span>{Icon.COPY.svg}
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
            return self.object.user_code_name
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
        <style>{style}</style>
        {first_line_html}
        {second_line_html}
        {third_line_html}
        <div style='height: 16px;'></div>
        """

        return header_html
