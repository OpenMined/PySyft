# stdlib
import datetime
from typing import Any

# third party
from pydantic import model_validator

# relative
from ....client.sync_decision import SyncDirection
from ....service.code.user_code import UserCode
from ....service.job.job_stash import Job
from ....service.request.request import Request
from ....service.user.user import UserView
from ....types.datetime import DateTime
from ....types.datetime import format_timedelta_human_readable
from ....types.errors import SyftException
from ....types.syft_object import SYFT_OBJECT_VERSION_1
from ....types.syft_object import SyftObject
from ..icons import Icon
from ..styles import CSS_CODE
from .base import HTMLComponentBase

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


class CopyButton(HTMLComponentBase):
    __canonical_name__ = "CopyButton"
    __version__ = SYFT_OBJECT_VERSION_1
    copy_text: str
    max_width: int = 50

    def format_copy_text(self, copy_text: str) -> str:
        return copy_text

    def to_html(self) -> str:
        copy_js = f"event.stopPropagation(); navigator.clipboard.writeText('{self.copy_text}');"
        text_formatted = self.format_copy_text(self.copy_text)
        button_html = f"""
        <style>{COPY_CSS}</style>
        <div class="copy-container" onclick="{copy_js}">
            <span class="copy-text-display" style="max-width: {self.max_width}px;">
                {text_formatted}
            </span>
            {Icon.COPY.svg}
        </div>
        """
        return button_html


class CopyIDButton(CopyButton):
    __canonical_name__ = "CopyIDButton"
    __version__ = SYFT_OBJECT_VERSION_1

    def format_copy_text(self, copy_text: str) -> str:
        return f"#{copy_text}"


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
            approval_decisions = list(code.status.status_dict.values())
            if len(approval_decisions) != 1:
                raise ValueError("Request code should have exactly one status")
            return approval_decisions[0].status.value
        return ""  # type: ignore

    def get_updated_by(self) -> str:
        # TODO replace with centralized SyftObject created/updated by attribute
        if isinstance(self.object, Request):
            email = self.object.requesting_user_email
            if email is not None:
                return f"Requested by {email}"

        user_view: UserView | None = None
        if isinstance(self.object, UserCode):
            try:
                user_view = self.object.user
            except SyftException:
                pass  # nosec

        if isinstance(user_view, UserView):
            return f"Created by {user_view.email}"
        return ""

    def get_updated_delta_str(self) -> str:
        # TODO replace with centralized SyftObject created/updated by attribute
        if isinstance(self.object, Job):
            # NOTE Job is not using DateTime for creation_time, so we need to handle it separately
            time_str = self.object.creation_time
            if time_str is not None:
                t = datetime.datetime.fromisoformat(time_str)
                delta = datetime.datetime.now(datetime.timezone.utc) - t
                return f"{format_timedelta_human_readable(delta)} ago"

        dt: DateTime | None = None
        if isinstance(self.object, Request):
            dt = self.object.request_time
        if isinstance(self.object, UserCode):
            dt = self.object.submit_time
        if dt is not None:
            delta = DateTime.now().timedelta(dt)
            delta_str = format_timedelta_human_readable(delta)
            return f"{delta_str} ago"

        return ""

    def to_html(self) -> str:
        type_html = TypeLabel(object=self.object).to_html()

        type_html = TypeLabel(object=self.object).to_html()
        description_html = MainDescription(object=self.object).to_html()
        copy_id_button = CopyIDButton(
            copy_text=str(self.object.id.id), max_width=60
        ).to_html()

        updated_delta_str = self.get_updated_delta_str()
        updated_by = self.get_updated_by()
        status_str = self.get_status_str()
        status_row = " • ".join(
            s for s in [status_str, updated_by, updated_delta_str] if s
        )
        summary_html = f"""
            <div style="display: flex; gap: 8px; justify-content: space-between; width: 100%; overflow: hidden; align-items: center;">
            <div style="display: flex; gap: 8px; justify-content: start; align-items: center;">
            {type_html} {description_html}
            </div>
            {copy_id_button}
            </div>
            <div style="display: table-row">
            <span class='syncstate-col-footer'>
            {status_row}
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
    __version__ = SYFT_OBJECT_VERSION_1
    message: str

    def to_html(self) -> str:
        full_message = f"{Icon.INFO.svg} {self.message}"
        return f"""
            <style>{ALERT_CSS}</style>
            <div class="syft-alert-container">
            <div class="syft-alert-info">
            {full_message}
            </div>
            </div>
            """


class Badge(HTMLComponentBase):
    __canonical_name__ = "Badge"
    __version__ = SYFT_OBJECT_VERSION_1
    value: str
    badge_class: str

    def to_html(self) -> str:
        value = str(self.value).upper()
        return f'<span class="badge {self.badge_class}">{value}</span>'


class Label(HTMLComponentBase):
    __canonical_name__ = "Label"
    __version__ = SYFT_OBJECT_VERSION_1
    value: str
    label_class: str

    def to_html(self) -> str:
        value = str(self.value).upper()
        return f'<span class="label {self.label_class}">{value}</span>'


class TypeLabel(Label):
    __canonical_name__ = "TypeLabel"
    __version__ = SYFT_OBJECT_VERSION_1
    object: SyftObject

    @model_validator(mode="before")
    @classmethod
    def validate_label(cls, data: dict) -> dict:
        obj = data["object"]
        data["label_class"] = cls.type_label_class(obj)
        data["value"] = type(obj).__name__.upper()
        return data

    @staticmethod
    def type_label_class(obj: Any) -> str:
        if isinstance(obj, UserCode):
            return "label-light-blue"
        elif isinstance(obj, Job):  # type: ignore
            return "label-light-blue"
        elif isinstance(obj, Request):  # type: ignore
            # TODO: handle other requests
            return "label-light-purple"
        return "label-light-blue"  # type: ignore


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

        type_html = TypeLabel(object=self.object).to_html()
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
        third_line_html = f"<span style='color: #5E5A72;'>This would sync <span style='color: #B8520A'>{num_diffs} changes </span> from <i>{source_side} Server</i> to <i>{target_side} Server</i></span>"  # noqa: E501

        header_html = f"""
        {style}
        {first_line_html}
        {second_line_html}
        {third_line_html}
        <div style='height: 16px;'></div>
        """

        return header_html
