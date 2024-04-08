# stdlib
from enum import Enum
from enum import auto
import html
import json
from typing import Any

# third party
from IPython.display import Javascript
from IPython.display import display
import ipywidgets as widgets
from ipywidgets import Button
from ipywidgets import HBox
from ipywidgets import HTML
from ipywidgets import Layout
from ipywidgets import VBox
from typing_extensions import Self

# relative
from ...client.api import APIRegistry
from ...client.sync_decision import SyncDecision
from ...client.sync_decision import SyncDirection
from ...node.credentials import SyftVerifyKey
from ...types.uid import UID
from ...util.notebook_ui.notebook_addons import CSS_CODE
from ..action.action_object import ActionObject
from ..log.log import SyftLog
from ..response import SyftError
from ..response import SyftSuccess
from .diff_state import ObjectDiff
from .diff_state import ObjectDiffBatch
from .diff_state import ResolvedSyncState
from .diff_state import SyncInstruction
from .sync_state import SyncView

# Standard div Jupyter Lab uses for notebook outputs
# This is needed to use alert styles from SyftSuccess and SyftError
NOTEBOOK_OUTPUT_DIV = """
<div class="lm-Widget
            jp-RenderedHTMLCommon
            jp-RenderedHTML
            jp-mod-trusted
            jp-OutputArea-output"
     data-mime-type="text/html">
    {content}<br>
</div>
"""


class DiffStatus(Enum):
    NEW = auto()
    SAME = auto()
    MODIFIED = auto()
    DELETED = auto()


background_colors = {
    DiffStatus.NEW: "#D5F1D5;",
    DiffStatus.SAME: "transparent",
    DiffStatus.MODIFIED: "#FEE9CD;",
    DiffStatus.DELETED: "#ffdddd;",
}


colors = {
    DiffStatus.NEW: "#256B24;",
    DiffStatus.SAME: "#353243",
    DiffStatus.MODIFIED: "#B8520A;",
    DiffStatus.DELETED: "#353243",
}


class HeaderWidget:
    def __init__(
        self,
        item_type: str,
        item_name: str,
        item_id: str,
        num_diffs: int,
        source_side: str,
        target_side: str,
    ):
        self.item_type = item_type
        self.item_name = item_name
        self.item_id = item_id
        self.num_diffs = num_diffs
        self.source_side = source_side
        self.target_side = target_side
        self.widget = self.create_widget()

    @classmethod
    def from_object_diff_batch(cls, obj_diff_batch: ObjectDiffBatch) -> Self:
        """
        (
            diff=self.obj_diff_batch.root_diff,
            item_type=self.obj_diff_batch.root_type_name,
            item_name="compute_mean",
            item_id=self.obj_diff_batch.root_id,
            num_diffs=2,
            source_side="Low",
            target_side="High",
        )

        """
        if obj_diff_batch.sync_direction == SyncDirection.LOW_TO_HIGH:
            source_side = "Low side"
            target_side = "High side"
        else:
            source_side = "High side"
            target_side = "Low side"

        root_diff = obj_diff_batch.root_diff
        root_obj = (
            root_diff.low_obj if root_diff.low_obj is not None else root_diff.high_obj
        )
        obj_view = SyncView(object=root_obj)
        return cls(
            item_type=obj_view.object_type_name,
            item_name=obj_view.main_object_description_str(),
            item_id=str(root_obj.id.id),  # type: ignore
            num_diffs=len(obj_diff_batch.get_dependencies(include_roots=True)),
            source_side=source_side,
            target_side=target_side,
        )

    def copy_text_button(self, text: str) -> widgets.Widget:
        button = widgets.Button(
            icon="clone",
            layout=widgets.Layout(width="25px", height="25px", margin="0", padding="0"),
        )
        output = widgets.Output(layout=widgets.Layout(display="none"))
        copy_js = Javascript(f"navigator.clipboard.writeText({json.dumps(text)})")

        def on_click(_: widgets.Button) -> None:
            output.clear_output()
            with output:
                display(copy_js)

        button.on_click(on_click)

        return widgets.Box(
            (button, output),
            layout=widgets.Layout(display="flex", align_items="center"),
        )

    def create_item_type_label(self, item_type: str) -> HTML:
        # TODO different bg for different types (levels?)
        style = (
            "background-color: #C2DEF0; "
            "border-radius: 4px; "
            "padding: 4px 6px; "
            "color: #373B7B;"
        )
        return HTML(
            value=f"<span style='{style}'>{item_type.upper()}</span>",
            layout=Layout(margin="0 5px 0 0"),
        )

    def create_name_id_label(self, item_name: str, item_id: str) -> HTML:
        item_id_short = item_id[:4] + "..." if len(item_id) > 4 else item_id
        return HTML(
            value=(
                f"<span style='margin-left: 5px; font-weight: bold; color: #373B7B;'>{item_name}</span> "
                f"<span style='margin-left: 5px; color: #B4B0BF;'>#{item_id_short}</span>"
            )
        )

    def create_widget(self) -> VBox:
        type_box = self.create_item_type_label(self.item_type)
        name_id_label = self.create_name_id_label(self.item_name, self.item_id)
        copy_button = self.copy_text_button(self.item_id)

        first_line = HTML(
            value="<span style='color: #B4B0BF;'>Syncing changes on</span>"
        )
        second_line = HBox(
            [type_box, name_id_label, copy_button], layout=Layout(align_items="center")
        )
        third_line = HTML(
            value=f"<span style='color: #5E5A72;'>This would sync <span style='color: #B8520A'>{self.num_diffs} changes </span> from <i>{self.source_side} Node</i> to <i>{self.target_side} Node</i></span>"  # noqa: E501
        )
        fourth_line = HTML(value="<div style='height: 16px;'></div>")
        header = VBox([first_line, second_line, third_line, fourth_line])
        return header


# TODO use ObjectDiff instead
class ObjectDiffWidget:
    def __init__(
        self,
        diff: ObjectDiff,
        low_properties: list[str],
        high_properties: list[str],
        statuses: list[DiffStatus],
        direction: SyncDirection,
        is_main_widget: bool = False,
    ):
        self.low_properties = low_properties
        self.high_properties = high_properties
        self.statuses = statuses
        self.direction = direction
        self.share_private_data = False
        self.diff: ObjectDiff = diff

        self.sync: bool = False

        self.is_main_widget = is_main_widget
        self.widget = self.build()
        self.set_and_disable_sync()

    @property
    def mockify(self) -> bool:
        if self.show_share_button and not self.share_private_data:
            return True
        else:
            return False

    @classmethod
    def from_diff(
        cls, diff: ObjectDiff, direction: SyncDirection, is_main_widget: bool
    ) -> Self:
        return cls(
            low_properties=diff.repr_attr_dict("low"),
            high_properties=diff.repr_attr_dict("high"),
            statuses=diff.repr_attr_diffstatus_dict(),
            is_main_widget=is_main_widget,
            diff=diff,
            direction=direction,
        )

    @property
    def show_share_button(self) -> bool:
        return isinstance(self.diff.non_empty_object, SyftLog | ActionObject)

    @property
    def show_sync_button(self) -> bool:
        return not self.is_main_widget

    @property
    def num_changes(self) -> int:
        return len([x for x in self.statuses.values() if x != DiffStatus.SAME])

    @property
    def title(self) -> str:
        return f"{self.diff.object_type} ({self.num_changes} changes)"

    def set_and_disable_sync(self) -> None:
        if self.show_sync_button:
            self._sync_checkbox.disabled = True
            self._sync_checkbox.value = True

    def enable_sync(self) -> None:
        if self.show_sync_button:
            self._sync_checkbox.disabled = False

    def set_share_private_data(self) -> None:
        if self.show_share_button:
            self._share_private_checkbox.value = True

    def create_diff_html(
        self,
        title: str,
        properties: dict[str, str],
        statuses: dict[str, DiffStatus],
    ) -> str:
        html_str = f"<div style='width: 100%;'>{title}<br>"
        html_str += "<div style='font-family: monospace; border-left: 1px solid #B4B0BF; padding-left: 10px;'>"

        for attr, val in properties.items():
            status = statuses[attr]
            val = val if val is not None else ""
            style = f"background-color: {background_colors[status]}; color: {colors[status]}; display: block; white-space: pre-wrap; margin-bottom: 5px;"  # noqa: E501
            content = html.escape(f"{attr}: {val}")
            html_str += f"<div style='{style}'>{content}</div>"

        html_str += "</div></div>"

        return html_str

    def build(self) -> widgets.VBox:
        all_keys = list(self.low_properties.keys()) + list(self.high_properties.keys())
        low_properties = {}
        high_properties = {}
        for k in all_keys:
            low_properties[k] = self.low_properties.get(k, None)
            high_properties[k] = self.high_properties.get(k, None)

        if self.direction == SyncDirection.LOW_TO_HIGH:
            from_properties = low_properties
            to_properties = high_properties
            source_side = "Low side"
            target_side = "High side"
        else:
            from_properties = high_properties
            to_properties = low_properties
            source_side = "High side"
            target_side = "Low side"

        html_from = self.create_diff_html(
            f"From <i>{source_side}</i> (new values)", from_properties, self.statuses
        )
        html_to = self.create_diff_html(
            f"To <i>{target_side}</i> (old values)", to_properties, self.statuses
        )

        widget_from = widgets.HTML(
            value=html_from, layout=widgets.Layout(width="50%", overflow="auto")
        )
        widget_to = widgets.HTML(
            value=html_to, layout=widgets.Layout(width="50%", overflow="auto")
        )
        content = widgets.HBox([widget_from, widget_to])

        checkboxes = []

        if self.show_sync_button:
            self._sync_checkbox = widgets.Checkbox(
                value=self.sync,
                description="Sync",
            )
            self._sync_checkbox.observe(self._on_sync_change, "value")
            checkboxes.append(self._sync_checkbox)

        if self.show_share_button:
            self._share_private_checkbox = widgets.Checkbox(
                value=self.share_private_data,
                description="Share private data",
            )
            self._share_private_checkbox.observe(
                self._on_share_private_data_change, "value"
            )
            checkboxes = [self._share_private_checkbox] + checkboxes

        checkboxes_widget = widgets.HBox(checkboxes)
        kwargs = {}
        if self.is_main_widget:
            kwargs["layout"] = Layout(border="#353243 solid 0.5px", padding="16px")

        widget = widgets.VBox([checkboxes_widget, content], **kwargs)
        return widget

    def _on_sync_change(self, change: Any) -> None:
        self.sync = change["new"]

    def _on_share_private_data_change(self, change: Any) -> None:
        self.share_private_data = change["new"]


class ResolveWidget:
    def __init__(self, obj_diff_batch: ObjectDiffBatch):
        self.obj_diff_batch: ObjectDiffBatch = obj_diff_batch
        self.id2widget: dict[UID, ObjectDiffWidget] = {}
        self.main_widget = self.build()
        self.result_widget = VBox()  # Placeholder for SyftSuccess / SyftError
        self.widget = VBox(
            [self.build_css_widget(), self.main_widget, self.result_widget]
        )
        self.is_synced = False
        self.hide_result_widget()

    def build_css_widget(self) -> HTML:
        return widgets.HTML(value=CSS_CODE)

    def _repr_mimebundle_(self, **kwargs: dict) -> dict[str, str] | None:
        # from IPython.display import display
        return self.widget._repr_mimebundle_(**kwargs)

    def click_sync(self) -> SyftSuccess | SyftError:
        return self.button_callback()

    def click_share_all_private_data(self) -> None:
        for widget in self.id2widget.values():
            widget.set_share_private_data()

    def click_share_private_data(self, uid: UID | str) -> SyftError | SyftSuccess:
        if isinstance(uid, str):
            uid = UID(uid)
        if uid not in self.id2widget:
            return SyftError(message="Object not found in this widget")

        widget = self.id2widget[uid]
        widget.set_share_private_data()
        return SyftSuccess(message="Private data shared")

    def button_callback(self, *args: list, **kwargs: dict) -> SyftSuccess | SyftError:
        if self.is_synced:
            return SyftError(
                message="The changes in this widget have already been synced."
            )

        if self.obj_diff_batch.sync_direction == SyncDirection.LOW_TO_HIGH:
            # TODO: make dynamic
            decision = "low"
        else:
            decision = "high"

        # previously_ignored_batches = state.low_state.ignored_batches
        previously_ignored_batches: dict = {}
        # TODO: only add permissions for objects where we manually give permission
        # Maybe default read permission for some objects (high -> low)

        # TODO: UID
        resolved_state_low = ResolvedSyncState(node_uid=UID(), alias="low")
        resolved_state_high = ResolvedSyncState(node_uid=UID(), alias="high")

        batch_diff = self.obj_diff_batch
        if batch_diff.is_unchanged:
            # Hierarchy has no diffs
            return SyftSuccess(message="No changes to sync")

        if batch_diff.decision is not None:
            # handles ignores
            batch_decision = batch_diff.decision
        elif decision is not None:
            batch_decision = SyncDecision(decision)
        else:
            pass
            # batch_decision = get_user_input_for_resolve()

        batch_diff.decision = batch_decision

        # TODO: FIX
        other_batches: list = []
        # other_batches = [b for b in state.batches if b is not batch_diff]
        # relative
        from ...client.syncing import handle_ignore_skip

        handle_ignore_skip(batch_diff, batch_decision, other_batches)

        if batch_decision not in [SyncDecision.skip, SyncDecision.ignore]:
            sync_instructions = []
            for diff in batch_diff.get_dependents(include_roots=True):
                # while making widget: bind buttons to right state
                # share_state = {diff.object_id: False for diff in obj_diff_batch.get_dependents(include_roots=False)}

                # diff id -> shared (bool)

                # onclick checkbox: set widget state
                widget = self.id2widget[diff.object_id]
                sync = widget.sync

                if sync or widget.is_main_widget:
                    share_to_user: SyftVerifyKey | None = getattr(
                        self.obj_diff_batch.user_code_high, "user_verify_key", None
                    )
                    instruction = SyncInstruction.from_widget_state(
                        widget=widget,
                        sync_direction=self.obj_diff_batch.sync_direction,
                        decision=decision,
                        share_to_user=share_to_user,
                    )
                    sync_instructions.append(instruction)
        else:
            sync_instructions = []
            if batch_decision == SyncDecision.ignore:
                resolved_state_high.add_ignored(batch_diff)
                resolved_state_low.add_ignored(batch_diff)

        if (
            batch_diff.root_id in previously_ignored_batches
            and batch_diff.decision != SyncDecision.ignore
        ):
            resolved_state_high.add_unignored(batch_diff.root_id)
            resolved_state_low.add_unignored(batch_diff.root_id)

        print(f"Decision: Syncing {len(sync_instructions)} objects")

        for sync_instruction in sync_instructions:
            resolved_state_low.add_sync_instruction(sync_instruction)
            resolved_state_high.add_sync_instruction(sync_instruction)

        # TODO: ONLY WORKS FOR LOW TO HIGH
        # relative
        from ...client.domain_client import DomainClient

        api = APIRegistry.api_for(
            self.obj_diff_batch.target_node_uid, self.obj_diff_batch.target_verify_key
        )
        client = DomainClient(
            api=api,
            connection=api.connection,  # type: ignore
            credentials=api.signing_key,  # type: ignore
        )

        if self.obj_diff_batch.sync_direction is None:
            raise ValueError("no direction specified")
        if self.obj_diff_batch.sync_direction == SyncDirection.LOW_TO_HIGH:
            res = client.apply_state(resolved_state_high)
        else:
            res = client.apply_state(resolved_state_low)

        self.is_synced = True
        self.set_result_state(res)
        self.hide_main_widget()
        self.show_result_widget()
        return res

    @property
    def batch_diff_widgets(self) -> list[ObjectDiffWidget]:
        dependents = self.obj_diff_batch.get_dependents(
            include_roots=False, include_batch_root=False
        )
        dependent_diff_widgets = [
            ObjectDiffWidget.from_diff(
                diff,
                is_main_widget=False,
                direction=self.obj_diff_batch.sync_direction,
            )
            for diff in dependents
        ]
        return dependent_diff_widgets

    @property
    def dependent_batch_diff_widgets(self) -> list[ObjectDiffWidget]:
        dependencies = self.obj_diff_batch.get_dependencies(
            include_roots=True, include_batch_root=False
        )
        other_roots = [
            d for d in dependencies if d.object_id in self.obj_diff_batch.global_roots
        ]
        dependent_root_diff_widgets = [
            ObjectDiffWidget.from_diff(
                diff, is_main_widget=False, direction=self.obj_diff_batch.sync_direction
            )
            for diff in other_roots
        ]
        return dependent_root_diff_widgets

    @property
    def main_object_diff_widget(self) -> ObjectDiffWidget:
        obj_diff_widget = ObjectDiffWidget.from_diff(
            self.obj_diff_batch.root_diff,
            is_main_widget=True,
            direction=self.obj_diff_batch.sync_direction,
        )
        return obj_diff_widget

    def set_result_state(self, result: SyftSuccess | SyftError) -> None:
        result_html = result._repr_html_()
        # Wrap in div to match Jupyter Lab output styling
        result_html = NOTEBOOK_OUTPUT_DIV.format(content=result_html)
        self.result_widget.children = [widgets.HTML(value=result_html)]

    def hide_main_widget(self) -> None:
        self.main_widget.layout.display = "none"

    def show_main_widget(self) -> None:
        self.main_widget.layout.display = "block"

    def hide_result_widget(self) -> None:
        self.result_widget.layout.display = "none"

    def show_result_widget(self) -> None:
        self.result_widget.layout.display = "block"

    def build(self) -> VBox:
        self.id2widget = {}

        batch_diff_widgets = self.batch_diff_widgets
        dependent_batch_diff_widgets = self.dependent_batch_diff_widgets
        main_object_diff_widget = self.main_object_diff_widget

        self.id2widget[main_object_diff_widget.diff.object_id] = main_object_diff_widget

        for widget in batch_diff_widgets:
            self.id2widget[widget.diff.object_id] = widget

        for widget in dependent_batch_diff_widgets:
            self.id2widget[widget.diff.object_id] = widget

        main_batch_items = widgets.Accordion(
            children=[d.widget for d in batch_diff_widgets],
            titles=[d.title for d in batch_diff_widgets],
        )
        dependency_items = widgets.Accordion(
            children=[d.widget for d in dependent_batch_diff_widgets],
            titles=[d.title for d in dependent_batch_diff_widgets],
        )

        full_widget = widgets.VBox(
            [
                self.build_header().widget,
                self.main_object_diff_widget.widget,
                self.spacer(16),
                main_batch_items,
                self.separator(),
                dependency_items,
                self.spacer(16),
                self.sync_button(),
            ]
        )
        return full_widget

    def sync_button(self) -> Button:
        sync_button = Button(
            description="Sync Selected Changes",
            style={
                "text_color": "#464A91",
                "button_color": "transparent",
                "float": "right",
            },
            layout=Layout(border="#464A91 solid 1.5px", width="200px"),
        )
        sync_button.on_click(self.button_callback)
        return sync_button

    def spacer(self, height: int) -> widgets.HTML:
        return widgets.HTML(f"<div style='height: {height}px'></div>")

    def separator(self) -> widgets.HTML:
        return widgets.HTML(
            value='<div style="text-align: center; margin: 10px 0; border: 1px dashed #B4B0BF;"></div>',
            layout=Layout(width="100%"),
        )

    def build_header(self) -> HeaderWidget:
        return HeaderWidget.from_object_diff_batch(self.obj_diff_batch)
