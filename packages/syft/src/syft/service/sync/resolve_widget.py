from enum import Enum, auto
from enum import Enum, auto
from typing import Any
import ipywidgets as widgets
from IPython.display import display

import json

from IPython.display import Javascript, display
from ipywidgets import HTML, Button, HBox, Layout, VBox, widgets, AppLayout, Box
from syft.service.sync.diff_state import ObjectDiffBatch


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
    def from_object_diff_batch(cls, obj_diff_batch):
        return cls(
            item_type=obj_diff_batch.root_type,
            item_name="TODO: object description",
            item_id=str(obj_diff_batch.root_id),
            num_diffs=len(obj_diff_batch.get_dependencies(include_roots=True)),
            source_side="Low side",
            target_side="High side",
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

    def create_item_type_box(self, item_type):
        # TODO different bg for different types (levels?)
        style = (
            "background-color: #C2DEF0; "
            "border-radius: 4px; "
            "padding: 4px 6px; "
            "color: #373B7B;"
        )
        return HTML(
            value=f"<span style='{style}'>{item_type}</span>",
            layout=Layout(margin="0 5px 0 0"),
        )

    def create_name_id_label(self, item_name, item_id):
        item_id_short = item_id[:4] + "..." if len(item_id) > 4 else item_id
        return HTML(
            value=(
                f"<span style='margin-left: 5px; font-weight: bold; color: #373B7B;'>{item_name}</span> "
                f"<span style='margin-left: 5px; color: #B4B0BF;'>#{item_id_short}</span>"
            )
        )

    def create_widget(self):
        type_box = self.create_item_type_box(self.item_type)
        name_id_label = self.create_name_id_label(self.item_name, self.item_id)
        copy_button = self.copy_text_button(self.item_id)

        first_line = HTML(
            value="<span style='color: #B4B0BF;'>Syncing changes on</span>"
        )
        second_line = HBox(
            [type_box, name_id_label, copy_button], layout=Layout(align_items="center")
        )
        third_line = HTML(
            value=f"<span style='color: #5E5A72;'>This would sync <span style='color: #B8520A'>{self.num_diffs} changes </span> from <i>{self.source_side} Node</i> to <i>{self.target_side} Node</i></span>"
        )
        fourth_line = HTML(value=f"<div style='height: 16px;'></div>")
        header = VBox([first_line, second_line, third_line, fourth_line])
        return header


# TODO use ObjectDiff instead
class ObjectDiffWidget:
    def __init__(
        self,
        item_type: str,
        low_properties: list[str],
        high_properties: list[str],
        statuses: list[DiffStatus],
        is_main_widget: bool = False,
    ):
        self.item_type = item_type
        self.low_properties = low_properties
        self.high_properties = high_properties
        self.statuses = statuses
        self.share_private_data = False

        self.sync: bool = False

        self.is_main_widget = is_main_widget
        self.widget = self.build()

    @classmethod
    def from_diff(cls, diff, is_main_widget):
        return cls(
            item_type=diff.obj_type.__name__,
            low_properties=diff.repr_attr_dict("low"),
            high_properties=diff.repr_attr_dict("high"),
            statuses=diff.repr_attr_diffstatus_dict(),
            is_main_widget=is_main_widget,
        )

    @property
    def show_share_button(self):
        return self.item_type in ["SyftLog", "ActionObject"]

    @property
    def show_sync_button(self):
        return not self.is_main_widget

    @property
    def num_changes(self):
        return len([x for x in self.statuses.values() if x != DiffStatus.SAME])

    @property
    def title(self):
        return f"{self.item_type} ({self.num_changes} changes)"

    def set_and_disable_sync(self):
        self._sync_checkbox.disabled = True
        self._sync_checkbox.value = True

    def enable_sync(self):
        self._sync_checkbox.disabled = False

    def create_diff_html(self, title, properties, statuses, line_length=80):
        html_str = f"<div style='font-family: monospace; width: 100%;'>{title}<br>"
        html_str += "<div style='border-left: 1px solid #B4B0BF; padding-left: 10px;'>"

        for attr, val in properties.items():
            status = statuses[attr]
            val = val if val is not None else ""
            style = f"background-color: {background_colors[status]}; color: {colors[status]}; display: block; white-space: pre-wrap; margin-bottom: 5px;"
            html_str += f"<div style='{style}'>{attr}: {val}</div>"

        html_str += "</div></div>"

        return html_str

    def build(self):
        all_keys = list(self.low_properties.keys()) + list(self.high_properties.keys())
        low_properties = {}
        high_properties = {}
        for k in all_keys:
            low_properties[k] = self.low_properties.get(k, None)
            high_properties[k] = self.high_properties.get(k, None)
        
        html_low = self.create_diff_html("From", low_properties, self.statuses)
        html_high = self.create_diff_html("To", high_properties, self.statuses)

        diff_display_widget_old = widgets.HTML(
            value=html_low, layout=widgets.Layout(width="50%", overflow="auto")
        )
        diff_display_widget_new = widgets.HTML(
            value=html_high, layout=widgets.Layout(width="50%", overflow="auto")
        )
        content = widgets.HBox([diff_display_widget_old, diff_display_widget_new])

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

    def _on_sync_change(self, change):
        self.sync = change["new"]

    def _on_share_private_data_change(self, change):
        self.share_private_data = change["new"]


class ResolveWidget:
    def __init__(self, obj_diff_batch, button_callback):
        self.obj_diff_batch: ObjectDiffBatch = obj_diff_batch
        self.button_callback=button_callback
        self.widget = self.build()

    @property
    def batch_diff_widgets(self):
        dependents = self.obj_diff_batch.get_dependents(
            include_roots=False, include_batch_root=False
        )
        dependent_diff_widgets = [
            ObjectDiffWidget.from_diff(diff, is_main_widget=False)
            for diff in dependents
        ]
        return dependent_diff_widgets

    @property
    def dependent_batch_diff_widgets(self):
        dependencies = self.obj_diff_batch.get_dependencies(
            include_roots=True, include_batch_root=False
        )
        other_roots = [
            d for d in dependencies if d.object_id in self.obj_diff_batch.global_roots
        ]
        dependent_root_diff_widgets = [
            ObjectDiffWidget.from_diff(diff, is_main_widget=False)
            for diff in other_roots
        ]
        return dependent_root_diff_widgets

    @property
    def main_object_diff_widget(self):
        obj_diff_widget = ObjectDiffWidget.from_diff(
            self.obj_diff_batch.root_diff, is_main_widget=True
        )
        return obj_diff_widget

    def build(self):
        main_batch_items = widgets.Accordion(
            children=[d.widget for d in self.batch_diff_widgets],
            titles=[d.title for d in self.batch_diff_widgets],
        )
        dependency_items = widgets.Accordion(
            children=[d.widget for d in self.dependent_batch_diff_widgets],
            titles=[d.title for d in self.dependent_batch_diff_widgets],
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

    def sync_button(self):
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

    def spacer(self, height):
        return widgets.HTML(f"<div style='height: {height}px'></div>")

    def separator(self):
        return widgets.HTML(
            value='<div style="text-align: center; margin: 10px 0; border: 1px dashed #B4B0BF;"></div>',
            layout=Layout(width="100%"),
        )

    def build_header(self):
        return HeaderWidget(
            item_type="Code",
            item_name="compute_mean",
            item_id="12345678",
            num_diffs=2,
            source_side="Low",
            target_side="High",
        )
