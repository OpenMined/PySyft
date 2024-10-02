# stdlib
from collections.abc import Callable
from enum import Enum
from enum import auto
from functools import partial
import html
import secrets
from typing import Any
from uuid import uuid4

# third party
from IPython import display
import ipywidgets as widgets
from ipywidgets import Button
from ipywidgets import Checkbox
from ipywidgets import HBox
from ipywidgets import HTML
from ipywidgets import Layout
from ipywidgets import VBox

# relative
from ...client.sync_decision import SyncDecision
from ...client.sync_decision import SyncDirection
from ...types.errors import SyftException
from ...types.uid import UID
from ...util.notebook_ui.components.sync import Alert
from ...util.notebook_ui.components.sync import CopyIDButton
from ...util.notebook_ui.components.sync import MainDescription
from ...util.notebook_ui.components.sync import SyncWidgetHeader
from ...util.notebook_ui.components.sync import TypeLabel
from ...util.notebook_ui.components.tabulator_template import build_tabulator_table
from ...util.notebook_ui.components.tabulator_template import highlight_single_row
from ...util.notebook_ui.components.tabulator_template import update_table_cell
from ...util.notebook_ui.styles import CSS_CODE
from ..action.action_object import ActionObject
from ..api.api import TwinAPIEndpoint
from ..log.log import SyftLog
from ..response import SyftSuccess
from .diff_state import ObjectDiff
from .diff_state import ObjectDiffBatch
from .widget_output import Output

# Standard div Jupyter Lab uses for notebook outputs
# This is needed to use alert styles from SyftSuccess and SyftException
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


def create_diff_html(
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


# TODO move CSS/HTML/JS outside function


class MainObjectDiffWidget:
    def __init__(
        self,
        diff: ObjectDiff,
        direction: SyncDirection,
        with_box: bool = True,
        show_share_warning: bool = False,
        build_state: bool = True,
    ):
        build_state = build_state

        if build_state:
            self.low_properties = diff.repr_attr_dict("low")
            self.high_properties = diff.repr_attr_dict("high")
            self.statuses = diff.repr_attr_diffstatus_dict()
        else:
            self.low_properties = {}
            self.high_properties = {}
            self.statuses = {}

        self.direction = direction
        self.diff: ObjectDiff = diff
        self.with_box = with_box
        self.show_share_warning = show_share_warning
        self.sync = True
        self.is_main_widget: bool = True

        self.widget = self.build()

    def set_share_private_data(self) -> None:
        # No-op for main widget
        pass

    @property
    def mockify(self) -> bool:
        return not self.share_private_data

    @property
    def share_private_data(self) -> bool:
        # there are TwinAPIEndpoint.__private_sync_attr_mocks__
        return not isinstance(self.diff.non_empty_object, TwinAPIEndpoint)

    @property
    def warning_html(self) -> str:
        if isinstance(self.diff.non_empty_object, TwinAPIEndpoint):
            message = "Only the public function of a TwinAPI will be synced to the low-side server."
            return Alert(message=message).to_html()
        elif self.show_share_warning:
            message = (
                "By default only the object wrapper will be synced. "
                "If you would like to sync the real data please "
                'activate the "Sync Real Data" button above.'
            )
            return Alert(message=message).to_html()
        else:
            return ""

    def build(self) -> widgets.HBox:
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

        html_from = create_diff_html(
            f"From <i>{source_side}</i> (new values)", from_properties, self.statuses
        )
        html_to = create_diff_html(
            f"To <i>{target_side}</i> (old values)", to_properties, self.statuses
        )

        widget_from = widgets.HTML(
            value=html_from, layout=widgets.Layout(width="50%", overflow="auto")
        )
        widget_to = widgets.HTML(
            value=html_to, layout=widgets.Layout(width="50%", overflow="auto")
        )
        css_accordion = """
            <style>
            .diff-container {
                border: 0.5px solid #B4B0BF;
            }
            </style>
        """

        result = widgets.HBox([HTML(css_accordion), widget_from, widget_to])

        warning = self.warning_html
        if warning:
            result = VBox([widgets.HTML(warning), result])

        if self.with_box:
            result._dom_classes = result._dom_classes + ("diff-container",)

        return result


class CollapsableObjectDiffWidget:
    def __init__(
        self,
        diff: ObjectDiff,
        direction: SyncDirection,
        build_state: bool = True,
    ):
        self.direction = direction
        self.build_state = build_state
        self.share_private_data = False
        self.diff: ObjectDiff = diff
        self.sync: bool = False
        self.is_main_widget: bool = False
        self.has_private_data = isinstance(
            self.diff.non_empty_object, SyftLog | ActionObject | TwinAPIEndpoint
        )
        self.widget = self.build()
        self.set_and_disable_sync()

    @property
    def mockify(self) -> bool:
        if self.has_private_data and not self.share_private_data:
            return True
        else:
            return False

    @property
    def warning_html(self) -> str:
        if self.show_share_button:
            message = (
                "By default only the object wrapper will be synced. "
                "If you would like to sync the real log data please "
                "activate the “Real Data” button above."
            )
            return Alert(message=message).to_html()
        return ""

    @property
    def show_share_button(self) -> bool:
        return self.has_private_data

    @property
    def title(self) -> str:
        object = self.diff.non_empty_object
        if object is None:
            return "n/a"
        type_html = TypeLabel(object=object).to_html()
        description_html = MainDescription(object=object).to_html()
        copy_id_button = CopyIDButton(copy_text=str(object.id.id), max_width=60)

        second_line_html = f"""
            <div class="widget-header2">
            <div class="widget-header2-2">
            {type_html} {description_html}
            </div>
            {copy_id_button.to_html()}
            </div>
        """  # noqa: E501
        return second_line_html

    def set_and_disable_sync(self) -> None:
        self._sync_checkbox.disabled = True
        self._sync_checkbox.value = True

    def enable_sync(self) -> None:
        if self.show_sync_button:
            self._sync_checkbox.disabled = False

    def set_share_private_data(self) -> None:
        if self.show_share_button:
            self._share_private_checkbox.value = True

    def build(self) -> widgets.VBox:
        content = MainObjectDiffWidget(
            self.diff,
            self.direction,
            with_box=False,
            show_share_warning=self.show_share_button,
            build_state=self.build_state,
        ).widget

        accordion, share_private_checkbox, sync_checkbox = self.build_accordion(
            accordion_body=content,
            show_sync_checkbox=True,
            show_share_private_checkbox=self.show_share_button,
        )

        self._sync_checkbox = sync_checkbox
        self._sync_checkbox.observe(self._on_sync_change, "value")

        self._share_private_checkbox = share_private_checkbox
        self._share_private_checkbox.observe(
            self._on_share_private_data_change, "value"
        )

        return accordion

    def create_accordion_css(
        self, header_id: str, body_id: str, class_name: str
    ) -> str:
        css_accordion = f"""
            <style>
            .accordion {{
                padding: 0 10px;
                margin: 3px 0px;
            }}

            .body-hidden {{
                display: none;
            }}

            .body-visible {{
                display: flex;
            }}

            .{header_id}{{
                display: flex;
                align-items: center;
            }}


            .{class_name}-folded {{
                background: #F4F3F6;
                border: 0.5px solid #B4B0BF;
            }}
            .{class_name}-unfolded {{
                background: white;
                border: 0.5px solid #B4B0BF;
            }}
            </style>
        """
        return css_accordion

    def build_accordion(
        self,
        accordion_body: MainObjectDiffWidget,
        show_sync_checkbox: bool = True,
        show_share_private_checkbox: bool = True,
    ) -> VBox:
        uid = str(uuid4())
        body_id = f"accordion-body-{uid}"
        header_id = f"accordion-header-{uid}"
        class_name = f"accordion-{uid}"
        caret_id = f"caret-{uid}"

        toggle_hide_body_js = f"""
            var body = document.getElementsByClassName('{body_id}')[0];
            var caret = document.getElementById('{caret_id}');
            if (body.classList.contains('body-hidden')) {{
                var vbox = document.getElementsByClassName('{class_name}-folded')[0];
                body.classList.remove('body-hidden');
                body.classList.add('body-visible');
                vbox.classList.remove('{class_name}-folded');
                vbox.classList.add('{class_name}-unfolded');
                caret.classList.remove('fa-caret-right');
                caret.classList.add('fa-caret-down');
            }} else {{
                var vbox = document.getElementsByClassName('{class_name}-unfolded')[0];
                body.classList.remove('body-visible');
                body.classList.add('body-hidden');
                vbox.classList.remove('{class_name}-unfolded');
                vbox.classList.add('{class_name}-folded');
                caret.classList.remove('fa-caret-down');
                caret.classList.add('fa-caret-right');
            }}
        """
        caret = f'<i id="{caret_id}" class="fa fa-fw fa-caret-right"></i>'
        title_html = HTML(
            value=f"<div class='{header_id}' onclick=\"{toggle_hide_body_js}\" style='cursor: pointer; flex-grow: 1; user-select: none; '>{caret} {self.title}</div>",  # noqa: E501
            layout=Layout(flex="1"),
        )

        if isinstance(self.diff.non_empty_object, ActionObject):
            share_data_description = "Share real data and approve"
        else:
            share_data_description = "Share real data"
        share_private_data_checkbox = Checkbox(
            description=share_data_description,
            layout=Layout(width="auto", margin="0 2px 0 0"),
        )
        sync_checkbox = Checkbox(
            description="Sync", layout=Layout(width="auto", margin="0 2px 0 0")
        )

        checkboxes = []
        if show_share_private_checkbox:
            checkboxes.append(share_private_data_checkbox)
        if show_sync_checkbox:
            checkboxes.append(sync_checkbox)

        accordion_header = HBox(
            [title_html] + checkboxes,
            layout=Layout(width="100%", justify_content="space-between"),
        )

        accordion_body.add_class(body_id)
        accordion_body.add_class("body-hidden")

        style = HTML(value=self.create_accordion_css(header_id, body_id, class_name))

        accordion = VBox(
            [style, accordion_header, accordion_body],
            _dom_classes=(f"accordion-{uid}-folded", "accordion"),
        )
        return accordion, share_private_data_checkbox, sync_checkbox

    def _on_sync_change(self, change: Any) -> None:
        self.sync = change["new"]

    def _on_share_private_data_change(self, change: Any) -> None:
        self.share_private_data = change["new"]


class ResolveWidget:
    def __init__(
        self,
        obj_diff_batch: ObjectDiffBatch,
        on_sync_callback: Callable | None = None,
        on_ignore_callback: Callable | None = None,
        build_state: bool = True,
    ):
        self.build_state = build_state
        self.obj_diff_batch: ObjectDiffBatch = obj_diff_batch
        self.id2widget: dict[
            UID, CollapsableObjectDiffWidget | MainObjectDiffWidget
        ] = {}
        self.on_sync_callback = on_sync_callback
        self.on_ignore_callback = on_ignore_callback
        self.main_widget = self.build()
        self.result_widget = VBox()  # Placeholder for SyftSuccess / SyftException
        self.widget = VBox(
            [self.build_css_widget(), self.main_widget, self.result_widget]
        )
        self.is_synced = False
        self.hide_result_widget()

    def build_css_widget(self) -> HTML:
        return widgets.HTML(value=CSS_CODE)

    def _repr_mimebundle_(self, **kwargs: dict) -> dict[str, str] | None:
        return self.widget._repr_mimebundle_(**kwargs)

    def click_share_all_private_data(self) -> None:
        for widget in self.id2widget.values():
            widget.set_share_private_data()

    def click_share_private_data(self, uid: UID | str) -> SyftSuccess:
        if isinstance(uid, str):
            uid = UID(uid)
        if uid not in self.id2widget:
            raise SyftException(public_message="Object not found in this widget")

        widget = self.id2widget[uid]
        widget.set_share_private_data()
        return SyftSuccess(message="Private data shared")

    def get_share_private_data_state(self) -> dict[UID, bool]:
        return {
            uid: widget.share_private_data for uid, widget in self.id2widget.items()
        }

    def get_mockify_state(self) -> dict[UID, bool]:
        return {uid: widget.mockify for uid, widget in self.id2widget.items()}

    def ignore(self) -> None:
        # self.obj_diff_batch.ignore()
        self.obj_diff_batch.ignore()
        if self.on_ignore_callback:
            self.on_ignore_callback()

    def deny_and_ignore(self, reason: str) -> None:
        self.ignore()
        batch = self.obj_diff_batch
        # relative
        from ..request.request import Request

        assert batch.root_type == Request, "method can only be excecuted on requests"  # nosec: B101
        request = batch.root.low_obj
        assert request is not None  # nosec: B101
        request.deny(reason)

    def click_sync(self, *args: list, **kwargs: dict) -> SyftSuccess:
        # relative
        from ...client.syncing import handle_sync_batch

        if self.is_synced:
            raise SyftException(
                public_message="The changes in this widget have already been synced."
            )

        res = handle_sync_batch(
            obj_diff_batch=self.obj_diff_batch,
            share_private_data=self.get_share_private_data_state(),
            mockify=self.get_mockify_state(),
        )

        self.set_widget_result_state(res)
        if self.on_sync_callback:
            self.on_sync_callback()
        return res

    @property
    def batch_diff_widgets(self) -> list[CollapsableObjectDiffWidget]:
        dependents = self.obj_diff_batch.get_dependents(
            include_roots=False, include_batch_root=False
        )
        dependent_diff_widgets = [
            CollapsableObjectDiffWidget(
                diff,
                direction=self.obj_diff_batch.sync_direction,
                build_state=self.build_state,
            )
            for diff in dependents
        ]
        return dependent_diff_widgets

    @property
    def dependency_root_diff_widgets(self) -> list[CollapsableObjectDiffWidget]:
        dependencies = self.obj_diff_batch.get_dependencies(
            include_roots=True, include_batch_root=False
        )

        # we show these above the line
        dependents = self.obj_diff_batch.get_dependents(
            include_roots=False, include_batch_root=False
        )
        dependent_ids = [x.object_id for x in dependents]
        # we skip the ones we already show above the line in the widget
        context_diffs = [d for d in dependencies if d.object_id not in dependent_ids]
        widgets = [
            CollapsableObjectDiffWidget(
                diff,
                direction=self.obj_diff_batch.sync_direction,
                build_state=self.build_state,
            )
            for diff in context_diffs
        ]
        return widgets

    @property
    def main_object_diff_widget(self) -> MainObjectDiffWidget:
        obj_diff_widget = MainObjectDiffWidget(
            self.obj_diff_batch.root_diff,
            direction=self.obj_diff_batch.sync_direction,
            build_state=self.build_state,
        )
        return obj_diff_widget

    def set_widget_result_state(self, res: SyftSuccess) -> None:
        self.is_synced = True
        self.set_result_message(res)
        self.hide_main_widget()
        self.show_result_widget()

    def set_result_message(self, result: SyftSuccess) -> None:
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
        dependent_batch_diff_widgets = self.dependency_root_diff_widgets
        main_object_diff_widget = self.main_object_diff_widget

        self.id2widget[main_object_diff_widget.diff.object_id] = main_object_diff_widget

        for widget in batch_diff_widgets:
            self.id2widget[widget.diff.object_id] = widget

        for widget in dependent_batch_diff_widgets:
            self.id2widget[widget.diff.object_id] = widget

        # put a 4px spacer between each item
        main_batch_items = widgets.VBox(
            children=[d.widget for d in batch_diff_widgets],
        )

        dependency_items = widgets.VBox(
            children=[d.widget for d in dependent_batch_diff_widgets],
        )

        full_widget = widgets.VBox(
            [
                self.build_header(),
                self.main_object_diff_widget.widget,
                self.spacer(8),
                main_batch_items,
                self.separator(),
                dependency_items,
                self.spacer(8),
                self.sync_button(),
            ]
        )
        return full_widget

    def sync_button(self) -> Button:
        sync_button = Button(
            description="Apply Selected Changes",
            style={
                "text_color": "#464A91",
                "button_color": "transparent",
                "float": "right",
            },
            layout=Layout(border="#464A91 solid 1.5px", width="200px"),
        )
        sync_button.on_click(self.click_sync)
        return sync_button

    def spacer(self, height: int) -> widgets.HTML:
        return widgets.HTML(f"<div style='height: {height}px'></div>")

    def separator(self) -> widgets.HTML:
        return widgets.HTML(
            value='<div style="text-align: center; margin: 3px 0; border: 1px dashed #B4B0BF;"></div>',
            layout=Layout(width="100%"),
        )

    def build_header(self) -> HTML:
        header_html = SyncWidgetHeader(diff_batch=self.obj_diff_batch).to_html()
        return HTML(value=header_html)


class PaginationControl:
    def __init__(self, data: list, callback: Callable[[int], None]):
        self.data = data
        self.callback = callback
        self.current_index = 0
        self.index_label = widgets.Label(value=f"Index: {self.current_index}")

        self.first_button = widgets.Button(description="First")
        self.previous_button = widgets.Button(description="Previous")
        self.next_button = widgets.Button(description="Next")
        self.last_button = widgets.Button(description="Last")

        self.first_button.on_click(self.go_to_first)
        self.previous_button.on_click(self.go_to_previous)
        self.next_button.on_click(self.go_to_next)
        self.last_button.on_click(self.go_to_last)
        self.output = Output()

        self.buttons = widgets.HBox(
            [
                self.first_button,
                self.previous_button,
                self.next_button,
                self.last_button,
            ]
        )
        self.update_buttons()
        self.update_index_callback()

    def update_index_label(self) -> None:
        self.index_label.value = f"Current: {self.current_index}"

    def update_buttons(self) -> None:
        self.first_button.disabled = self.current_index == 0
        self.previous_button.disabled = self.current_index == 0
        self.next_button.disabled = self.current_index == len(self.data) - 1
        self.last_button.disabled = self.current_index == len(self.data) - 1

    def go_to_first(self, b: Button | None) -> None:
        self.current_index = 0
        self.update_index_callback()

    def go_to_previous(self, b: Button | None) -> None:
        if self.current_index > 0:
            self.current_index -= 1
        self.update_index_callback()

    def go_to_next(self, b: Button | None) -> None:
        if self.current_index < len(self.data) - 1:
            self.current_index += 1
        self.update_index_callback()

    def go_to_last(self, b: Button | None) -> None:
        self.current_index = len(self.data) - 1
        self.update_index_callback()

    def update_index_callback(self) -> None:
        self.update_index_label()
        self.update_buttons()

        # NOTE self.output is required to display IPython.display.HTML
        # IPython.display.HTML is used to execute JS code
        with self.output:
            self.callback(self.current_index)

    def build(self) -> widgets.VBox:
        return widgets.VBox(
            [widgets.HBox([self.buttons, self.index_label]), self.output]
        )


class PaginatedWidget:
    def __init__(
        self, children: list, on_paginate_callback: Callable[[int], None] | None = None
    ):
        # on_paginate_callback is an optional secondary callback,
        # called after updating the page index and displaying the new widget
        self.children = children
        self.on_paginate_callback = on_paginate_callback
        self.current_index = 0
        self.container = widgets.VBox()

        self.pagination_control = PaginationControl(children, self.on_paginate)

        # Initial display
        self.on_paginate(self.pagination_control.current_index)

    def __getitem__(self, index: int) -> widgets.Widget:
        return self.children[index]

    def on_paginate(self, index: int) -> None:
        self.container.children = [self.children[index]] if self.children else []
        if self.on_paginate_callback:
            self.on_paginate_callback(index)

    def spacer(self, height: int) -> widgets.HTML:
        return widgets.HTML(f"<div style='height: {height}px'></div>")

    def build(self) -> widgets.VBox:
        return widgets.VBox(
            [self.pagination_control.build(), self.spacer(8), self.container]
        )


class PaginatedResolveWidget:
    """
    PaginatedResolveWidget is a widget that displays
    a ResolveWidget for each ObjectDiffBatch,
    paginated by a PaginationControl widget.
    """

    def __init__(self, batches: list[ObjectDiffBatch], build_state: bool = True):
        self.build_state = build_state
        self.batches = batches
        self.resolve_widgets: list[ResolveWidget] = [
            ResolveWidget(
                batch,
                on_sync_callback=partial(self.on_click_sync, i),
                on_ignore_callback=partial(self.on_ignore, i),
                build_state=build_state,
            )
            for i, batch in enumerate(self.batches)
        ]

        self.table_uid = secrets.token_hex(4)

        # Disable the table pagination to avoid the double pagination buttons

        self.paginated_widget = PaginatedWidget(
            children=[widget.widget for widget in self.resolve_widgets],
            on_paginate_callback=self.on_paginate,
        )

        self.table_output = Output()
        self.draw_table()

        self.widget = self.build()

    def draw_table(self) -> None:
        self.batch_table = build_tabulator_table(
            obj=self.batches,
            uid=self.table_uid,
            max_height=500,
            pagination=False,
            header_sort=False,
        )
        self.table_output.clear_output()
        with self.table_output:
            display.display(display.HTML(self.batch_table))
            highlight_single_row(
                self.table_uid, self.paginated_widget.current_index, jump_to_row=True
            )

    def on_ignore(self, index: int) -> None:
        self.update_table_sync_decision(index)
        self.draw_table()

    def on_click_sync(self, index: int) -> None:
        self.update_table_sync_decision(index)
        if self.batches[index].decision is not None:
            self.paginated_widget.pagination_control.go_to_next(None)

    def update_table_sync_decision(self, index: int) -> None:
        new_decision = self.batches[index].decision_badge()
        with self.table_output:
            update_table_cell(
                uid=self.table_uid,
                index=index,
                field="Decision",
                value=new_decision,
            )

    def __getitem__(self, index: int) -> ResolveWidget:
        return self.resolve_widgets[index]

    def __len__(self) -> int:
        return len(self.resolve_widgets)

    def on_paginate(self, index: int) -> None:
        return highlight_single_row(self.table_uid, index, jump_to_row=True)

    def build(self) -> widgets.VBox:
        return widgets.VBox([self.table_output, self.paginated_widget.build()])

    def click_sync(self, index: int) -> SyftSuccess:
        return self.resolve_widgets[index].click_sync()

    def click_share_all_private_data(self, index: int) -> None:
        self.resolve_widgets[index].click_share_all_private_data()

    def _share_all(self) -> None:
        for widget in self.resolve_widgets:
            widget.click_share_all_private_data()

    def _sync_all(self) -> None:
        for idx, widget in enumerate(self.resolve_widgets):
            if widget.obj_diff_batch.decision in [
                SyncDecision.IGNORE,
                SyncDecision.SKIP,
            ]:
                print(f"skipping row {idx} (skipped/ignored)")
            else:
                widget.click_sync()

    def _repr_mimebundle_(self, **kwargs: dict) -> dict[str, str] | None:
        return self.widget._repr_mimebundle_(**kwargs)
