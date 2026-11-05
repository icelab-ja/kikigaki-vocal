from dataclasses import dataclass
from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import (
    QHBoxLayout,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from shiboken6 import isValid

from piano_roll_items import NoteItem, SelectionTimeMarkerLineItem
from piano_roll_scene import MTPRGridScene, PianoRollScene
from piano_roll_widgets import (
    PianoKeyboardWidget,
    PianoRollView,
    PianoRollViewCallbacks,
    VocalWaveformWidget,
)


@dataclass(frozen=True)
class PianoRollPanelCallbacks:
    on_confidence_update: Callable[[], None]
    on_candidates_update: Callable[[], None]
    on_ptpr_notes_changed: Callable[[], None]
    on_mtpr_notes_changed: Callable[[], None]
    on_notes_audition_requested: Callable[[list[int]], None]
    on_playhead_dragged: Callable[[float], None]
    on_keyboard_clicked: Callable[[int], None]
    on_selection_changed: Callable[[], None]
    on_dev_update: Callable[[], None]
    beat_insert: Callable[[float], None]
    time_sec_to_abs_tick: Callable[[float], float]
    abs_tick_to_time_sec: Callable[[float], float]


class PianoRollPanel:
    def __init__(self, callbacks: PianoRollPanelCallbacks):
        self.callbacks = callbacks
        self.scene: Optional[PianoRollScene] = None
        self.view: Optional[PianoRollView] = None
        self.keyboard: Optional[PianoKeyboardWidget] = None
        self.mtpr_scene: Optional[MTPRGridScene] = None
        self.mtpr_view: Optional[PianoRollView] = None
        self.mtpr_keyboard: Optional[PianoKeyboardWidget] = None
        self.waveform_widget: Optional[VocalWaveformWidget] = None
        self.splitter: Optional[QSplitter] = None
        self.container: Optional[QWidget] = None
        self._syncing_horizontal_scroll = False
        self._syncing_zoom = False
        self._syncing_vertical_scroll = False
        self._syncing_selection = False
        self._selection_marker_scene = None
        self._selection_marker_onset = None
        self._selection_marker_offset = None
        self._selection_marker_updating = False
        self._selection_marker_preferred_scene = None

    def clear(self) -> None:
        self.clear_selection_time_markers()
        if self.view is not None:
            self.view.set_keyboard(None)
            self.view.set_waveform_widget(None)
            self.view.setParent(None)
        if self.keyboard is not None:
            self.keyboard.setParent(None)
        if self.mtpr_view is not None:
            self.mtpr_view.set_keyboard(None)
            self.mtpr_view.setParent(None)
        if self.mtpr_keyboard is not None:
            self.mtpr_keyboard.setParent(None)
        if self.waveform_widget is not None:
            self.waveform_widget.setParent(None)

        self.scene = None
        self.view = None
        self.keyboard = None
        self.mtpr_scene = None
        self.mtpr_view = None
        self.mtpr_keyboard = None
        self.waveform_widget = None
        self.splitter = None
        self.container = None
        self._selection_marker_preferred_scene = None

    def rebuild(
        self,
        *,
        edit_logger,
        dtpr_adapter,
        beat_edit_mode: bool,
        subdivisions_per_beat: int,
        mtpr_max_tick: int,
        mtpr_px_per_tick: float,
        view_callbacks: PianoRollViewCallbacks,
        vocal_audio=None,
        audio_sample_rate: int | None = None,
    ) -> QWidget:
        self.clear()

        self.scene = PianoRollScene(
            edit_logger=edit_logger,
            dtpr_adapter=dtpr_adapter,
        )
        self.scene.beat_edit_mode = bool(beat_edit_mode)
        self.scene.beat_insert_callback = self.callbacks.beat_insert
        self.scene.note_editing_enabled = not bool(beat_edit_mode)
        self.view = PianoRollView(self.scene, callbacks=view_callbacks)

        self.scene.notesChanged.connect(self.callbacks.on_confidence_update)
        self.scene.notesChanged.connect(self.callbacks.on_candidates_update)
        self.scene.notesChanged.connect(self.callbacks.on_ptpr_notes_changed)
        self.scene.notesAuditionRequested.connect(self.callbacks.on_notes_audition_requested)
        self.scene.playheadDragged.connect(self.callbacks.on_playhead_dragged)

        self.mtpr_scene = MTPRGridScene(
            edit_logger=edit_logger,
            dtpr_adapter=dtpr_adapter,
            subdivisions_per_beat=subdivisions_per_beat,
        )
        self.mtpr_scene.undo_stack = self.scene.undo_stack
        self.mtpr_scene.redo_stack = self.scene.redo_stack
        self.mtpr_scene.playhead_draggable = False
        try:
            self.mtpr_scene.set_px_per_tick(float(mtpr_px_per_tick))
        except Exception:
            pass
        self.mtpr_scene.set_tick_range(int(mtpr_max_tick))
        self.mtpr_scene.notesChanged.connect(self.callbacks.on_mtpr_notes_changed)
        self.mtpr_scene.notesAuditionRequested.connect(self.callbacks.on_notes_audition_requested)
        try:
            self.mtpr_scene.set_bdb_grid(
                dtpr_adapter.dtpr.BDB,
                subdivisions_per_beat=subdivisions_per_beat,
            )
            self.mtpr_scene.set_mtpr_notes(list(dtpr_adapter.dtpr.MTPR.notes.values()))
        except Exception:
            pass

        self.mtpr_view = PianoRollView(self.mtpr_scene, callbacks=view_callbacks)
        self.mtpr_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.mtpr_view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.mtpr_keyboard = PianoKeyboardWidget(self.mtpr_scene, self.mtpr_view)
        self.mtpr_view.set_keyboard(self.mtpr_keyboard)

        self.view.horizontalScrollBar().valueChanged.connect(
            lambda _value: self.on_horizontal_scroll_changed(self.view)
        )
        self.mtpr_view.horizontalScrollBar().valueChanged.connect(
            lambda _value: self.on_horizontal_scroll_changed(self.mtpr_view)
        )
        self.view.verticalScrollBar().valueChanged.connect(
            lambda _value: self.on_vertical_scroll_changed(self.view)
        )
        self.mtpr_view.verticalScrollBar().valueChanged.connect(
            lambda _value: self.on_vertical_scroll_changed(self.mtpr_view)
        )
        self.mtpr_keyboard.keyClicked.connect(self.callbacks.on_keyboard_clicked)

        ptpr_row = QWidget()
        ptpr_row.setMinimumHeight(80)
        ptpr_layout = QHBoxLayout(ptpr_row)
        ptpr_layout.setContentsMargins(0, 0, 0, 0)
        ptpr_layout.setSpacing(0)
        self.keyboard = PianoKeyboardWidget(self.scene, self.view)
        self.view.set_keyboard(self.keyboard)
        self.keyboard.keyClicked.connect(self.callbacks.on_keyboard_clicked)
        ptpr_layout.addWidget(self.keyboard)
        ptpr_layout.addWidget(self.view, 1)

        self.waveform_widget = VocalWaveformWidget()
        self.waveform_widget.setMinimumHeight(12)
        self.waveform_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.waveform_widget.set_scene_and_view(
            self.scene,
            self.view,
            on_seek_requested=self.callbacks.on_playhead_dragged,
        )
        self.view.set_waveform_widget(self.waveform_widget)

        mtpr_row = QWidget()
        mtpr_row.setMinimumHeight(80)
        mtpr_layout = QHBoxLayout(mtpr_row)
        mtpr_layout.setContentsMargins(0, 0, 0, 0)
        mtpr_layout.setSpacing(0)
        mtpr_layout.addWidget(self.mtpr_keyboard)
        mtpr_layout.addWidget(self.mtpr_view, 1)

        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(mtpr_row)
        self.splitter.addWidget(ptpr_row)
        self.splitter.addWidget(self.waveform_widget)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([260, 260, 70])

        self.container = QWidget()
        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        container_layout.addWidget(self.splitter, 1)

        if vocal_audio is not None and audio_sample_rate is not None:
            self.waveform_widget.set_audio(vocal_audio, int(audio_sample_rate))

        self.scene.notesChanged.connect(self.callbacks.on_dev_update)
        ptpr_scene = self.scene
        mtpr_scene = self.mtpr_scene
        self.scene.selectionChanged.connect(
            lambda scene=ptpr_scene: self.on_scene_selection_changed(scene)
        )
        self.mtpr_scene.selectionChanged.connect(
            lambda scene=mtpr_scene: self.on_scene_selection_changed(scene)
        )
        self.scene.noteItemGeometryChanged.connect(self.update_selection_time_markers)
        self.mtpr_scene.noteItemGeometryChanged.connect(self.update_selection_time_markers)
        return self.container

    def scroll_to_start(self) -> None:
        if self.view is not None:
            scrollbar = self.view.horizontalScrollBar()
            scrollbar.setValue(scrollbar.minimum())
        if self.mtpr_view is not None:
            scrollbar = self.mtpr_view.horizontalScrollBar()
            scrollbar.setValue(scrollbar.minimum())

    def active_view(self) -> Optional[PianoRollView]:
        if self.mtpr_view is not None and self.mtpr_view.hasFocus():
            return self.mtpr_view
        return self.view

    def _history_scene(self, stack_name: str):
        stack = getattr(self.scene, stack_name, None) if self.scene is not None else None
        if stack:
            entry = stack[-1]
            domain = entry.get("domain") if isinstance(entry, dict) else None
            if domain == "mtpr" and self.mtpr_scene is not None:
                return self.mtpr_scene
            if domain == "ptpr" and self.scene is not None:
                return self.scene

        active_view = self.active_view()
        if active_view is self.mtpr_view and self.mtpr_scene is not None:
            return self.mtpr_scene
        return self.scene if self.scene is not None else self.mtpr_scene

    def undo(self) -> None:
        target_scene = self._history_scene("undo_stack")
        if target_scene is not None:
            target_scene.undo()

    def redo(self) -> None:
        target_scene = self._history_scene("redo_stack")
        if target_scene is not None:
            target_scene.redo()

    def apply_x_zoom(self, *, zoom_in: bool) -> None:
        view = self.active_view()
        if view is not None:
            view.apply_x_zoom(zoom_in=zoom_in)

    def apply_y_zoom(self, *, zoom_in: bool) -> None:
        view = self.active_view()
        if view is not None:
            view.apply_y_zoom(zoom_in=zoom_in)

    def ptpr_view_center_time_sec(self) -> Optional[float]:
        if self.view is None or self.scene is None:
            return None
        rect = self.view.mapToScene(self.view.viewport().rect()).boundingRect()
        if rect.isNull():
            return None
        return float(rect.center().x() * self.scene.seconds_per_pixel)

    def mtpr_view_center_time_sec(self) -> Optional[float]:
        if self.mtpr_view is None or self.mtpr_scene is None:
            return None
        rect = self.mtpr_view.mapToScene(self.mtpr_view.viewport().rect()).boundingRect()
        if rect.isNull():
            return None
        abs_tick = float(rect.center().x() * self.mtpr_scene.seconds_per_pixel)
        return float(self.callbacks.abs_tick_to_time_sec(abs_tick))

    def center_views_to_time_sec(self, time_sec: float, *, target: str = "both") -> None:
        if target in ("both", "ptpr") and self.view is not None and self.scene is not None:
            rect = self.view.mapToScene(self.view.viewport().rect()).boundingRect()
            if not rect.isNull():
                self.view.centerOn(float(self.scene.time_to_x(time_sec)), rect.center().y())
        if target in ("both", "mtpr") and self.mtpr_view is not None and self.mtpr_scene is not None:
            rect = self.mtpr_view.mapToScene(self.mtpr_view.viewport().rect()).boundingRect()
            if not rect.isNull():
                abs_tick = float(self.callbacks.time_sec_to_abs_tick(float(time_sec)))
                self.mtpr_view.centerOn(float(self.mtpr_scene.tick_to_x(abs_tick)), rect.center().y())

    def on_horizontal_scroll_changed(self, source_view) -> None:
        if self._syncing_horizontal_scroll:
            return
        self._syncing_horizontal_scroll = True
        try:
            if source_view is self.view:
                time_sec = self.ptpr_view_center_time_sec()
                if time_sec is not None:
                    self.center_views_to_time_sec(time_sec, target="mtpr")
            elif source_view is self.mtpr_view:
                time_sec = self.mtpr_view_center_time_sec()
                if time_sec is not None:
                    self.center_views_to_time_sec(time_sec, target="ptpr")
        finally:
            self._syncing_horizontal_scroll = False

    @staticmethod
    def view_center_scene_y(view) -> Optional[float]:
        if view is None:
            return None
        rect = view.mapToScene(view.viewport().rect()).boundingRect()
        if rect.isNull():
            return None
        return float(rect.center().y())

    @staticmethod
    def center_view_to_scene_y(view, scene_y: float) -> None:
        if view is None:
            return
        rect = view.mapToScene(view.viewport().rect()).boundingRect()
        if not rect.isNull():
            view.centerOn(float(rect.center().x()), float(scene_y))

    def on_vertical_scroll_changed(self, source_view) -> None:
        if self._syncing_vertical_scroll:
            return
        target_view = self.mtpr_view if source_view is self.view else self.view
        center_y = self.view_center_scene_y(source_view)
        if target_view is None or center_y is None:
            return
        self._syncing_vertical_scroll = True
        try:
            self.center_view_to_scene_y(target_view, center_y)
            keyboard = getattr(target_view, "keyboard", None)
            if keyboard is not None and isValid(keyboard):
                keyboard.update()
        finally:
            self._syncing_vertical_scroll = False

    def on_x_zoom_changed(self, source_view, factor: float) -> None:
        if self._syncing_zoom:
            return
        self._syncing_zoom = True
        try:
            if source_view is self.view and self.mtpr_view is not None:
                time_sec = self.ptpr_view_center_time_sec()
                self.mtpr_view.scale(float(factor), 1.0)
                keyboard = getattr(self.mtpr_view, "keyboard", None)
                if keyboard is not None and isValid(keyboard):
                    keyboard.update()
                if time_sec is not None:
                    self.center_views_to_time_sec(time_sec, target="mtpr")
            elif source_view is self.mtpr_view and self.view is not None:
                time_sec = self.mtpr_view_center_time_sec()
                self.view.scale(float(factor), 1.0)
                keyboard = getattr(self.view, "keyboard", None)
                if keyboard is not None and isValid(keyboard):
                    keyboard.update()
                waveform = getattr(self.view, "waveform_widget", None)
                if waveform is not None and isValid(waveform):
                    waveform.update()
                if time_sec is not None:
                    self.center_views_to_time_sec(time_sec, target="ptpr")
        finally:
            self._syncing_zoom = False

    def on_y_zoom_changed(self, source_view, _factor: float, center_scene_y: float) -> None:
        if self._syncing_zoom:
            return
        target_view = self.mtpr_view if source_view is self.view else self.view
        if target_view is None:
            return
        source_zoom = abs(float(source_view.transform().m22()))
        target_zoom = abs(float(target_view.transform().m22()))
        if target_zoom <= 1e-9:
            return

        self._syncing_zoom = True
        self._syncing_vertical_scroll = True
        try:
            target_view.scale(1.0, source_zoom / target_zoom)
            self.center_view_to_scene_y(target_view, float(center_scene_y))
            target_view.viewport().update()
            keyboard = getattr(target_view, "keyboard", None)
            if keyboard is not None and isValid(keyboard):
                keyboard.update()
        finally:
            self._syncing_vertical_scroll = False
            self._syncing_zoom = False

    def sync_note_selection(self, source_scene) -> None:
        if self._syncing_selection:
            return
        if self.scene is None or self.mtpr_scene is None:
            return
        if source_scene not in (self.scene, self.mtpr_scene):
            return
        target_scene = self.mtpr_scene if source_scene is self.scene else self.scene
        selected_ids = {
            int(item.note_id)
            for item in source_scene.selectedItems()
            if isinstance(item, NoteItem)
        }

        self._syncing_selection = True
        try:
            target_scene.blockSignals(True)
            target_scene.clearSelection()
            for note_id in selected_ids:
                item = target_scene.note_items.get(note_id)
                if item is not None:
                    item.setSelected(True)
            target_scene.blockSignals(False)
        finally:
            if target_scene.signalsBlocked():
                target_scene.blockSignals(False)
            self._syncing_selection = False

    def on_scene_selection_changed(self, source_scene) -> None:
        if source_scene not in (self.scene, self.mtpr_scene):
            return
        self._selection_marker_preferred_scene = source_scene
        self.sync_note_selection(source_scene)
        self.update_selection_time_markers()
        self.callbacks.on_selection_changed()

    def clear_selection_time_markers(self) -> None:
        target_scene = self._selection_marker_scene
        if target_scene is None:
            target_scene = self.scene
        if target_scene is None:
            self._selection_marker_onset = None
            self._selection_marker_offset = None
            self._selection_marker_scene = None
            return

        for attr in ("_selection_marker_onset", "_selection_marker_offset"):
            item = getattr(self, attr)
            if item is not None:
                try:
                    target_scene.removeItem(item)
                except Exception:
                    pass
                setattr(self, attr, None)
        self._selection_marker_scene = None

    def _ensure_selection_time_markers(self, target_scene) -> None:
        if target_scene is None:
            return
        if (
            self._selection_marker_scene is not None
            and self._selection_marker_scene is not target_scene
        ):
            self.clear_selection_time_markers()

        pen = QPen(QColor(255, 80, 80, 200))
        pen.setCosmetic(True)
        pen.setWidth(4)
        if self._selection_marker_onset is None:
            self._selection_marker_onset = SelectionTimeMarkerLineItem(
                kind="onset",
                on_geometry_changed=self.update_selection_time_markers,
            )
            self._selection_marker_onset.setPen(pen)
            self._selection_marker_onset.setZValue(15000)
            target_scene.addItem(self._selection_marker_onset)

        if self._selection_marker_offset is None:
            self._selection_marker_offset = SelectionTimeMarkerLineItem(
                kind="offset",
                on_geometry_changed=self.update_selection_time_markers,
            )
            self._selection_marker_offset.setPen(pen)
            self._selection_marker_offset.setZValue(15000)
            target_scene.addItem(self._selection_marker_offset)
        self._selection_marker_scene = target_scene

    def update_selection_time_markers(self) -> None:
        if self.scene is None or self._selection_marker_updating:
            return

        self._selection_marker_updating = True
        try:
            candidate_scenes = []
            preferred_scene = self._selection_marker_preferred_scene
            if preferred_scene in (self.scene, self.mtpr_scene):
                candidate_scenes.append(preferred_scene)
            for scene in (self.scene, self.mtpr_scene):
                if scene is not None and scene not in candidate_scenes:
                    candidate_scenes.append(scene)

            target_scene = None
            selected_item = None
            for scene in candidate_scenes:
                selected = [
                    item
                    for item in scene.selectedItems()
                    if isinstance(item, NoteItem)
                ]
                if len(selected) == 1:
                    target_scene = scene
                    selected_item = selected[0]
                    break

            if target_scene is None or selected_item is None:
                self.clear_selection_time_markers()
                return

            onset_x = float(selected_item.pos().x())
            offset_x = float(selected_item.pos().x() + selected_item.rect().width())
            scene_rect = target_scene.sceneRect()
            y0 = float(scene_rect.top())
            y1 = float(scene_rect.bottom())

            self._ensure_selection_time_markers(target_scene)
            if self._selection_marker_onset is not None:
                self._selection_marker_onset.setLine(onset_x, y0, onset_x, y1)
            if self._selection_marker_offset is not None:
                self._selection_marker_offset.setLine(offset_x, y0, offset_x, y1)
        finally:
            self._selection_marker_updating = False

    def scroll_to_time(self, time_sec: float) -> None:
        if self.view is None or self.scene is None:
            return
        x = float(self.scene.time_to_x(time_sec))
        rect = self.view.mapToScene(self.view.viewport().rect()).boundingRect()
        if not rect.isNull():
            self.view.centerOn(x + rect.width() * 0.3, rect.center().y())

        if self.mtpr_view is not None and self.mtpr_scene is not None:
            abs_tick = float(self.callbacks.time_sec_to_abs_tick(float(time_sec)))
            mtpr_x = float(self.mtpr_scene.tick_to_x(abs_tick))
            mtpr_rect = self.mtpr_view.mapToScene(self.mtpr_view.viewport().rect()).boundingRect()
            if not mtpr_rect.isNull():
                self.mtpr_view.centerOn(mtpr_x + mtpr_rect.width() * 0.3, mtpr_rect.center().y())
        if self.waveform_widget is not None:
            self.waveform_widget.update()

    @staticmethod
    def auto_scroll_view_for_x(view, x: float) -> None:
        if view is None:
            return
        rect = view.mapToScene(view.viewport().rect()).boundingRect()
        if rect.isNull() or rect.left() <= x <= rect.right():
            return
        if x > rect.right():
            view.centerOn(rect.center().x() + rect.width(), rect.center().y())
        else:
            view.centerOn(x + rect.width() * 0.3, rect.center().y())
