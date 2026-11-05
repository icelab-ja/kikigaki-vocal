import sys
import traceback
from bisect import bisect_right
from dataclasses import dataclass
from typing import Callable, Optional

from PySide6.QtCore import QCoreApplication

from beats_view import BeatLineItem
from piano_roll_items import GRID_SEC


def _tr(text: str) -> str:
    return QCoreApplication.translate("BeatEditController", text)


@dataclass(frozen=True)
class BeatEditControllerCallbacks:
    get_scene: Callable[[], object | None]
    get_dtpr_adapter: Callable[[], object | None]
    get_beat_events: Callable[[], list]
    refresh_after_edit: Callable[[], None]
    sync_beats_from_dtpr: Callable[[], None]
    show_information: Callable[[str, str], None]
    show_warning: Callable[[str, str], None]


class BeatEditController:
    def __init__(self, callbacks: BeatEditControllerCallbacks):
        self.callbacks = callbacks
        self.edit_mode = False
        self.line_items = []
        self._dragging_beat_id: Optional[int] = None
        self._dragging_beat_start_time: Optional[float] = None

    @property
    def scene(self):
        return self.callbacks.get_scene()

    @property
    def dtpr_adapter(self):
        return self.callbacks.get_dtpr_adapter()

    def set_edit_mode(self, enabled: bool) -> None:
        self.edit_mode = bool(enabled)

    def beat_index_by_id(self, beat_id: int) -> Optional[int]:
        adapter = self.dtpr_adapter
        if adapter is None:
            return None
        for index, (_beat_time, current_id) in enumerate(list(adapter.dtpr.BDB.index)):
            if int(current_id) == int(beat_id):
                return index
        return None

    def is_internal_beat(self, beat_id: int) -> bool:
        adapter = self.dtpr_adapter
        index = self.beat_index_by_id(int(beat_id))
        if adapter is None or index is None:
            return False
        return 0 < index < len(adapter.dtpr.BDB.index) - 1

    def snap_beat_time(self, time_sec: float) -> float:
        scene = self.scene
        if scene is None:
            return max(0.0, float(time_sec))
        return float(scene.snap_domain_time(max(0.0, float(time_sec))))

    def internal_beat_move_target(self, beat_id: int, requested_time: float) -> Optional[float]:
        adapter = self.dtpr_adapter
        if adapter is None or not self.is_internal_beat(int(beat_id)):
            return None
        index = self.beat_index_by_id(int(beat_id))
        if index is None:
            return None
        bdb_index = list(adapter.dtpr.BDB.index)
        previous_time = float(bdb_index[index - 1][0])
        next_time = float(bdb_index[index + 1][0])
        minimum_time = previous_time + float(GRID_SEC)
        maximum_time = next_time - float(GRID_SEC)
        if minimum_time >= maximum_time:
            return None
        snapped_time = self.snap_beat_time(float(requested_time))
        return max(minimum_time, min(maximum_time, snapped_time))

    def on_drag_started(self, item: BeatLineItem) -> None:
        scene = self.scene
        adapter = self.dtpr_adapter
        if not self.edit_mode or scene is None or adapter is None or item.beat_id is None:
            return
        beat = adapter.get_beat_by_id(int(item.beat_id))
        if beat is None:
            return
        self._dragging_beat_id = int(item.beat_id)
        self._dragging_beat_start_time = float(beat.beat_time)
        if not self.is_internal_beat(int(item.beat_id)):
            return
        scene.start_edit_transaction(
            operation="beat_move",
            meta={
                "beatId": int(item.beat_id),
                "beforeTimeSec": float(beat.beat_time),
                "afterTimeSec": float(beat.beat_time),
                "isDownbeat": bool(beat.is_downbeat),
            },
        )

    def on_drag_moved(self, item: BeatLineItem, scene_pos) -> Optional[float]:
        scene = self.scene
        adapter = self.dtpr_adapter
        if not self.edit_mode or scene is None or adapter is None or item.beat_id is None:
            return None
        target_time = self.internal_beat_move_target(
            int(item.beat_id),
            scene.x_to_domain_time(scene_pos.x()),
        )
        if target_time is None:
            beat = adapter.get_beat_by_id(int(item.beat_id))
            return None if beat is None else float(scene.time_to_x(float(beat.beat_time)))
        return float(scene.time_to_x(float(target_time)))

    def on_drag_finished(self, item: BeatLineItem, scene_pos) -> None:
        scene = self.scene
        adapter = self.dtpr_adapter
        if not self.edit_mode or scene is None or adapter is None or item.beat_id is None:
            return
        beat_id = int(item.beat_id)
        target_time = self.internal_beat_move_target(
            beat_id,
            scene.x_to_domain_time(scene_pos.x()),
        )
        beat = adapter.get_beat_by_id(beat_id)
        if beat is None or target_time is None or not self.is_internal_beat(beat_id):
            if scene.transaction_active:
                scene.commit_edit_transaction()
            self._clear_drag_state()
            return

        old_time = float(beat.beat_time)
        if abs(float(target_time) - old_time) <= 1e-9:
            if scene.transaction_active:
                scene.commit_edit_transaction()
            self._clear_drag_state()
            return

        try:
            adapter.beat_local_move(beat_id, float(target_time))
            if scene.transaction_meta is not None:
                scene.transaction_meta["afterTimeSec"] = float(target_time)
            self.callbacks.refresh_after_edit()
        except Exception as exc:
            print(f"[DTPR beat move] failed: {exc}", file=sys.stderr)
            traceback.print_exc()
            self.callbacks.sync_beats_from_dtpr()
        scene.commit_edit_transaction()
        self._clear_drag_state()

    def _clear_drag_state(self) -> None:
        self._dragging_beat_id = None
        self._dragging_beat_start_time = None

    def insert_beat_at_time(self, requested_time: float) -> None:
        scene = self.scene
        adapter = self.dtpr_adapter
        if not self.edit_mode or scene is None or adapter is None:
            return
        bdb_index = list(adapter.dtpr.BDB.index)
        if len(bdb_index) < 2:
            return
        target_time = self.snap_beat_time(float(requested_time))
        previous_index = bisect_right(
            [float(time_sec) for time_sec, _beat_id in bdb_index],
            target_time,
        ) - 1
        if previous_index < 0 or previous_index >= len(bdb_index) - 1:
            self.callbacks.show_information(
                _tr("Cannot Insert"),
                _tr("A beat can only be inserted between existing beats."),
            )
            return
        previous_time = float(bdb_index[previous_index][0])
        next_time = float(bdb_index[previous_index + 1][0])
        if not (previous_time + GRID_SEC <= target_time <= next_time - GRID_SEC):
            self.callbacks.show_information(
                _tr("Cannot Insert"),
                _tr("Choose a position far enough from the adjacent beats."),
            )
            return

        scene.start_edit_transaction(
            operation="beat_insert",
            meta={
                "beatId": None,
                "beforeTimeSec": None,
                "afterTimeSec": float(target_time),
                "isDownbeat": False,
            },
        )
        try:
            beat = adapter.beat_insert(float(target_time), False)
            if beat is None:
                raise RuntimeError("Inserted beat could not be identified.")
            if scene.transaction_meta is not None:
                scene.transaction_meta["beatId"] = int(beat.id)
            self.callbacks.refresh_after_edit()
        except Exception as exc:
            print(f"[DTPR beat insert] failed: {exc}", file=sys.stderr)
            traceback.print_exc()
            self.callbacks.show_warning(
                _tr("Cannot Insert"),
                _tr("An error occurred while inserting the beat:\n{error}").format(error=exc),
            )
        scene.commit_edit_transaction()

    def delete_selected_beats(self) -> None:
        scene = self.scene
        adapter = self.dtpr_adapter
        if not self.edit_mode or scene is None or adapter is None:
            return
        selected = [item for item in scene.selectedItems() if isinstance(item, BeatLineItem)]
        if not selected:
            return
        internal_items = [
            item
            for item in selected
            if item.beat_id is not None and self.is_internal_beat(int(item.beat_id))
        ]
        if not internal_items:
            self.callbacks.show_information(
                _tr("Cannot Delete"),
                _tr("The first and last beats cannot be deleted."),
            )
            return
        item = internal_items[0]
        beat = adapter.get_beat_by_id(int(item.beat_id))
        if beat is None:
            return

        scene.start_edit_transaction(
            operation="beat_delete",
            meta={
                "beatId": int(beat.id),
                "beforeTimeSec": float(beat.beat_time),
                "afterTimeSec": None,
                "isDownbeat": bool(beat.is_downbeat),
            },
        )
        try:
            adapter.beat_delete(int(beat.id))
            self.callbacks.refresh_after_edit()
        except Exception as exc:
            print(f"[DTPR beat delete] failed: {exc}", file=sys.stderr)
            traceback.print_exc()
            self.callbacks.show_warning(
                _tr("Cannot Delete"),
                _tr("An error occurred while deleting the beat:\n{error}").format(error=exc),
            )
        scene.commit_edit_transaction()

    def toggle_selected_downbeat(self) -> None:
        scene = self.scene
        adapter = self.dtpr_adapter
        if not self.edit_mode or scene is None or adapter is None:
            return
        selected = [item for item in scene.selectedItems() if isinstance(item, BeatLineItem)]
        if not selected or selected[0].beat_id is None:
            return
        beat = adapter.get_beat_by_id(int(selected[0].beat_id))
        if beat is None:
            return

        beat_id = int(beat.id)
        before_is_downbeat = bool(beat.is_downbeat)
        after_is_downbeat = not before_is_downbeat
        scene.start_edit_transaction(
            operation="beat_downbeat_toggle",
            meta={
                "beatId": beat_id,
                "beforeTimeSec": float(beat.beat_time),
                "afterTimeSec": float(beat.beat_time),
                "beforeIsDownbeat": before_is_downbeat,
                "isDownbeat": after_is_downbeat,
            },
        )
        try:
            adapter.beat_set_downbeat(beat_id, after_is_downbeat)
            self.callbacks.refresh_after_edit()
            scene.clearSelection()
            for line_item in self.line_items:
                item_beat_id = getattr(line_item, "beat_id", None)
                if item_beat_id is not None and int(item_beat_id) == beat_id:
                    line_item.setSelected(True)
                    break
        except Exception as exc:
            print(f"[DTPR beat downbeat toggle] failed: {exc}", file=sys.stderr)
            traceback.print_exc()
            self.callbacks.show_warning(
                _tr("Cannot Change"),
                _tr("An error occurred while toggling the downbeat:\n{error}").format(error=exc),
            )
        scene.commit_edit_transaction()

    def set_checked(self, beat_id: int, checked: bool) -> None:
        for item in self.line_items:
            item_beat_id = getattr(item, "beat_id", None)
            if item_beat_id is not None and int(item_beat_id) == int(beat_id):
                if hasattr(item, "set_checked"):
                    item.set_checked(bool(checked))
                return

    def clear_beats(self) -> None:
        scene = self.scene
        if scene is not None:
            for item in self.line_items:
                try:
                    scene.removeItem(item)
                except Exception:
                    pass
        self.line_items = []

    def redraw_beats(
        self,
        *,
        visible: bool,
        beat_color,
        downbeat_color,
        beat_width: int,
        downbeat_width: int,
    ) -> None:
        scene = self.scene
        events = self.callbacks.get_beat_events()
        if scene is None:
            return
        if not events or not visible:
            self.clear_beats()
            return
        self.clear_beats()

        try:
            y0 = float(scene.pitch_to_y(int(scene.max_pitch)))
            y1 = float(scene.pitch_to_y(int(scene.min_pitch))) + float(scene.px_per_pitch)
        except Exception:
            rect = scene.sceneRect()
            y0, y1 = float(rect.top()), float(rect.bottom())

        items = []
        for event in events:
            item = BeatLineItem(
                float(scene.time_to_x(float(event.time_sec))),
                y0,
                y1,
                beat_id=getattr(event, "beat_id", None),
                is_downbeat=bool(event.is_downbeat),
                beat_color=beat_color,
                downbeat_color=downbeat_color,
                checked=bool(getattr(event, "checked", False)),
                beat_width=int(beat_width),
                downbeat_width=int(downbeat_width),
                editable=bool(self.edit_mode),
                on_drag_started=self.on_drag_started,
                on_drag_moved=self.on_drag_moved,
                on_drag_finished=self.on_drag_finished,
            )
            scene.addItem(item)
            items.append(item)
        self.line_items = items
