from dataclasses import dataclass
import sys
import traceback
from typing import Callable

from PySide6.QtCore import Qt

from beats_view import BeatLineItem
from piano_roll_items import NoteItem


@dataclass
class CheckedControllerCallbacks:
    get_dtpr_adapter: Callable
    get_ptpr_scene: Callable
    get_mtpr_scene: Callable
    get_edit_logger: Callable
    set_beat_checked: Callable[[int, bool], None]
    on_changed: Callable[[], None]


class CheckedController:
    def __init__(self, callbacks: CheckedControllerCallbacks):
        self.callbacks = callbacks

    def _record_change(self, operation: str, targets: list[dict]) -> None:
        logger = self.callbacks.get_edit_logger()
        if not targets or logger is None:
            return
        try:
            op_id = logger.next_op_id()
            timestamp = logger._now_iso()
            record = {
                "recordType": "checked",
                "opId": op_id,
                "timestamp": timestamp,
                "operation": str(operation),
                "targets": [
                    {
                        "domain": str(target["domain"]),
                        "id": int(target["id"]),
                        "checked": bool(target["checked"]),
                    }
                    for target in targets
                ],
            }
            logger._append_jsonl_dict(record)
        except Exception as exc:
            print(f"[edit logger] checked record failed: {exc}", file=sys.stderr)
            traceback.print_exc()

    def _apply_targets(self, targets: list[dict], *, operation: str) -> bool:
        adapter = self.callbacks.get_dtpr_adapter()
        if adapter is None:
            return False

        changed_targets: list[dict] = []
        for target in targets:
            domain = str(target["domain"]).lower()
            target_id = int(target["id"])
            checked = bool(target["checked"])
            if adapter.set_checked(domain, target_id, checked):
                changed_targets.append({
                    "domain": domain,
                    "id": target_id,
                    "checked": checked,
                })

        if not changed_targets:
            return False

        for target in changed_targets:
            domain = target["domain"]
            target_id = int(target["id"])
            checked = bool(target["checked"])
            if domain == "beat":
                self.callbacks.set_beat_checked(target_id, checked)
                continue

            scene = (
                self.callbacks.get_ptpr_scene()
                if domain == "ptpr"
                else self.callbacks.get_mtpr_scene()
            )
            item = getattr(scene, "note_items", {}).get(target_id) if scene is not None else None
            if item is not None and hasattr(item, "set_checked"):
                item.set_checked(checked)

        self._record_change(operation, changed_targets)
        self.callbacks.on_changed()
        return True

    def _selected_note_targets(self, scene) -> list[dict]:
        adapter = self.callbacks.get_dtpr_adapter()
        if scene is None or adapter is None:
            return []
        domain = str(scene.domain).lower()
        targets = []
        for item in scene.selectedItems():
            if not isinstance(item, NoteItem):
                continue
            note_id = int(item.note_id)
            current = adapter.get_checked(domain, note_id)
            targets.append({"domain": domain, "id": note_id, "checked": not current})
        return targets

    def _selected_beat_targets(self, scene) -> list[dict]:
        adapter = self.callbacks.get_dtpr_adapter()
        if scene is None or adapter is None:
            return []
        targets = []
        for item in scene.selectedItems():
            if not isinstance(item, BeatLineItem) or item.beat_id is None:
                continue
            beat_id = int(item.beat_id)
            current = adapter.get_checked("beat", beat_id)
            targets.append({"domain": "beat", "id": beat_id, "checked": not current})
        return targets

    def toggle_selected(self, scene) -> bool:
        if self.callbacks.get_dtpr_adapter() is None or scene is None:
            return False
        targets = self._selected_note_targets(scene)
        if not targets and scene._is_ptpr() and bool(getattr(scene, "beat_edit_mode", False)):
            targets = self._selected_beat_targets(scene)
        return self._apply_targets(targets, operation="checked_toggle")

    def _until_note_targets(self, scene, checked: bool) -> list[dict]:
        selected = [item for item in scene.selectedItems() if isinstance(item, NoteItem)]
        if not selected:
            return []
        selected_times = []
        for item in selected:
            note = scene._get_display_note_by_id(int(item.note_id))
            if note is not None:
                selected_times.append(float(scene._note_onset(note)))
        if not selected_times:
            return []

        threshold = max(selected_times)
        domain = str(scene.domain).lower()
        targets = []
        for note in scene._get_display_notes():
            note_id = int(scene._note_id(note))
            if float(scene._note_onset(note)) <= threshold + 1e-9:
                targets.append({"domain": domain, "id": note_id, "checked": bool(checked)})
        return targets

    def _until_beat_targets(self, scene, checked: bool) -> list[dict]:
        selected = [
            item
            for item in scene.selectedItems()
            if isinstance(item, BeatLineItem) and item.beat_id is not None
        ]
        adapter = self.callbacks.get_dtpr_adapter()
        if not selected or adapter is None:
            return []
        selected_times = []
        for item in selected:
            beat = adapter.get_beat_by_id(int(item.beat_id))
            if beat is not None:
                selected_times.append(float(beat.beat_time))
        if not selected_times:
            return []

        threshold = max(selected_times)
        return [
            {"domain": "beat", "id": int(beat.id), "checked": bool(checked)}
            for beat in adapter.get_beats_sorted()
            if float(beat.beat_time) <= threshold + 1e-9
        ]

    def set_until_selection(self, scene, checked: bool) -> bool:
        if self.callbacks.get_dtpr_adapter() is None or scene is None:
            return False
        targets = self._until_note_targets(scene, checked)
        if not targets and scene._is_ptpr() and bool(getattr(scene, "beat_edit_mode", False)):
            targets = self._until_beat_targets(scene, checked)
        return self._apply_targets(targets, operation="checked_set_until")

    def handle_key(self, scene, modifiers) -> bool:
        command = bool(modifiers & (Qt.ControlModifier | Qt.MetaModifier))
        alt = bool(modifiers & Qt.AltModifier)
        if command:
            return self.set_until_selection(scene, checked=not alt)
        return self.toggle_selected(scene)
