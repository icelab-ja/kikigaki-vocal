from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QTableWidgetItem

from piano_roll_items import NoteItem


@dataclass
class DeveloperPanelCallbacks:
    get_table: Callable
    get_dtpr_adapter: Callable
    get_ptpr_scene: Callable
    get_mtpr_scene: Callable
    get_ptpr_view: Callable
    get_mtpr_view: Callable
    get_ptpr_keyboard: Callable
    get_mtpr_keyboard: Callable


class DeveloperPanelController:
    def __init__(self, callbacks: DeveloperPanelCallbacks):
        self.callbacks = callbacks
        self.row_to_note_id: list[int] = []
        self.note_id_to_row: dict[int, int] = {}

    def set_visible(self, visible: bool) -> None:
        table = self.callbacks.get_table()
        if table is None:
            return
        if visible:
            table.show()
            self.update()
        else:
            table.hide()

    def update(self) -> None:
        table = self.callbacks.get_table()
        if table is None or not table.isVisible():
            return

        adapter = self.callbacks.get_dtpr_adapter()
        ptpr_scene = self.callbacks.get_ptpr_scene()
        if ptpr_scene is None or adapter is None:
            table.setRowCount(0)
            self.row_to_note_id = []
            self.note_id_to_row = {}
            return

        notes = adapter.get_ptpr_notes_sorted()
        table.setRowCount(len(notes))
        self.row_to_note_id = []
        self.note_id_to_row = {}

        def format_int(value):
            return "" if value is None else str(int(value))

        def format_sec(value):
            return "" if value is None else f"{float(value):.3f}"

        for row, note in enumerate(notes):
            note_id = int(getattr(note, "id"))
            self.row_to_note_id.append(note_id)
            self.note_id_to_row[note_id] = row

            mtpr_note = adapter.get_mtpr_note_by_id(note_id)
            values = [
                format_int(note_id),
                format_int(getattr(note, "pitch", None)),
                format_int(getattr(mtpr_note, "abs_ontick", None) if mtpr_note is not None else None),
                format_int(getattr(mtpr_note, "on_beat_id", None) if mtpr_note is not None else None),
                format_int(getattr(mtpr_note, "on_metrical_pos", None) if mtpr_note is not None else None),
                format_int(getattr(mtpr_note, "abs_offtick", None) if mtpr_note is not None else None),
                format_int(getattr(mtpr_note, "off_beat_id", None) if mtpr_note is not None else None),
                format_int(getattr(mtpr_note, "off_metrical_pos", None) if mtpr_note is not None else None),
                format_sec(getattr(note, "ideal_ontime", None)),
                format_sec(getattr(note, "real_ontime", None)),
                format_sec(getattr(note, "ideal_offtime", None)),
                format_sec(getattr(note, "real_offtime", None)),
            ]

            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                table.setItem(row, column, item)

        self.sync_selection()

    def sync_selection(self) -> None:
        table = self.callbacks.get_table()
        if table is None or not table.isVisible():
            return

        ptpr_scene = self.callbacks.get_ptpr_scene()
        mtpr_scene = self.callbacks.get_mtpr_scene()
        if ptpr_scene is None and mtpr_scene is None:
            return

        selected_ids = set()
        for scene in (ptpr_scene, mtpr_scene):
            if scene is None:
                continue
            for item in scene.selectedItems():
                if isinstance(item, NoteItem):
                    selected_ids.add(item.note_id)

        table.blockSignals(True)
        for row in range(table.rowCount()):
            note_id = self.row_to_note_id[row] if row < len(self.row_to_note_id) else None
            is_selected = note_id in selected_ids
            color = QColor(220, 235, 255) if is_selected else QColor(255, 255, 255)
            for column in range(table.columnCount()):
                item = table.item(row, column)
                if item is not None:
                    item.setBackground(color)

        table.clearSelection()
        for note_id in selected_ids:
            row = self.note_id_to_row.get(note_id)
            if row is not None:
                table.selectRow(row)
        table.blockSignals(False)

    def on_cell_clicked(self, row: int, _column: int) -> None:
        if row < 0 or row >= len(self.row_to_note_id):
            return

        ptpr_scene = self.callbacks.get_ptpr_scene()
        mtpr_scene = self.callbacks.get_mtpr_scene()
        if ptpr_scene is None and mtpr_scene is None:
            return

        note_id = self.row_to_note_id[row]
        ptpr_item = None
        mtpr_item = None

        if ptpr_scene is not None:
            ptpr_scene.blockSignals(True)
            ptpr_scene.clearSelection()
            ptpr_item = ptpr_scene.note_items.get(note_id)
            if ptpr_item is not None:
                ptpr_item.setSelected(True)
            ptpr_scene.blockSignals(False)

        if mtpr_scene is not None:
            mtpr_scene.blockSignals(True)
            mtpr_scene.clearSelection()
            mtpr_item = mtpr_scene.note_items.get(note_id)
            if mtpr_item is not None:
                mtpr_item.setSelected(True)
            mtpr_scene.blockSignals(False)

        self.sync_selection()

        ptpr_view = self.callbacks.get_ptpr_view()
        if ptpr_view is not None and ptpr_item is not None:
            ptpr_view.centerOn(ptpr_item)
            keyboard = self.callbacks.get_ptpr_keyboard()
            if keyboard is not None:
                keyboard.update()

        mtpr_view = self.callbacks.get_mtpr_view()
        if mtpr_view is not None and mtpr_item is not None:
            mtpr_view.centerOn(mtpr_item)
            keyboard = self.callbacks.get_mtpr_keyboard()
            if keyboard is not None:
                keyboard.update()
