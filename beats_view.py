# beats_view.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainterPath, QPainterPathStroker, QPen
from PySide6.QtWidgets import QGraphicsLineItem, QGraphicsItem


@dataclass(frozen=True)
class BeatEvent:
    time_sec: float     # seconds
    is_downbeat: bool   # True if 1 else 0
    beat_id: Optional[int] = None
    checked: bool = False


def load_beats_file(path: str) -> List[BeatEvent]:
    """
    .beats file:
      <time_sec>\t<flag>
    flag: 0=beat, 1=downbeat
    """
    out: List[BeatEvent] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                # 無視（壊れてる行）
                continue
            try:
                t = float(parts[0])
                flag = int(parts[1])
            except Exception:
                continue
            out.append(BeatEvent(time_sec=t, is_downbeat=(flag == 1)))
    # 時刻順に整列（念のため）
    out.sort(key=lambda e: e.time_sec)
    return out


class BeatLineItem(QGraphicsLineItem):
    """
    エディタ全高に縦線を引く。beat 編集モードでは選択とドラッグを受け付ける。
    """
    def __init__(
        self,
        x: float,
        y0: float,
        y1: float,
        *,
        beat_id: Optional[int],
        is_downbeat: bool,
        beat_color: QColor,
        downbeat_color: QColor,
        checked: bool = False,
        beat_width: int = 1,
        downbeat_width: int = 2,
        editable: bool = False,
        on_drag_started: Optional[Callable[["BeatLineItem"], None]] = None,
        on_drag_moved: Optional[Callable[["BeatLineItem", QPointF], Optional[float]]] = None,
        on_drag_finished: Optional[Callable[["BeatLineItem", QPointF], None]] = None,
    ):
        super().__init__(x, y0, x, y1)
        self.beat_id = None if beat_id is None else int(beat_id)
        self.is_downbeat = bool(is_downbeat)
        self.checked = bool(checked)
        self.editable = bool(editable)
        self._on_drag_started = on_drag_started
        self._on_drag_moved = on_drag_moved
        self._on_drag_finished = on_drag_finished

        self._beat_color = QColor(beat_color)
        self._downbeat_color = QColor(downbeat_color)
        self._checked_color = QColor(30, 105, 255)
        self._selected_color = QColor(0, 180, 255)
        self._beat_width = int(beat_width)
        self._downbeat_width = int(downbeat_width)
        self._selected_width = max(self._downbeat_width, self._beat_width + 2)
        self._update_pen()

        self.setZValue(9000)      # 背景より上、ノートより下/上は好みで調整
        self.setAcceptedMouseButtons(Qt.LeftButton if self.editable else Qt.NoButton)
        self.setFlag(QGraphicsItem.ItemIsSelectable, self.editable)
        self.setCursor(Qt.SizeHorCursor if self.editable else Qt.ArrowCursor)

    def _update_pen(self) -> None:
        if self.isSelected():
            col = self._selected_color
            w = self._selected_width
        elif self.checked:
            col = self._checked_color
            w = max(self._beat_width + 1, self._downbeat_width if self.is_downbeat else self._beat_width)
        else:
            col = self._downbeat_color if self.is_downbeat else self._beat_color
            w = self._downbeat_width if self.is_downbeat else self._beat_width

        pen = QPen(col)
        pen.setCosmetic(True)     # ズームしても太さ一定
        pen.setWidth(w)
        self.setPen(pen)

    def set_checked(self, checked: bool) -> None:
        self.checked = bool(checked)
        self._update_pen()
        self.update()

    def itemChange(self, change, value):
        result = super().itemChange(change, value)
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self._update_pen()
        return result

    def shape(self) -> QPainterPath:
        path = super().shape()
        if not self.editable:
            return path
        stroker = QPainterPathStroker()
        stroker.setWidth(10.0)
        return stroker.createStroke(path)

    def mousePressEvent(self, event):
        if self.editable and event.button() == Qt.LeftButton:
            self.setSelected(True)
            if callable(self._on_drag_started):
                self._on_drag_started(self)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.editable and event.buttons() & Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            x = None
            if callable(self._on_drag_moved):
                x = self._on_drag_moved(self, scene_pos)
            if x is not None:
                line = self.line()
                self.setLine(float(x), line.y1(), float(x), line.y2())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if self.editable and event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            if callable(self._on_drag_finished):
                self._on_drag_finished(self, scene_pos)
