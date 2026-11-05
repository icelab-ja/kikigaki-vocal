from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import QGraphicsItem, QGraphicsLineItem


class OnsetCandidateLineItem(QGraphicsLineItem):
    def __init__(self, x: float, y0: float, y1: float, time_sec: float, owner_note_id: int, on_click, kind: str = "onset"):
        super().__init__(x, y0, x, y1)
        self.time_sec = float(time_sec)
        self.owner_note_id = int(owner_note_id)
        self.kind = str(kind)
        self._on_click = on_click

        pen = QPen(QColor(0, 200, 0))  # 緑
        pen.setWidth(6)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setZValue(10_000)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if callable(self._on_click):
                self._on_click(self, event.scenePos())
                event.accept()
                return
        super().mousePressEvent(event)


class MTPRCandidateTickLineItem(QGraphicsLineItem):
    def __init__(self, x: float, y0: float, y1: float, abs_tick: int, owner_note_id: int, on_click, kind: str = "onset"):
        super().__init__(x, y0, x, y1)
        self.abs_tick = int(abs_tick)
        self.owner_note_id = int(owner_note_id)
        self.kind = str(kind)
        self._on_click = on_click

        color = QColor(0, 200, 0)
        if self.kind == "offset":
            color = QColor(160, 70, 220)
        elif self.kind == "split":
            color = QColor(0, 180, 180)

        pen = QPen(color)
        pen.setWidth(6)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setZValue(10_200)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if callable(self._on_click):
                self._on_click(self, event.scenePos())
                event.accept()
                return
        super().mousePressEvent(event)


def _candidate_alpha_for_rank(rank: int, total: int) -> int:
    rank = max(1, int(rank))
    total = max(1, int(total))
    if total <= 1:
        return 255
    rank = min(rank, total)
    alpha = 255.0 + (80.0 - 255.0) * float(rank - 1) / float(total - 1)
    return int(round(alpha))


def _apply_candidate_rank_alpha(item, rank: int, total: int) -> None:
    pen = item.pen()
    color = pen.color()
    color.setAlpha(_candidate_alpha_for_rank(rank, total))
    pen.setColor(color)
    item.setPen(pen)
