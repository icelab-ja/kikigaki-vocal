from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPainterPathStroker, QPen
from PySide6.QtWidgets import QApplication, QGraphicsItem, QGraphicsLineItem, QGraphicsRectItem


GRID_SEC = 0.01


class PlayheadLineItem(QGraphicsLineItem):
    def __init__(self, scene: "PianoRollScene"):
        super().__init__()
        self._scene = scene
        self._dragging = False
        self._scrolling = False
        self.setZValue(1000)
        self.setCursor(Qt.SizeHorCursor)
        self.setAcceptedMouseButtons(Qt.LeftButton)

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        scene = self._scene
        if scene is None or not getattr(scene, "playhead_draggable", True):
            return
        self._dragging = True
        self._update_from_scene_pos(event.scenePos())

    def mouseMoveEvent(self, event):
        if not self._dragging:
            return
        self._update_from_scene_pos(event.scenePos())

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        if not self._dragging:
            return
        self._dragging = False
        self._update_from_scene_pos(event.scenePos())

    def _update_from_scene_pos(self, scene_pos: QPointF):
        scene = self._scene
        if scene is None:
            return
        x = max(0.0, float(scene_pos.x()))
        t = x * float(scene.seconds_per_pixel)
        if GRID_SEC > 0:
            t = round(t / GRID_SEC) * GRID_SEC
        scene.set_playhead_time(t)
        if hasattr(scene, "playheadDragged"):
            scene.playheadDragged.emit(float(t))

        if self._scrolling:
            return

        views = scene.views()
        if views:
            view = views[0]
            visible_rect = view.mapToScene(view.viewport().rect()).boundingRect()
            if not visible_rect.isNull():
                scale_x = view.transform().m11() or 1.0
                margin = 60.0 / scale_x
                if x > visible_rect.right() - margin:
                    dx_scene = x - (visible_rect.right() - margin)
                elif x < visible_rect.left() + margin:
                    dx_scene = x - (visible_rect.left() + margin)
                else:
                    dx_scene = 0.0

                if dx_scene != 0.0:
                    sb = view.horizontalScrollBar()
                    dx_view = int(round(dx_scene * scale_x))
                    if dx_view != 0:
                        step = max(-80, min(80, dx_view))
                        new_val = sb.value() + step
                        new_val = max(sb.minimum(), min(sb.maximum(), new_val))
                        self._scrolling = True
                        try:
                            sb.setValue(new_val)
                        finally:
                            self._scrolling = False


def _is_piano_roll_scene(scene) -> bool:
    return bool(getattr(scene, "is_piano_roll_scene", False))


NOTE_FILL_ALPHA = 120  # 0..255（値が小さいほど透過）
ONSET_EDGE_COLOR = QColor(220, 30, 30)     # 赤
OFFSET_EDGE_COLOR = QColor(30, 180, 60)    # 緑（任意に変更可）
SELECT_BORDER_COLOR = QColor(255, 200, 0)  # 選択枠（黄）

class NoteItem(QGraphicsRectItem):
    """
    1つの Note を表す矩形アイテム。
    - 中央部ドラッグ: ノート全体を移動（10ms / 半音グリッドにスナップ）
    - 左端ドラッグ: オンセットのみ変更（長さも変わる）
    - 右端ドラッグ: オフセットのみ変更（長さも変わる）
    """

    def __init__(self, note, x: float, y: float, w: float, h: float, *args, **kwargs):
        # ★ 先に状態変数を定義（super().__init__ より前）
        self.note_id = note.id
        self.resize_mode: Optional[str] = None   # "left" | "right" | None
        self.edge_hover: Optional[str] = None
        self.drag_start_scene_pos: Optional[QPointF] = None
        self.original_pos: Optional[QPointF] = None
        self.original_rect: Optional[QRectF] = None
        self._move_press_scene_pos: Optional[QPointF] = None
        self._move_pending_selected_ids: list[int] = []
        self._move_transaction_started: bool = False
        self.checked: bool = bool(getattr(note, "checked", False))

        # 矩形はローカル座標 (0,0,w,h)
        super().__init__(0, 0, w, h, *args, **kwargs)

        self.setBrush(QBrush(QColor(100, 150, 240)))
        self.setPen(QPen(Qt.black, 0.5))
        self.setFlag(QGraphicsRectItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsRectItem.ItemIsMovable, True)
        self.setFlag(QGraphicsRectItem.ItemSendsGeometryChanges, True)

        # ホバーイベントを受け取る
        self.setAcceptHoverEvents(True)

        # シーン上の初期位置
        self.setPos(x, y)

    def set_checked(self, checked: bool) -> None:
        self.checked = bool(checked)
        self.update()

    def paint(self, painter: QPainter, option, widget=None):
        """
        - 音符は透過（背景が見える）
        - 選択時は太い枠線で強調（点線に依存しない）
        - onset/offset の縦線は描かない
        """
        painter.save()

        # ピクセル境界のずれ（0.5px問題）を避けるため、アンチエイリアスを切る
        painter.setRenderHint(QPainter.Antialiasing, False)

        r = self.rect()

        # --- fill (透過) ---
        fill = self.brush().color()
        if not fill.isValid():
            fill = QColor(150, 150, 255)
        fill = QColor(fill)
        fill.setAlpha(20)  # 0..255（小さいほど透明）
        painter.setBrush(QBrush(fill))
        painter.setPen(Qt.NoPen)
        painter.drawRect(r)

        # --- base pen（選択強調を優先：太さはここで決定） ---
        if self.isSelected():
            border_w = 4
            base_border_color = QColor(255, 200, 0)  # 選択時：上/下の色（外枠感）
        elif self.checked:
            border_w = 4
            base_border_color = QColor(30, 105, 255)
        else:
            border_w = 3
            base_border_color = QColor(0, 0, 0)

        # 座標（0.5pxずれを避けるため rect そのまま使う）
        xL = r.left()
        xR = r.right()
        yT = r.top()
        yB = r.bottom()

        # 上辺
        pen = QPen(base_border_color, border_w)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawLine(QPointF(xL, yT), QPointF(xR, yT))

        # 下辺
        painter.drawLine(QPointF(xL, yB), QPointF(xR, yB))

        # 左辺（オンセット色）
        penL = QPen(ONSET_EDGE_COLOR, border_w)
        penL.setCosmetic(True)
        painter.setPen(penL)
        painter.drawLine(QPointF(xL, yT), QPointF(xL, yB))

        # 右辺（オフセット色） ※オンセットと重なる場合は描かない
        if xR != xL:
            # penR = QPen(OFFSET_EDGE_COLOR, border_w)
            penR = QPen(base_border_color, border_w)
            penR.setCosmetic(True)
            painter.setPen(penR)
            painter.drawLine(QPointF(xR, yT), QPointF(xR, yB))

        painter.restore()

    # === ホバー時のカーソル変更 ===
    def hoverMoveEvent(self, event):
        rect = self.rect()
        x = event.pos().x()
        margin = 2.0                    # 端判定（px相当）を狭く
        margin = min(margin, rect.width() * 0.25)  # 短い音符で端判定が支配しないよう上限
        margin = max(margin, 1.0)

        if x <= rect.left() + margin:
            self.setCursor(Qt.SizeHorCursor)
            self.edge_hover = "left"
        elif x >= rect.right() - margin:
            self.setCursor(Qt.SizeHorCursor)
            self.edge_hover = "right"
        else:
            self.setCursor(Qt.ArrowCursor)
            self.edge_hover = None

        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event):
        self.setCursor(Qt.ArrowCursor)
        self.edge_hover = None
        super().hoverLeaveEvent(event)

    # === マウス押下 ===
    def mousePressEvent(self, event):
        scene = self.scene()

        # Alt + 左クリック → このノートをクリック位置で分割
        if event.button() == Qt.LeftButton and (event.modifiers() & Qt.AltModifier):
            if _is_piano_roll_scene(scene):
                # scene.start_edit_transaction()
                note = scene._get_display_note_by_id(self.note_id)
                if note is not None:
                    scene_pos = self.mapToScene(event.pos())
                    t_split = scene.snap_domain_time(scene.x_to_domain_time(scene_pos.x()))
                    scene.split_note_at(note, t_split)
            # 分割で完結させるので、その後のドラッグ処理には進まない
            return

        # Alt 以外の左クリック → リサイズは即開始、移動はドラッグが発生してから開始
        if event.button() == Qt.LeftButton and _is_piano_roll_scene(scene):
            selected_ids = [it.note_id for it in scene.selectedItems() if isinstance(it, NoteItem)]
            if self.note_id not in selected_ids:
                selected_ids.append(self.note_id)
            self._move_pending_selected_ids = [int(nid) for nid in selected_ids]
            self._move_press_scene_pos = self.mapToScene(event.pos())
            self._move_transaction_started = False

            # 伸縮か移動かをここで確定
            if self.edge_hover == "left":
                scene.start_edit_transaction(
                    operation="resizeOnset",
                    meta={"noteIds": selected_ids}
                )
            elif self.edge_hover == "right":
                scene.start_edit_transaction(
                    operation="resizeOffset",
                    meta={"noteIds": selected_ids}
                )

        if event.button() == Qt.LeftButton and self.edge_hover in ("left", "right"):
            # 端のリサイズモードに入る
            self.resize_mode = self.edge_hover
            self.drag_start_scene_pos = self.mapToScene(event.pos())
            self.original_pos = QPointF(self.pos())
            self.original_rect = QRectF(self.rect())
            # リサイズ中は「移動モード」を無効化
            self.setFlag(QGraphicsRectItem.ItemIsMovable, False)
        else:
            # 通常の移動モード
            self.resize_mode = None
            self.drag_start_scene_pos = None
            self.original_pos = None
            self.original_rect = None
            super().mousePressEvent(event)

    # === マウスドラッグ中 ===
    def mouseMoveEvent(self, event):
        if self.resize_mode is not None:
            self._apply_resize_from_scene_pos(self.mapToScene(event.pos()))

            # ★追加：マーカー追従更新用
            scene = self.scene()
            if _is_piano_roll_scene(scene):
                scene.noteItemGeometryChanged.emit(int(self.note_id))
        else:
            scene = self.scene()
            if (
                _is_piano_roll_scene(scene)
                and event.buttons() & Qt.LeftButton
                and not self._move_transaction_started
                and self._move_press_scene_pos is not None
            ):
                scene_pos = self.mapToScene(event.pos())
                delta = scene_pos - self._move_press_scene_pos
                if abs(float(delta.x())) >= 1.0 or abs(float(delta.y())) >= 1.0:
                    scene.start_edit_transaction(
                        operation="move",
                        meta={"noteIds": list(self._move_pending_selected_ids)}
                    )
                    self._move_transaction_started = True

            super().mouseMoveEvent(event)

            scene = self.scene()
            if _is_piano_roll_scene(scene) and not scene._is_ptpr():
                # MTPR: drag中表示もグリッド単位に固定
                p = self.pos()
                x_new = scene.snap_scene_x(float(p.x()))
                pitch_f = scene.max_pitch - float(p.y()) / float(scene.px_per_pitch)
                pitch_snap = int(round(pitch_f))
                pitch_snap = max(scene.min_pitch, min(scene.max_pitch, pitch_snap))
                y_new = scene.pitch_to_y(pitch_snap)
                if abs(float(p.x()) - x_new) > 1e-6 or abs(float(p.y()) - float(y_new)) > 1e-6:
                    self.setPos(x_new, y_new)
            if _is_piano_roll_scene(scene) and hasattr(scene, "noteItemGeometryChanged"):
                scene.noteItemGeometryChanged.emit(int(self.note_id))

    # === マウス解放 ===
    def mouseReleaseEvent(self, event):
        scene = self.scene()

        if self.resize_mode is not None:
            # リサイズ終了 → 移動を再び有効化
            self.setFlag(QGraphicsRectItem.ItemIsMovable, True)
            self.resize_mode = None
            self.drag_start_scene_pos = None
            self.original_pos = None
            self.original_rect = None

            if _is_piano_roll_scene(scene) and event.button() == Qt.LeftButton:
                scene.commit_edit_transaction()
            self._move_press_scene_pos = None
            self._move_pending_selected_ids = []
            self._move_transaction_started = False
        else:
            # 通常の移動モード
            super().mouseReleaseEvent(event)
            if (
                _is_piano_roll_scene(scene)
                and event.button() == Qt.LeftButton
                and self._move_transaction_started
            ):
                scene.commit_edit_transaction()
            self._move_press_scene_pos = None
            self._move_pending_selected_ids = []
            self._move_transaction_started = False

    # === 移動時のグリッドスナップ（中央ドラッグ時） ===
    def itemChange(self, change, value):
        # ★ コンストラクタ途中で呼ばれた場合などに備えた保険
        if not hasattr(self, "resize_mode"):
            return super().itemChange(change, value)

        if change == QGraphicsRectItem.ItemPositionChange and self.resize_mode is None:
            new_pos: QPointF = value
            scene = self.scene()
            if _is_piano_roll_scene(scene):
                old_pos = self.pos()

                # X方向: 10ms グリッド
                modifiers = QApplication.keyboardModifiers()
                if modifiers & Qt.ShiftModifier:
                    x_new = old_pos.x()
                else:
                    x_new = scene.snap_scene_x(new_pos.x())
                    if x_new < 0.0:
                        x_new = 0.0
                    ordered_items = sorted(
                        [it for it in scene.note_items.values() if isinstance(it, NoteItem)],
                        key=lambda it: (float(it.pos().x()), float(it.pos().x() + it.rect().width()), int(it.note_id)),
                    )
                    item_pos = {int(it.note_id): idx for idx, it in enumerate(ordered_items)}
                    idx = item_pos.get(int(self.note_id))
                    if idx is not None:
                        width = float(self.rect().width())
                        min_x = 0.0
                        max_x = float("inf")
                        if idx > 0:
                            prev_item = ordered_items[idx - 1]
                            min_x = max(min_x, float(prev_item.pos().x()) + scene.min_note_width_px())
                        if idx < len(ordered_items) - 1:
                            next_item = ordered_items[idx + 1]
                            max_x = float(next_item.pos().x()) - width
                        if x_new < min_x:
                            x_new = min_x
                        if x_new > max_x:
                            x_new = max_x

                # Y方向: 半音グリッド
                px_per_grid_y = scene.px_per_pitch
                dy = new_pos.y() - old_pos.y()
                if px_per_grid_y > 0:
                    snapped_dy = round(dy / px_per_grid_y) * px_per_grid_y
                else:
                    snapped_dy = dy
                y_new = old_pos.y() + snapped_dy

                # ピッチ範囲内に収める
                pitch_f = scene.max_pitch - y_new / scene.px_per_pitch
                pitch_snap = int(round(pitch_f))
                pitch_snap = max(scene.min_pitch, min(scene.max_pitch, pitch_snap))
                y_new = scene.pitch_to_y(pitch_snap)

                return QPointF(x_new, y_new)

        if change == QGraphicsRectItem.ItemPositionHasChanged and self.resize_mode is None:
            scene = self.scene()
            if _is_piano_roll_scene(scene) and hasattr(scene, "noteItemGeometryChanged"):
                scene.noteItemGeometryChanged.emit(int(self.note_id))

        return super().itemChange(change, value)

    def _apply_resize_from_scene_pos(self, current_scene_pos: QPointF):
        scene = self.scene()
        if not _is_piano_roll_scene(scene):
            return
        if self.drag_start_scene_pos is None or self.original_pos is None or self.original_rect is None:
            return

        dx_scene = current_scene_pos.x() - self.drag_start_scene_pos.x()

        base_time = scene.x_to_domain_time(self.original_pos.x())
        moved_time = scene.x_to_domain_time(self.original_pos.x() + dx_scene)
        moved_time_q = scene.snap_domain_time(moved_time)
        dx_snap = scene.domain_time_to_x(moved_time_q) - scene.domain_time_to_x(base_time)

        min_width = scene.min_note_width_px()

        ordered_items = sorted(
            [it for it in scene.note_items.values() if isinstance(it, NoteItem)],
            key=lambda it: (float(it.pos().x()), float(it.pos().x() + it.rect().width()), int(it.note_id)),
        )
        item_pos = {int(it.note_id): idx for idx, it in enumerate(ordered_items)}
        current_idx = item_pos.get(int(self.note_id))

        if self.resize_mode == "left":
            original_right = self.original_pos.x() + self.original_rect.width()
            new_x = self.original_pos.x() + dx_snap
            if new_x < 0.0:
                new_x = 0.0
            if current_idx is not None and current_idx > 0:
                prev_item = ordered_items[current_idx - 1]
                prev_min_right = float(prev_item.pos().x()) + min_width
                if new_x < prev_min_right:
                    new_x = prev_min_right
            max_x = original_right - min_width
            if new_x > max_x:
                new_x = max_x

            new_w = original_right - new_x
            if new_w < min_width:
                new_w = min_width

            self.setPos(new_x, self.original_pos.y())
            r = QRectF(self.rect())
            r.setWidth(new_w)
            self.setRect(r)

        elif self.resize_mode == "right":
            new_right_x = self.original_pos.x() + self.original_rect.width() + dx_snap
            min_right = self.original_pos.x() + min_width
            if new_right_x < min_right:
                new_right_x = min_right
            if current_idx is not None and current_idx < len(ordered_items) - 1:
                next_item = ordered_items[current_idx + 1]
                max_right = float(next_item.pos().x())
                if max_right < min_right:
                    min_right = max_right
                if new_right_x > max_right:
                    new_right_x = max_right

            new_w = new_right_x - self.original_pos.x()
            if new_w < min_width and min_right >= self.original_pos.x() + min_width:
                new_w = min_width
            elif new_w < 0.1:
                new_w = 0.1

            self.setRect(0, 0, new_w, self.original_rect.height())
            self.setPos(self.original_pos)

    def begin_forced_resize(self, *, mode: str, scene_pos: QPointF):
        scene = self.scene()
        if not _is_piano_roll_scene(scene):
            return

        # 既存の操作名に合わせる
        op = "resizeOnset" if mode == "left" else "resizeOffset"
        scene.start_edit_transaction(operation=op, meta={"noteIds": [int(self.note_id)]})

        self.resize_mode = mode
        self.drag_start_scene_pos = QPointF(scene_pos)
        self.original_pos = QPointF(self.pos())
        self.original_rect = QRectF(self.rect())

        self.setFlag(QGraphicsRectItem.ItemIsMovable, False)

    def update_forced_resize(self, *, scene_pos: QPointF):
        if self.resize_mode is None:
            return
        self._apply_resize_from_scene_pos(QPointF(scene_pos))

        scene = self.scene()
        if _is_piano_roll_scene(scene):
            scene.noteItemGeometryChanged.emit(int(self.note_id))

    def end_forced_resize(self):
        scene = self.scene()
        if not _is_piano_roll_scene(scene):
            return

        self.setFlag(QGraphicsRectItem.ItemIsMovable, True)

        # リサイズ状態をクリア
        self.resize_mode = None
        self.drag_start_scene_pos = None
        self.original_pos = None
        self.original_rect = None

        # 既存 Undo/Redo の単位で確定
        scene.commit_edit_transaction()

        # 念のため通知
        scene.noteItemGeometryChanged.emit(int(self.note_id))

class SelectionTimeMarkerLineItem(QGraphicsLineItem):
    """
    選択中ノートの onset/offset を示す縦線。
    線上ドラッグで NoteItem を左/右リサイズできる“ハンドル”としても機能させる。
    """
    def __init__(self, *, kind: str, on_geometry_changed):
        super().__init__()
        self.kind = kind            # "onset" or "offset"
        self._on_geometry_changed = on_geometry_changed
        self._drag_target = None    # NoteItem

        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)

    def shape(self):
        # ★見た目は太くせず当たり判定だけ広げる（掴みやすさ改善）
        path = QPainterPath()
        ln = self.line()
        path.moveTo(ln.p1())
        path.lineTo(ln.p2())
        stroker = QPainterPathStroker()
        stroker.setWidth(6.0)  # 当たり判定だけ太く（好みで 8〜14 くらい）
        return stroker.createStroke(path)

    def hoverMoveEvent(self, event):
        self.setCursor(Qt.SizeHorCursor)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return

        scene = self.scene()
        if scene is None:
            return

        selected = [it for it in scene.selectedItems() if isinstance(it, NoteItem)]
        if len(selected) != 1:
            return

        note_item = selected[0]
        self._drag_target = note_item

        mode = "left" if self.kind == "onset" else "right"
        note_item.begin_forced_resize(mode=mode, scene_pos=event.scenePos())

        event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_target is None:
            return
        self._drag_target.update_forced_resize(scene_pos=event.scenePos())
        # マーカー自身も追従更新（選択状態は変わらないが、表示位置更新のため）
        if callable(self._on_geometry_changed):
            self._on_geometry_changed()
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._drag_target is None:
            return
        self._drag_target.end_forced_resize()
        self._drag_target = None
        if callable(self._on_geometry_changed):
            self._on_geometry_changed()
        event.accept()
