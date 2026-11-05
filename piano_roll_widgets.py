import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QGraphicsView, QWidget
from shiboken6 import isValid

from piano_roll_items import GRID_SEC
from piano_roll_scene import PianoRollScene


@dataclass
class PianoRollViewCallbacks:
    apply_candidate_by_number: Optional[Callable] = None
    toggle_selected_beat_downbeat: Optional[Callable] = None
    handle_checked_key: Optional[Callable] = None
    delete_selected_beats: Optional[Callable] = None
    on_x_zoom_changed: Optional[Callable] = None
    on_y_zoom_changed: Optional[Callable] = None
    update_mouse_coord: Optional[Callable] = None
    clear_mouse_coord: Optional[Callable] = None

class VocalWaveformWidget(QWidget):
    """
    歌声音源の振幅を表示するウィジェット。
    - PianoRollView と時間軸を共有（scene.seconds_per_pixel + view のスクロール/ズームに追従）
    - 再生中のプレイヘッド位置も縦線で表示
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.audio: Optional[np.ndarray] = None
        self.sample_rate: Optional[int] = None
        self.audio_max_amp: float = 1.0

        self.scene: Optional[PianoRollScene] = None
        self.view: Optional[QGraphicsView] = None
        self._on_seek_requested = None

        self.playhead_time: float = 0.0

        # 背景色などの設定
        self.setMinimumHeight(12)
        self.setFocusPolicy(Qt.StrongFocus)

    def set_scene_and_view(
        self,
        scene: "PianoRollScene",
        view: QGraphicsView,
        on_seek_requested=None,
    ):
        """時間軸同期のために、ピアノロールの scene / view を渡す。"""
        self.scene = scene
        self.view = view
        self._on_seek_requested = on_seek_requested
        self.update()

    def set_audio(self, audio: np.ndarray, sample_rate: int):
        """
        歌声音源の波形データを設定する。
        audio: (N,) または (N, ch) の numpy 配列
        """
        if audio.ndim == 2:
            # ステレオの場合は片チャンネルか平均を使用（ここでは平均）
            audio = audio.mean(axis=1)
        self.audio = audio.astype(np.float32, copy=False)
        self.sample_rate = int(sample_rate)
        max_amp = float(np.max(np.abs(self.audio))) if self.audio.size > 0 else 0.0
        self.audio_max_amp = max(max_amp, 1e-6)
        self.update()

    def set_playhead_time(self, t: float):
        """再生位置（秒）を更新し、プレイヘッド線を動かす。"""
        self.playhead_time = max(0.0, float(t))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        try:
            rect = self.rect()

            # 背景
            painter.fillRect(rect, QColor(245, 245, 245))

            if self.audio is None or self.sample_rate is None:
                painter.setPen(QColor(150, 150, 150))
                painter.drawText(rect, Qt.AlignCenter, "No vocal audio")
                return

            if self.scene is None or self.view is None:
                return

            # --- 鍵盤幅ぶんのオフセットを計算 ---
            keyboard_w = 0
            # 左側に keyboard を addWidget している前提なので、親のレイアウトから推定可能
            # 最も安全なのは MainWindow 側で keyboard.width() を渡す方法ですが、
            # ここでは view の global pos を使って「波形ウィジェット内での view 左端」を求めます。

            # waveform 内座標での view 左端 x
            view_left_global = self.view.viewport().mapToGlobal(self.view.viewport().rect().topLeft())
            wf_left_global = self.mapToGlobal(self.rect().topLeft())
            x0 = view_left_global.x() - wf_left_global.x()
            if x0 < 0:
                x0 = 0

            draw_w = self.view.viewport().width()
            if draw_w <= 1:
                return

            # 波形の描画領域（鍵盤の下は描かない）
            draw_rect = QRectF(x0, rect.top(), draw_w, rect.height())

            # --- 現在ビューで見えているシーン範囲（時間範囲） ---
            visible_scene = self.view.mapToScene(self.view.viewport().rect()).boundingRect()
            if visible_scene.isNull():
                return

            sec_per_px = self.scene.seconds_per_pixel
            t_start = max(0.0, visible_scene.left() * sec_per_px)
            t_end = max(t_start, visible_scene.right() * sec_per_px)

            sr = self.sample_rate
            audio_duration = float(self.audio.shape[0]) / float(sr)
            if t_start >= audio_duration:
                return

            orig_t_end = t_end
            draw_w_eff = int(draw_w)
            if t_end > audio_duration:
                t_end = audio_duration
                span = max(1e-9, orig_t_end - t_start)
                ratio = max(0.0, min(1.0, (t_end - t_start) / span))
                draw_w_eff = max(1, int(round(draw_w * ratio)))

            start_idx = int(t_start * sr)
            end_idx = int(t_end * sr)
            start_idx = max(0, min(start_idx, self.audio.shape[0]))
            end_idx = max(start_idx + 1, min(end_idx, self.audio.shape[0]))

            total_samples = end_idx - start_idx
            if total_samples <= 1:
                return

            # 縦方向スケール
            H = draw_rect.height()
            mid_y = draw_rect.top() + H / 2.0
            segment = self.audio[start_idx:end_idx]
            amp_scale = (H * 0.45) / float(self.audio_max_amp)

            painter.setRenderHint(QPainter.Antialiasing, False)
            pen = QPen(QColor(80, 120, 200))
            pen.setWidth(1)
            painter.setPen(pen)

            # 1ピクセルあたりのサンプル数（描画幅は view の viewport 幅に合わせる）
            samples_per_pixel = max(1, total_samples // int(draw_w_eff))

            # 波形描画（min/max 縦線）
            for i in range(int(draw_w_eff)):
                i0 = start_idx + i * samples_per_pixel
                i1 = min(start_idx + (i + 1) * samples_per_pixel, end_idx)
                if i0 >= i1:
                    break
                chunk = self.audio[i0:i1]
                v_min = float(chunk.min())
                v_max = float(chunk.max())
                y1 = mid_y - v_max * amp_scale
                y2 = mid_y - v_min * amp_scale
                x = draw_rect.left() + i
                painter.drawLine(x, y1, x, y2)

            # ゼロライン
            painter.setPen(QPen(QColor(180, 180, 180), 1, Qt.DashLine))
            painter.drawLine(draw_rect.left(), mid_y, draw_rect.right(), mid_y)

            # プレイヘッド（波形描画領域内で）
            if t_end > t_start and (t_start <= self.playhead_time <= t_end):
                ratio = (self.playhead_time - t_start) / (t_end - t_start)
                px = draw_rect.left() + ratio * draw_w_eff
                painter.setPen(QPen(QColor(255, 200, 0), 1))
                painter.drawLine(px, draw_rect.top(), px, draw_rect.bottom())
        finally:
            painter.end()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        if self.scene is None or self.view is None or self._on_seek_requested is None:
            return
        if self.audio is None or self.sample_rate is None:
            return

        view_left_global = self.view.viewport().mapToGlobal(self.view.viewport().rect().topLeft())
        wf_left_global = self.mapToGlobal(self.rect().topLeft())
        x0 = view_left_global.x() - wf_left_global.x()
        if x0 < 0:
            x0 = 0

        draw_w = self.view.viewport().width()
        if draw_w <= 1:
            return

        x = float(event.position().x())
        if x < x0 or x > x0 + draw_w:
            return

        visible_scene = self.view.mapToScene(self.view.viewport().rect()).boundingRect()
        if visible_scene.isNull():
            return

        sec_per_px = self.scene.seconds_per_pixel
        t_start = max(0.0, visible_scene.left() * sec_per_px)
        t_end = max(t_start, visible_scene.right() * sec_per_px)
        if t_end <= t_start:
            return

        ratio = (x - x0) / float(draw_w)
        ratio = max(0.0, min(1.0, ratio))
        t = t_start + (t_end - t_start) * ratio
        if GRID_SEC > 0:
            t = round(t / GRID_SEC) * GRID_SEC

        if callable(self._on_seek_requested):
            self._on_seek_requested(float(t))

    def wheelEvent(self, event):
        if self.view is None:
            super().wheelEvent(event)
            return

        angle = event.angleDelta()
        pixel = event.pixelDelta()

        if event.modifiers() & Qt.ControlModifier:
            delta = angle.y() if angle.y() != 0 else pixel.y()
            if delta == 0:
                event.accept()
                return
            if hasattr(self.view, "apply_x_zoom"):
                self.view.apply_x_zoom(zoom_in=(delta > 0))
                event.accept()
                return

        sb = self.view.horizontalScrollBar()
        if sb is None:
            super().wheelEvent(event)
            return

        if not pixel.isNull():
            delta = pixel.x() if pixel.x() != 0 else pixel.y()
            step = int(delta)
        else:
            delta = angle.x() if angle.x() != 0 else angle.y()
            if delta == 0:
                super().wheelEvent(event)
                return
            steps = float(delta) / 120.0
            step = int(round(steps * max(20, sb.singleStep() * 3)))

        sb.setValue(max(sb.minimum(), min(sb.maximum(), sb.value() - step)))
        self.update()
        event.accept()

class PianoRollView(QGraphicsView):
    MIN_Y_ZOOM = 0.75
    MAX_Y_ZOOM = 4.0

    def __init__(
        self,
        scene: PianoRollScene,
        parent=None,
        callbacks: Optional[PianoRollViewCallbacks] = None,
    ):
        self.waveform_widget: Optional[VocalWaveformWidget] = None
        self.keyboard = None  # type: Optional["PianoKeyboardWidget"]
        self.callbacks = callbacks or PianoRollViewCallbacks()
        super().__init__(scene, parent)
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    
    def set_keyboard(self, keyboard):
        self.keyboard = keyboard

    def keyPressEvent(self, event):
        # ここで一度だけ取得しておく
        scene = self.scene()

        key = event.key()
        if Qt.Key_1 <= key <= Qt.Key_5:
            n = int(key - Qt.Key_1) + 1
            callback = self.callbacks.apply_candidate_by_number
            if callable(callback) and callback(n):
                event.accept()
                return
            # 候補が無い / 範囲外なら何もしない
            event.ignore()
            return

        if event.key() == Qt.Key_Up:
            if event.modifiers() & Qt.ControlModifier:
                self.apply_y_zoom(zoom_in=True)
            else:
                self.apply_x_zoom(zoom_in=True)
            event.accept()
            return
        if event.key() == Qt.Key_Down:
            if event.modifiers() & Qt.ControlModifier:
                self.apply_y_zoom(zoom_in=False)
            else:
                self.apply_x_zoom(zoom_in=False)
            event.accept()
            return

        if event.key() == Qt.Key_B:
            if (
                isinstance(scene, PianoRollScene)
                and scene._is_ptpr()
                and bool(getattr(scene, "beat_edit_mode", False))
            ):
                callback = self.callbacks.toggle_selected_beat_downbeat
                if callable(callback):
                    callback()
                event.accept()
                return

        if event.key() == Qt.Key_V:
            if isinstance(scene, PianoRollScene):
                callback = self.callbacks.handle_checked_key
                if callable(callback) and callback(scene, event.modifiers()):
                    event.accept()
                    return

        # 削除
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if (
                isinstance(scene, PianoRollScene)
                and scene._is_ptpr()
                and bool(getattr(scene, "beat_edit_mode", False))
            ):
                callback = self.callbacks.delete_selected_beats
                if callable(callback):
                    callback()
                event.accept()
                return
            if isinstance(scene, PianoRollScene):
                scene.delete_selected_notes()
            else:
                # 念のため、想定外のシーン型ならデフォルト挙動にフォールバック
                super().keyPressEvent(event)

        # 結合（Mキー）
        elif event.key() == Qt.Key_M:
            if isinstance(scene, PianoRollScene):
                scene.join_selected_notes()
            else:
                super().keyPressEvent(event)

        else:
            # それ以外のキーは従来どおりビュー側のデフォルト処理へ
            super().keyPressEvent(event)

    def apply_x_zoom(self, *, zoom_in: bool, notify: bool = True):
        factor = 1.2 if zoom_in else 1 / 1.2
        self.scale(factor, 1.0)

        keyboard = getattr(self, "keyboard", None)
        waveform_widget = getattr(self, "waveform_widget", None)
        if keyboard is not None and isValid(keyboard):
            keyboard.update()
        if waveform_widget is not None and isValid(waveform_widget):
            waveform_widget.update()
        if notify:
            callback = self.callbacks.on_x_zoom_changed
            if callable(callback):
                callback(self, factor)

    def apply_y_zoom(self, *, zoom_in: bool, notify: bool = True):
        requested_factor = 1.2 if zoom_in else 1 / 1.2
        current_zoom = abs(float(self.transform().m22()))
        target_zoom = max(
            self.MIN_Y_ZOOM,
            min(self.MAX_Y_ZOOM, current_zoom * requested_factor),
        )
        if current_zoom <= 1e-9:
            return
        factor = target_zoom / current_zoom
        if abs(factor - 1.0) <= 1e-9:
            return

        center = self.mapToScene(self.viewport().rect().center())
        self.scale(1.0, factor)
        self.centerOn(center)
        self.viewport().update()

        keyboard = getattr(self, "keyboard", None)
        if keyboard is not None and isValid(keyboard):
            keyboard.update()
        if notify:
            callback = self.callbacks.on_y_zoom_changed
            if callable(callback):
                callback(self, factor, float(center.y()))

    def wheelEvent(self, event):
        modifiers = event.modifiers()

        # Ctrlで横ズーム、Ctrl+Shiftで縦ズーム
        if modifiers & Qt.ControlModifier:
            angle = event.angleDelta().y()
            if angle == 0:
                return

            zoom_in = angle > 0
            if modifiers & Qt.ShiftModifier:
                self.apply_y_zoom(zoom_in=zoom_in)
            else:
                self.apply_x_zoom(zoom_in=zoom_in)

            # ★ keyboard がまだ生きているときだけ update
            keyboard = getattr(self, "keyboard", None)
            if keyboard is not None and isValid(keyboard):
                keyboard.update()

            waveform_widget = getattr(self, "waveform_widget", None)
            if waveform_widget is not None and isValid(waveform_widget):
                waveform_widget.update()   # ★ 追加
            event.accept()
        else:
            # 通常のスクロールはそのままビューに任せる
            super().wheelEvent(event)
            keyboard = getattr(self, "keyboard", None)
            if keyboard is not None and isValid(keyboard):
                keyboard.update()

            waveform_widget = getattr(self, "waveform_widget", None)
            if waveform_widget is not None and isValid(waveform_widget):
                waveform_widget.update()   # ★ 追加

    def mouseMoveEvent(self, event):
        scene = self.scene()
        if isinstance(scene, PianoRollScene):
            pos = event.position().toPoint()
            scene_pos = self.mapToScene(pos)
            callback = self.callbacks.update_mouse_coord
            if callable(callback):
                callback(scene, scene_pos)
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        callback = self.callbacks.clear_mouse_coord
        if callable(callback):
            callback()
        super().leaveEvent(event)
    
    def scrollContentsBy(self, dx, dy):
        super().scrollContentsBy(dx, dy)
        keyboard = getattr(self, "keyboard", None)
        if keyboard is not None and isValid(keyboard):
            keyboard.update()

        waveform_widget = getattr(self, "waveform_widget", None)
        if waveform_widget is not None and isValid(waveform_widget):
            waveform_widget.update()       # ★ 追加

    def set_keyboard(self, keyboard: "PianoKeyboardWidget"):
        self.keyboard = keyboard

    def set_waveform_widget(self, widget: VocalWaveformWidget):
        """波形ウィジェットを登録しておく（スクロール/ズーム時に update 用）。"""
        self.waveform_widget = widget

class PianoKeyboardWidget(QWidget):
    """
    ピアノロール左側に表示する鍵盤っぽい縦バー。
    - 白鍵: 明るい背景 + 黒文字
    - 黒鍵: 暗い背景 + 白文字
    - 各行に MIDI ピッチ番号を表示
    - 縦位置は QGraphicsView のズーム・スクロールと同期
    """
    keyClicked = Signal(int)
    def __init__(self, scene: PianoRollScene, view: PianoRollView, parent=None):
        super().__init__(parent)
        self.scene = scene
        self.view = view
        self.setMinimumWidth(60)
        self.setMaximumWidth(80)

    def set_scene_and_view(self, scene: PianoRollScene, view: PianoRollView):
        self.scene = scene
        self.view = view
        self.update()

    def sizeHint(self):
        return QSize(60, 200)

    @staticmethod
    def _pitch_label(pitch: int) -> str:
        names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
        p = int(pitch)
        return f"{names[p % 12]}{(p // 12) - 1}({p})"

    def _viewport_origin_y_in_keyboard(self) -> float:
        """viewport(0,0) を keyboard ローカルへ写したときの Y"""
        vp_global = self.view.viewport().mapToGlobal(QPoint(0, 0))
        return float(self.mapFromGlobal(vp_global).y())

    def _pitch_height_in_view_pixels(self) -> float:
        """1 pitch（scene.px_per_pitch）が view 上で何pxか"""
        scale_y = abs(float(self.view.transform().m22()))
        return float(self.scene.px_per_pitch) * scale_y

    def _y0_vp_for_max_pitch(self) -> float:
        """max_pitch の上端（scene→viewport）の Y（viewport座標）"""
        y_scene = float(self.scene.pitch_to_y(self.scene.max_pitch))
        return float(self.view.mapFromScene(QPointF(0.0, y_scene)).y())

    def paintEvent(self, event):
        painter = QPainter(self)
        try:
            painter.fillRect(self.rect(), QColor(230, 230, 230))

            if self.scene is None or self.view is None:
                return

            min_pitch = self.scene.min_pitch
            max_pitch = self.scene.max_pitch
            h_widget = self.height()
            w_widget = self.width()

            # 変換の基準を1回だけ取る
            vp0_key_y = self._viewport_origin_y_in_keyboard()          # keyboard座標
            y0_vp = self._y0_vp_for_max_pitch()                        # viewport座標
            pitch_h = self._pitch_height_in_view_pixels()              # px / pitch（float）

            if pitch_h <= 0.5:
                return  # 極端に小さい場合は描画しない（任意）

            font_size = max(7, min(12, int(round(pitch_h * 0.55))))
            painter.setFont(QFont("Consolas", font_size))

            for pitch in range(max_pitch, min_pitch - 1, -1):
                idx = (max_pitch - pitch)  # max_pitchが上、下へ行くほど増える

                y_top_f = vp0_key_y + y0_vp + idx * pitch_h
                y_bottom_f = y_top_f + pitch_h

                y_top = int(round(y_top_f))
                y_bottom = int(round(y_bottom_f))

                if y_bottom < 0 or y_top > h_widget:
                    continue
                if y_bottom <= y_top:
                    continue

                y_top_clamped = max(0, y_top)
                y_bottom_clamped = min(h_widget, y_bottom)
                height = y_bottom_clamped - y_top_clamped
                if height <= 0:
                    continue

                rect = QRect(0, y_top_clamped, w_widget, height)

                pc = pitch % 12
                is_black = pc in (1, 3, 6, 8, 10)
                bg = QColor(40, 40, 40) if is_black else QColor(255, 255, 255)
                fg = QColor(255, 255, 255) if is_black else QColor(0, 0, 0)

                painter.fillRect(rect, bg)
                painter.setPen(QColor(180, 180, 180))
                painter.drawRect(rect)

                painter.setPen(fg)
                painter.drawText(rect.adjusted(2, 0, -2, 0),
                                Qt.AlignVCenter | Qt.AlignLeft,
                                self._pitch_label(pitch))
        finally:
            painter.end()

    def pitch_at_y(self, y: int) -> Optional[int]:
        if self.scene is None or self.view is None:
            return None

        min_pitch = self.scene.min_pitch
        max_pitch = self.scene.max_pitch

        vp0_key_y = self._viewport_origin_y_in_keyboard()
        y0_vp = self._y0_vp_for_max_pitch()
        pitch_h = self._pitch_height_in_view_pixels()
        if pitch_h <= 0.5:
            return None

        # keyboard座標 -> viewport座標へ戻す
        y_vp = float(y) - vp0_key_y

        idx_f = (y_vp - y0_vp) / pitch_h
        idx = int(math.floor(idx_f))
        pitch = max_pitch - idx

        if pitch < min_pitch or pitch > max_pitch:
            return None
        return pitch
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # PySide6 では position() も pos() も使えるが、ここでは int で扱いやすい pos() を使用
            y = int(round(event.position().y()))
            pitch = self.pitch_at_y(y)
            if pitch is not None:
                self.keyClicked.emit(pitch)

        # 必要なら親クラスの処理も呼ぶ
        super().mousePressEvent(event)

    def _scene_y_to_keyboard_y(self, y_scene: float) -> float:
        """
        sceneのY座標を、keyboard widget内のY座標に変換する（兄弟widget間でも安全）。
        """
        # viewport(0,0) の global 座標を keyboard ローカルに落とした起点
        vp_origin_key = self.mapFromGlobal(self.view.viewport().mapToGlobal(QPoint(0, 0)))

        # scene -> viewport 座標
        p_vp = self.view.mapFromScene(QPointF(0.0, float(y_scene)))

        # keyboardローカル = viewport起点 + viewport内座標
        return float(vp_origin_key.y() + p_vp.y())
