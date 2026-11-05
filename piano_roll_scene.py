import math
import sys
import traceback
from bisect import bisect_right
from typing import Dict, Optional

import numpy as np
from PySide6.QtCore import QCoreApplication, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QGraphicsLineItem, QGraphicsScene, QMessageBox

from DataRepresentation import TPQN
from dtpr_adapter import DTPRAdapter
from edit_log import EditOperationLogger
from piano_roll_items import GRID_SEC, NoteItem, PlayheadLineItem


def _tr(text: str) -> str:
    return QCoreApplication.translate("PianoRollScene", text)


def normalize_to_u8(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    任意の 2D/ND float 配列を 0..255 の uint8 に正規化して返す。
    可視化用なので、外れ値に強いよう percentiles を使う。
    """
    a = np.asarray(x, dtype=np.float32)
    if a.size == 0:
        return np.zeros_like(a, dtype=np.uint8)

    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    # 外れ値に強いスケーリング（必要なら調整）
    lo = np.percentile(a, 1.0)
    hi = np.percentile(a, 99.0)
    if (hi - lo) < eps:
        return np.zeros_like(a, dtype=np.uint8)

    a = np.clip(a, lo, hi)
    a = (a - lo) / (hi - lo)
    return (a * 255.0 + 0.5).astype(np.uint8)


def _to_grayscale_qimage(u8_2d: np.ndarray) -> QImage:
    """
    u8_2d: (H, W) dtype=uint8 を QImage(Grayscale8) に変換。
    """
    u8 = np.ascontiguousarray(u8_2d, dtype=np.uint8)
    h, w = u8.shape
    # bytesPerLine = w（Grayscale8 は 1pixel=1byte）
    return QImage(u8.data, w, h, w, QImage.Format_Grayscale8).copy()

# ====== ピアノロール表示 ======
MIN_NOTE_SEC = 0.02

class PianoRollScene(QGraphicsScene):
    is_piano_roll_scene = True
    notesChanged = Signal()
    notesAuditionRequested = Signal(list)
    noteItemGeometryChanged = Signal(int)
    playheadDragged = Signal(float)
    def __init__(
        self,
        parent=None,
        edit_logger: Optional[EditOperationLogger] = None,
        dtpr_adapter: Optional[DTPRAdapter] = None,
        domain: str = "ptpr",
    ):
        super().__init__(parent)
        self.edit_logger = edit_logger
        self.dtpr_adapter = dtpr_adapter
        self.domain = str(domain or "ptpr").lower()
        if self.domain not in ("ptpr", "mtpr"):
            raise ValueError(f"Unsupported domain: {domain}")

        display_notes = self._get_display_notes()

        # 時間軸: シーン座標で 1px = 0.01秒（10ms）
        self.seconds_per_pixel = 0.01

        # ピッチ軸
        self.px_per_pitch = 14
        self.min_pitch = 21            # A0
        self.max_pitch = 108           # C8

        if display_notes:
            self.next_note_id = max(self._note_id(n) for n in display_notes) + 1
        else:
            self.next_note_id = 0

        # ★ Undo / Redo 用の履歴管理
        self.undo_stack: list[list[dict]] = []
        self.redo_stack: list[list[dict]] = []
        self.transaction_active: bool = False
        self.transaction_snapshot: list[dict] | None = None
        self.transaction_log_snapshot: list[dict] | None = None
        self.transaction_operation = None  # type: Optional[str]
        self.transaction_meta = None       # type: Optional[dict]
        self.transaction_candidate_meta = None  # type: Optional[dict]

        # シーンサイズ
        self.total_time = max((self._note_offset(n) for n in display_notes), default=10.0)
        self.scene_width = self.time_to_x(self.total_time + 1.0)
        self.scene_height = (self.max_pitch - self.min_pitch + 1) * self.px_per_pitch

        self.setBackgroundBrush(QColor(255, 255, 255))
        self.setSceneRect(0, 0, self.scene_width, self.scene_height)

        self.note_items: Dict[int, NoteItem] = {}
        self.note_editing_enabled = True
        self.beat_edit_mode = False
        self.beat_insert_callback = None

        # ★ プレイヘッド（再生位置の縦線）
        self.playhead_item: Optional[QGraphicsLineItem] = None
        self.populate_notes()
        self.playhead_draggable = True

        self.show_spectrogram = False
        self.show_chroma = False
        self.show_pitch_prob = False

        # スペクトログラム背景（表示用キャッシュ）
        self._spec_u8 = None         # np.uint8 (H,W) を保持して QImage の寿命を担保
        self._spec_img = None        # QImage
        self._spec_hop_sec = None
        self._spec_x_edges = None
        self._spec_pitch_min = 21
        self._spec_pitch_max = 108

        # クロマ背景
        self._chroma_u8 = None
        self._chroma_img = None
        self._chroma_hop_sec = None
        self._chroma_x_edges = None
        self._chroma_pitch_min = 21
        self._chroma_pitch_max = 108

        # ピッチ確率(0..127)背景
        self._pitch_prob_u8 = None
        self._pitch_prob_img = None
        self._pitch_prob_hop_sec = None
        self._pitch_prob_pitch_min = 0
        self._pitch_prob_pitch_max = 127
        self._pitch_prob_x_edges = None

    # --- 座標変換 ---

    def time_to_x(self, t: float) -> float:
        return t / self.seconds_per_pixel

    def dur_to_w(self, d: float) -> float:
        return max(3.0, d / self.seconds_per_pixel)

    def pitch_to_y(self, pitch: int) -> float:
        return (self.max_pitch - pitch) * self.px_per_pitch

    def _is_ptpr(self) -> bool:
        return self.domain == "ptpr"

    def _time_grid_step(self) -> float:
        if self._is_ptpr():
            return float(GRID_SEC)
        return 1.0

    def x_to_domain_time(self, x: float) -> float:
        return float(x) * float(self.seconds_per_pixel)

    def domain_time_to_x(self, t: float) -> float:
        if self.seconds_per_pixel <= 0:
            return float(t)
        return float(t) / float(self.seconds_per_pixel)

    def snap_domain_time(self, t: float) -> float:
        if self._is_ptpr():
            return self._quantize_sec(float(t), GRID_SEC)
        step = float(self._time_grid_step())
        if step <= 0:
            return float(t)
        return self._quantize_sec(float(t), step)

    def snap_scene_x(self, x: float) -> float:
        return self.domain_time_to_x(self.snap_domain_time(self.x_to_domain_time(x)))

    def min_note_width_px(self) -> float:
        return max(1.0, self.domain_time_to_x(self._domain_min_note_len()))

    def _get_display_notes(self):
        return self.dtpr_adapter.get_notes(domain=self.domain)

    def _get_display_note_by_id(self, note_id: int):
        return self.dtpr_adapter.get_note_by_id(self.domain, int(note_id))

    @staticmethod
    def _note_id(note) -> int:
        return int(getattr(note, "id"))

    @staticmethod
    def _note_pitch(note) -> int:
        return int(getattr(note, "pitch"))

    def _note_onset(self, note) -> float:
        if self._is_ptpr():
            if hasattr(note, "real_ontime"):
                return float(note.real_ontime)
            return float(note.onset)
        if hasattr(note, "abs_ontick"):
            return float(note.abs_ontick)
        return float(getattr(note, "onset", 0.0))

    def _note_offset(self, note) -> float:
        if self._is_ptpr():
            if hasattr(note, "real_offtime"):
                return float(note.real_offtime)
            return float(note.offset)
        if hasattr(note, "abs_offtick"):
            return float(note.abs_offtick)
        return float(getattr(note, "offset", 0.0))

    def _note_duration(self, note) -> float:
        return max(0.0, self._note_offset(note) - self._note_onset(note))

    def set_total_time(self, total_time: float) -> None:
        total_time = max(0.0, float(total_time))
        self.total_time = total_time
        self.scene_width = self.time_to_x(self.total_time + 1.0)
        self.setSceneRect(0, 0, self.scene_width, self.scene_height)

    def _draw_black_key_rows(self, painter: QPainter, rect: QRectF) -> None:
        black_key_classes = {1, 3, 6, 8, 10}
        row_color = QColor(70, 135, 220, 28)
        for pitch in range(self.min_pitch, self.max_pitch + 1):
            if pitch % 12 not in black_key_classes:
                continue
            y = float(self.pitch_to_y(pitch))
            if y + self.px_per_pitch < rect.top() or y > rect.bottom():
                continue
            painter.fillRect(
                QRectF(rect.left(), y, rect.width(), self.px_per_pitch),
                row_color,
            )

    # --- 背景（グリッド）描画 ---

    def drawBackground(self, painter, rect: QRectF):
        super().drawBackground(painter, rect)
        painter.save()

        self._draw_feature_background(painter, rect)
        self._draw_black_key_rows(painter, rect)

        left = rect.left()
        right = rect.right()
        top = rect.top()
        bottom = rect.bottom()

        # ピッチ方向の横線（半音ごと）
        pitch_pen = QPen(QColor(230, 230, 230), 0)
        for p in range(self.min_pitch, self.max_pitch + 1):
            y = self.pitch_to_y(p)
            if y < top - self.px_per_pitch or y > bottom + self.px_per_pitch:
                continue
            painter.setPen(pitch_pen)
            painter.drawLine(left, y, right, y)

        painter.restore()

    def set_playhead_time(self, t: float):
        """
        再生経過時間 t [sec] に対応する x 座標に縦線を描く。
        """
        if t < 0:
            t = 0.0
        x = t / self.seconds_per_pixel

        # シーンの縦方向範囲
        top = self.pitch_to_y(self.max_pitch)
        bottom = self.pitch_to_y(self.min_pitch) + self.px_per_pitch

        if self.playhead_item is None or self.playhead_item.scene() is None:
            self.playhead_item = PlayheadLineItem(self)
            pen = QPen(QColor(255, 200, 0))  # 黄色線
            pen.setWidth(1)
            self.playhead_item.setPen(pen)
            self.addItem(self.playhead_item)

        self.playhead_item.setLine(x, top, x, bottom)

    # --- ノート描画 ---

    def populate_notes(self):
        for it in list(self.note_items.values()):
            try:
                self.removeItem(it)
            except Exception:
                pass
        self.note_items.clear()

        for note in self._get_display_notes():
            x = self.time_to_x(self._note_onset(note))
            w = self.dur_to_w(self._note_duration(note))
            y = self.pitch_to_y(self._note_pitch(note))
            h = self.px_per_pitch
            item = NoteItem(note, x, y, w, h)
            item.set_checked(bool(getattr(note, "checked", False)))
            item.setEnabled(bool(self.note_editing_enabled))
            self.addItem(item)
            self.note_items[self._note_id(note)] = item

    def _quantize_sec(self, t: float, grid: float = GRID_SEC) -> float:
        # round() の銀行丸めが気になる場合に備え、half-up に寄せる
        return math.floor(t / grid + 0.5) * grid

    def _quantize_domain_time(self, t: float) -> float:
        step = float(self._time_grid_step())
        if step <= 0:
            return float(t)
        return self._quantize_sec(float(t), step)

    def _domain_min_note_len(self) -> float:
        if self._is_ptpr():
            return float(MIN_NOTE_SEC)
        return float(max(1.0, self._time_grid_step()))

    def _mtpr_split_min_ticks(self) -> int:
        dtpr = getattr(self.dtpr_adapter, "dtpr", None)
        if dtpr is None:
            return max(1, TPQN // 4)
        return max(1, int(getattr(dtpr, "quantize_delta", TPQN // 4)))

    def _mtpr_split_tick(self, split_time: float) -> Optional[int]:
        if not self._is_ptpr():
            return int(round(float(split_time)))

        dtpr = getattr(self.dtpr_adapter, "dtpr", None)
        if dtpr is None:
            return None
        try:
            real_split_time = float(split_time)
            beat_id = dtpr.BDB.get_beat_id_from_time(real_split_time)
            _ideal_time, abs_tick, _beat_id, _metrical_pos = dtpr.real_time_to_ideal_time(
                real_split_time,
                beat_id,
                self._mtpr_split_min_ticks(),
            )
            return int(abs_tick)
        except Exception:
            return None

    def can_split_note_at(self, note, split_time: float) -> bool:
        note_id = self._note_id(note)
        mtpr_note = self.dtpr_adapter.get_mtpr_note_by_id(note_id)
        split_tick = self._mtpr_split_tick(split_time)
        if mtpr_note is None or split_tick is None:
            return False

        min_ticks = self._mtpr_split_min_ticks()
        return (
            split_tick - int(mtpr_note.abs_ontick) >= min_ticks
            and int(mtpr_note.abs_offtick) - split_tick >= min_ticks
        )

    def _collect_note_item_values(self) -> dict[int, dict]:
        values = {}
        for note_id, item in list(self.note_items.items()):
            pos = item.pos()
            rect = item.rect()
            onset_raw = float((pos.x() + rect.x()) * self.seconds_per_pixel)
            offset_raw = float((pos.x() + rect.x() + rect.width()) * self.seconds_per_pixel)
            onset_q = self._quantize_domain_time(max(0.0, onset_raw))
            offset_q = self._quantize_domain_time(max(0.0, offset_raw))
            min_len = self._domain_min_note_len()
            if offset_q < onset_q + min_len:
                offset_q = onset_q + min_len
            pitch_f = self.max_pitch - float((pos.y() + rect.y()) / self.px_per_pitch)
            pitch = int(round(pitch_f))
            pitch = max(self.min_pitch, min(self.max_pitch, pitch))
            values[int(note_id)] = {
                "onset": float(onset_q),
                "offset": float(offset_q),
                "pitch": int(pitch),
            }
        return values

    def _apply_neighbor_time_constraints(self, values: dict[int, dict], target_ids: set[int], op: str) -> set[int]:
        changed_ids: set[int] = set()
        if not target_ids:
            return changed_ids
        if op not in {"move", "resizeOnset", "resizeOffset"}:
            return changed_ids

        ordered_ids = sorted(
            values.keys(),
            key=lambda nid: (float(values[nid]["onset"]), float(values[nid]["offset"]), int(nid)),
        )
        pos_by_id = {nid: i for i, nid in enumerate(ordered_ids)}

        for note_id in sorted(target_ids):
            if note_id not in values or note_id not in pos_by_id:
                continue
            idx = pos_by_id[note_id]

            if op in {"move", "resizeOnset"} and idx > 0:
                prev_id = ordered_ids[idx - 1]
                if prev_id in values:
                    min_prev_offset = float(values[prev_id]["onset"]) + float(self._domain_min_note_len())
                    if float(values[note_id]["onset"]) < min_prev_offset:
                        values[note_id]["onset"] = min_prev_offset
                        if float(values[note_id]["offset"]) < min_prev_offset:
                            values[note_id]["offset"] = min_prev_offset
                        changed_ids.add(int(note_id))
                    if float(values[note_id]["onset"]) < float(values[prev_id]["offset"]):
                        # Onset の移動を優先し、前ノートの offset を現在ノートの onset に合わせる。
                        # ただし前ノートの音長が0にならない下限は維持する。
                        values[prev_id]["offset"] = max(
                            float(values[note_id]["onset"]),
                            min_prev_offset,
                        )
                        changed_ids.add(int(prev_id))

            if op in {"move", "resizeOffset"} and idx < len(ordered_ids) - 1:
                next_id = ordered_ids[idx + 1]
                if next_id in values and float(values[note_id]["offset"]) > float(values[next_id]["onset"]):
                    # Offset は次ノート onset まで。等しい時刻は許可する。
                    values[note_id]["offset"] = float(values[next_id]["onset"])
                    changed_ids.add(int(note_id))

        return changed_ids

    @staticmethod
    def _select_insert_offset_target(
        provisional_abs_offtick: int,
        next_abs_ontick: int,
    ) -> tuple[int, str]:
        """quantize_offsetと同じTPQN/2基準を挿入音符のoffset選択へ適用する。"""
        provisional = int(provisional_abs_offtick)
        next_onset = int(next_abs_ontick)
        if next_onset - provisional < TPQN / 2:
            return next_onset, "next_onset"
        if provisional % TPQN == 0:
            return provisional, "beat"

        next_beat = provisional - (provisional % TPQN) + TPQN
        if next_onset <= next_beat:
            return next_onset, "next_onset"
        return next_beat, "beat"

    def _mtpr_insert_delta(self) -> int:
        dtpr = getattr(getattr(self, "dtpr_adapter", None), "dtpr", None)
        if dtpr is None:
            return max(1, TPQN // 4)
        return max(1, int(getattr(dtpr, "quantize_delta", TPQN // 4)))

    def _insert_onsets_for_all_domains(
        self,
        onset: float,
    ) -> tuple[Optional[dict[str, float]], Optional[str]]:
        dtpr = getattr(getattr(self, "dtpr_adapter", None), "dtpr", None)
        if dtpr is None:
            return None, _tr("Cannot insert a note because DTPR is not initialized.")

        try:
            if self._is_ptpr():
                real_on = float(onset)
                on_beat_id = dtpr.BDB.get_beat_id_from_time(real_on)
                _ideal_on, abs_on, _on_beat_id, _on_rel = dtpr.real_time_to_ideal_time(
                    real_on,
                    on_beat_id,
                    self._mtpr_insert_delta(),
                )
                return {
                    "ptpr": real_on,
                    "mtpr": float(abs_on),
                }, None

            abs_on = int(round(float(onset)))
            on_beat_id, on_rel = dtpr.abs_tick_to_rel_tick(abs_on)
            ideal_on = float(dtpr.rel_tick_to_time(on_beat_id, on_rel))
            return {
                "ptpr": ideal_on,
                "mtpr": float(abs_on),
            }, None
        except Exception as e:
            return None, _tr(
                "Cannot insert the note because its position could not be converted to the other time domain: {error}"
            ).format(error=e)

    def _prepare_insert_ranges(
        self,
        onset: float,
    ) -> tuple[Optional[dict[str, tuple[float, float]]], Optional[str]]:
        onsets, error_message = self._insert_onsets_for_all_domains(onset)
        if error_message is not None or onsets is None:
            return None, error_message
        if self.dtpr_adapter is None:
            return None, _tr("Cannot insert a note because DTPR is not initialized.")

        abs_on = int(round(onsets["mtpr"]))
        provisional_abs_off = abs_on + self._mtpr_insert_delta()
        next_mtpr_note = None
        for candidate in self.dtpr_adapter.get_mtpr_notes_sorted():
            if int(candidate.abs_ontick) >= abs_on:
                next_mtpr_note = candidate
                break

        target_kind = "tail"
        target_abs_off = provisional_abs_off
        if next_mtpr_note is not None:
            target_abs_off, target_kind = self._select_insert_offset_target(
                provisional_abs_off,
                int(next_mtpr_note.abs_ontick),
            )

        if target_abs_off <= abs_on:
            return None, _tr(
                "Cannot insert the note because its length in the Metrical-time domain would be zero."
            )

        real_on = float(onsets["ptpr"])
        if target_kind == "next_onset" and next_mtpr_note is not None:
            next_ptpr_note = self.dtpr_adapter.get_ptpr_note_by_id(int(next_mtpr_note.id))
            if next_ptpr_note is None:
                return None, _tr(
                    "Cannot insert the note because no corresponding Physical-time note was found for the next note."
                )
            real_off = float(next_ptpr_note.real_ontime)
        else:
            real_off = float(self.dtpr_adapter.dtpr.abs_tick_to_time(target_abs_off))

        return {
            "ptpr": (real_on, real_off),
            "mtpr": (float(abs_on), float(target_abs_off)),
        }, None

    def _collect_insert_overlap_adjustments_all_domains(
        self,
        ranges: dict[str, tuple[float, float]],
    ) -> tuple[Optional[dict[str, list[dict]]], Optional[str]]:
        if self.dtpr_adapter is None:
            return None, _tr("Cannot insert a note because DTPR is not initialized.")

        insert_real_on = float(ranges["ptpr"][0])
        insert_abs_on = int(round(ranges["mtpr"][0]))
        insert_abs_off = int(round(ranges["mtpr"][1]))
        adjustments_by_domain: dict[str, list[dict]] = {"ptpr": [], "mtpr": []}

        for mtpr_note in self.dtpr_adapter.get_mtpr_notes_sorted():
            note_onset = int(mtpr_note.abs_ontick)
            note_offset = int(mtpr_note.abs_offtick)
            if note_offset <= insert_abs_on or note_onset >= insert_abs_off:
                continue
            if insert_abs_on <= note_onset:
                return None, _tr(
                    "The insertion position overlaps an existing Metrical-time note, "
                    "and shortening that note would reduce its length to zero."
                )

            note_id = int(mtpr_note.id)
            ptpr_note = self.dtpr_adapter.get_ptpr_note_by_id(note_id)
            if ptpr_note is None:
                return None, _tr(
                    "Cannot insert the note because no corresponding Physical-time note was found."
                )
            adjustments_by_domain["mtpr"].append(
                {
                    "id": note_id,
                    "pitch": int(mtpr_note.pitch),
                    "onset": float(note_onset),
                    "offset": float(insert_abs_on),
                }
            )
            adjustments_by_domain["ptpr"].append(
                {
                    "id": note_id,
                    "pitch": int(ptpr_note.pitch),
                    "onset": float(ptpr_note.real_ontime),
                    "offset": insert_real_on,
                }
            )

        return adjustments_by_domain, None

    def _apply_insert_overlap_adjustments_for_domain(self, domain: str, adjustments: list[dict]) -> None:
        d = str(domain or self.domain).lower()
        for adj in adjustments:
            note_id = int(adj["id"])
            pitch = int(adj["pitch"])
            onset = float(adj["onset"])
            offset = float(adj["offset"])

            if d == "ptpr":
                self.dtpr_adapter.ptpr_note_offtime_shift(
                    id=note_id,
                    real_offtime=offset,
                    update_ideal=False,
                )
                self.dtpr_adapter.ptpr_note_move(
                    id=note_id,
                    pitch=pitch,
                    real_ontime=onset,
                    real_offtime=offset,
                    update_ideal=False,
                )
            else:
                on_tick = int(round(onset))
                off_tick = int(round(offset))
                self.dtpr_adapter.mtpr_note_offtime_shift(
                    id=note_id,
                    abs_offtick=off_tick,
                    keep_real=True,
                )
                self.dtpr_adapter.mtpr_note_move(
                    id=note_id,
                    pitch=pitch,
                    abs_ontick=on_tick,
                    abs_offtick=off_tick,
                    keep_real=True,
                )

    def _apply_insert_overlap_adjustments_all_domains(self, adjustments_by_domain: dict[str, list[dict]]) -> None:
        self._apply_insert_overlap_adjustments_for_domain("ptpr", adjustments_by_domain.get("ptpr", []))
        self._apply_insert_overlap_adjustments_for_domain("mtpr", adjustments_by_domain.get("mtpr", []))

    def sync_back_to_model(self):
        op = str(self.transaction_operation or "")
        values = self._collect_note_item_values()
        target_ids = {
            int(nid)
            for nid in (self.transaction_meta or {}).get("noteIds", [])
            if int(nid) in values
        }
        constrained_ids = self._apply_neighbor_time_constraints(values, target_ids, op)
        if self._is_ptpr():
            sync_ids = set(values.keys())
        else:
            # MTPR側で全ノートを書き戻すと、keep_real=False 経由でPTPR real timeまで
            # 全量子化されるため、操作対象と制約で連動変更されたノートだけ同期する。
            sync_ids = {
                int(nid)
                for nid in (set(target_ids) | set(constrained_ids))
                if int(nid) in values
            }

        for note_id in sorted(sync_ids):
            item_values = values[note_id]
            onset_q = float(item_values["onset"])
            offset_q = float(item_values["offset"])
            pitch = int(item_values["pitch"])

            if self._is_ptpr():
                if op == "resizeOnset":
                    self.dtpr_adapter.ptpr_note_ontime_shift(
                        id=int(note_id),
                        real_ontime=float(onset_q),
                        update_ideal=False,
                    )
                    self.dtpr_adapter.ptpr_note_move(
                        id=int(note_id),
                        pitch=pitch,
                        real_ontime=float(onset_q),
                        real_offtime=float(offset_q),
                        update_ideal=False,
                    )
                elif op == "resizeOffset":
                    self.dtpr_adapter.ptpr_note_offtime_shift(
                        id=int(note_id),
                        real_offtime=float(offset_q),
                        update_ideal=False,
                    )
                    self.dtpr_adapter.ptpr_note_move(
                        id=int(note_id),
                        pitch=pitch,
                        real_ontime=float(onset_q),
                        real_offtime=float(offset_q),
                        update_ideal=False,
                    )
                else:
                    self.dtpr_adapter.ptpr_note_move(
                        id=int(note_id),
                        pitch=pitch,
                        real_ontime=float(onset_q),
                        real_offtime=float(offset_q),
                        update_ideal=False,
                    )
            else:
                on_tick = int(round(onset_q))
                off_tick = int(round(offset_q))
                if op == "resizeOnset":
                    self.dtpr_adapter.mtpr_note_ontime_shift(
                        id=int(note_id),
                        abs_ontick=on_tick,
                        keep_real=False,
                    )
                    self.dtpr_adapter.mtpr_note_move(
                        id=int(note_id),
                        pitch=pitch,
                        abs_ontick=on_tick,
                        abs_offtick=off_tick,
                        keep_real=False,
                    )
                elif op == "resizeOffset":
                    self.dtpr_adapter.mtpr_note_offtime_shift(
                        id=int(note_id),
                        abs_offtick=off_tick,
                        keep_real=False,
                    )
                    self.dtpr_adapter.mtpr_note_move(
                        id=int(note_id),
                        pitch=pitch,
                        abs_ontick=on_tick,
                        abs_offtick=off_tick,
                        keep_real=False,
                    )
                else:
                    self.dtpr_adapter.mtpr_note_move(
                        id=int(note_id),
                        pitch=pitch,
                        abs_ontick=on_tick,
                        abs_offtick=off_tick,
                        keep_real=False,
                    )
        return

    def delete_selected_notes(self):
        selected = [item for item in self.selectedItems() if isinstance(item, NoteItem)]
        if not selected:
            return

        selected_ids = [int(item.note_id) for item in selected]

        # 履歴開始
        self.start_edit_transaction(operation="delete")
        try:
            for note_id in selected_ids:
                self.dtpr_adapter.note_delete(int(note_id), domain=self.domain)
        except Exception as e:
            print(f"[DTPR delete] failed: {e}", file=sys.stderr)
            traceback.print_exc()

        for item in selected:
            note_id = item.note_id
            self.removeItem(item)
            if note_id in self.note_items:
                del self.note_items[note_id]
        
        # 履歴確定
        self.commit_edit_transaction()

    def add_note_at_scene_pos(self, scene_x: float, scene_y: float):
        """
        シーン座標 (scene_x, scene_y) に現在のMTPRグリッドを基準としたノートを挿入する。
        ピッチは半音グリッドにスナップする。
        """
        # 履歴開始
        self.start_edit_transaction(operation="insert", meta={"domain": self.domain})
        # ---- 時間（オンセット） ----
        onset_val = scene_x * self.seconds_per_pixel
        onset_val = self._quantize_domain_time(onset_val)
        if onset_val < 0.0:
            onset_val = 0.0

        # ---- ピッチ ----
        pitch_f = self.max_pitch - scene_y / self.px_per_pitch
        pitch = int(round(pitch_f))
        pitch = max(self.min_pitch, min(self.max_pitch, pitch))

        # ---- Note モデルを作成 ----
        note = None
        rollback_ptpr_state = None
        rollback_mtpr_state = None
        try:
            ranges, error_message = self._prepare_insert_ranges(onset_val)
            if error_message is not None or ranges is None:
                QMessageBox.information(
                    None,
                    _tr("Cannot Insert"),
                    error_message or _tr("The insertion time could not be determined."),
                )
                self.commit_edit_transaction()
                return

            adjustments, error_message = self._collect_insert_overlap_adjustments_all_domains(ranges)
            if error_message is not None:
                QMessageBox.information(None, _tr("Cannot Insert"), error_message)
                self.commit_edit_transaction()
                return

            rollback_ptpr_state = self.dtpr_adapter.snapshot_state(domain="ptpr")
            rollback_mtpr_state = self.dtpr_adapter.snapshot_state(domain="mtpr")
            self._apply_insert_overlap_adjustments_all_domains(adjustments or {})

            mtpr_note = self.dtpr_adapter.mtpr_note_insert(
                pitch=pitch,
                abs_ontick=int(round(ranges["mtpr"][0])),
                abs_offtick=int(round(ranges["mtpr"][1])),
            )
            if mtpr_note is not None:
                note_id = int(mtpr_note.id)
                self.dtpr_adapter.ptpr_note_move_by_id(
                    note_id=note_id,
                    pitch=pitch,
                    real_ontime=float(ranges["ptpr"][0]),
                    real_offtime=float(ranges["ptpr"][1]),
                    update_ideal=False,
                )
                note = self.dtpr_adapter.get_note_by_id(self.domain, note_id)
        except Exception as e:
            print(f"[DTPR insert] failed: {e}", file=sys.stderr)
            traceback.print_exc()
            if rollback_ptpr_state is not None and rollback_mtpr_state is not None:
                self.dtpr_adapter.restore_state("ptpr", rollback_ptpr_state)
                self.dtpr_adapter.restore_state("mtpr", rollback_mtpr_state)
                rollback_ptpr_state = None
                rollback_mtpr_state = None

        if note is None:
            if rollback_ptpr_state is not None and rollback_mtpr_state is not None:
                self.dtpr_adapter.restore_state("ptpr", rollback_ptpr_state)
                self.dtpr_adapter.restore_state("mtpr", rollback_mtpr_state)
            self.commit_edit_transaction()
            return
        self.next_note_id = max(self.next_note_id, int(note.id) + 1)

        self.populate_notes()

        # シーンの横幅が足りなければ広げる
        if self._note_offset(note) > self.total_time:
            self.total_time = self._note_offset(note)
            self.scene_width = self.time_to_x(self.total_time + 1.0)
            self.setSceneRect(0, 0, self.scene_width, self.scene_height)
        # 履歴確定
        self.commit_edit_transaction()

    def mouseDoubleClickEvent(self, event):
        """
        左ボタンのダブルクリックで、その位置にノートを挿入。
        """
        if self._is_ptpr() and self.beat_edit_mode:
            if event.button() == Qt.LeftButton and callable(self.beat_insert_callback):
                scene_pos = event.scenePos()
                self.beat_insert_callback(self.x_to_domain_time(scene_pos.x()))
            return
        if event.button() == Qt.LeftButton:
            scene_pos = event.scenePos()
            self.add_note_at_scene_pos(scene_pos.x(), scene_pos.y())
            # イベントを消費して終了
            return
        super().mouseDoubleClickEvent(event)
    
    def join_selected_notes(self):
        """
        選択中のノートを結合する。
        条件:
          - 2つ以上選択されている
        結合後:
          - pitch, velocity, track は「最初のノート」を引き継ぐ
          - onset = 最初の onset, offset = 最後の offset
        """
        # 選択されている NoteItem を取得
        selected_items = [it for it in self.selectedItems() if isinstance(it, NoteItem)]
        if len(selected_items) < 2:
            QMessageBox.information(
                None,
                _tr("Cannot Merge"),
                _tr("Select at least two notes to merge."),
            )
            return

        # ★ 履歴開始
        selected_ids = [int(item.note_id) for item in selected_items]
        self.start_edit_transaction(
            operation="join",
            meta={"selected_ids": selected_ids},
        )
        merged_id = None
        try:
            merged_id = self.dtpr_adapter.note_join(selected_ids, domain=self.domain)
        except Exception as e:
            print(f"[DTPR join] failed: {e}", file=sys.stderr)
            traceback.print_exc()
        if merged_id is None:
            self.commit_edit_transaction()
            return

        self.populate_notes()
        self.clearSelection()
        merged_item = self.note_items.get(int(merged_id))
        if merged_item is not None:
            merged_item.setSelected(True)

        self.total_time = max((self._note_offset(n) for n in self._get_display_notes()), default=self.total_time)
        self.scene_width = self.time_to_x(self.total_time + 1.0)
        self.setSceneRect(0, 0, self.scene_width, self.scene_height)
        self.commit_edit_transaction()
        return


    def split_note_at(self, note, t_split: float):
        """
        1つのノートを、指定した領域時刻で2つに分割する。
        分割後の両音符がMT領域の最小長以上になる場合だけ分割する。
        """
        t_split = self._quantize_domain_time(float(t_split))
        note_id = self._note_id(note)
        note_onset = self._note_onset(note)
        note_offset = self._note_offset(note)

        if (
            t_split <= note_onset
            or t_split >= note_offset
            or not self.can_split_note_at(note, t_split)
        ):
            QMessageBox.information(
                None,
                _tr("Cannot Split"),
                _tr(
                    "Cannot split because one of the resulting notes would be shorter "
                    "than the minimum Metrical-time length ({ticks} ticks)."
                ).format(ticks=self._mtpr_split_min_ticks()),
            )
            return

        self.start_edit_transaction(
            operation="split",
            meta={"source_id": note_id, "split_time": t_split},
        )

        # 元ノートに対応する NoteItem を削除
        split_ids = None
        try:
            if self._is_ptpr():
                split_ids = self.dtpr_adapter.ptpr_note_split(
                    id=note_id,
                    real_split_time=float(t_split),
                )
            else:
                split_ids = self.dtpr_adapter.mtpr_note_split(
                    id=note_id,
                    abs_split_tick=int(round(t_split)),
                )
        except Exception as e:
            print(f"[DTPR split] failed: {e}", file=sys.stderr)
            traceback.print_exc()
        if not split_ids or len(split_ids) < 2:
            self.commit_edit_transaction()
            return

        self.populate_notes()
        self.clearSelection()
        for sid in split_ids:
            split_item = self.note_items.get(int(sid))
            if split_item is not None:
                split_item.setSelected(True)

        self.total_time = max(
            (self._note_offset(n) for n in self._get_display_notes()),
            default=self.total_time,
        )
        self.scene_width = self.time_to_x(self.total_time + 1.0)
        self.setSceneRect(0, 0, self.scene_width, self.scene_height)
        self.commit_edit_transaction()
        return

    def _audition_note_ids_from_transaction(
        self,
        before_notes: dict[str, list[dict]],
        after_notes: dict[str, list[dict]],
    ) -> list[int]:
        operation = str(self.transaction_operation or "")
        if not operation or operation == "delete" or operation.startswith("beat_"):
            return []

        audition_operations = {
            "insert",
            "move",
            "resizeOnset",
            "resizeOffset",
            "join",
            "split",
            "mtpr_onset_candidate_apply",
            "mtpr_offset_candidate_apply",
            "mtpr_split_candidate_apply",
            "mtpr_pitch_candidate_apply",
            "ptpr_onset_candidate_apply",
            "split_candidate_apply",
            "pitch_candidate_apply",
        }
        if operation not in audition_operations:
            return []

        domain = (
            str((self.transaction_meta or {}).get("domain", self.domain)).lower()
            if self.transaction_meta is not None
            else self.domain
        )
        if domain not in ("ptpr", "mtpr"):
            domain = self.domain

        before_by_id = {
            int(item["id"]): dict(item)
            for item in before_notes.get(domain, [])
            if "id" in item
        }
        after_by_id = {
            int(item["id"]): dict(item)
            for item in after_notes.get(domain, [])
            if "id" in item
        }

        target_ids = [
            int(nid)
            for nid in (self.transaction_meta or {}).get("noteIds", [])
            if int(nid) in after_by_id
        ]
        source_id = (self.transaction_meta or {}).get("source_id")
        if source_id is not None:
            target_ids.append(int(source_id))
        selected_ids = [
            int(nid)
            for nid in (self.transaction_meta or {}).get("selected_ids", [])
            if int(nid) in after_by_id
        ]
        target_ids.extend(selected_ids)

        changed_ids = []
        new_ids = []
        for note_id, after_item in after_by_id.items():
            before_item = before_by_id.get(note_id)
            if before_item is None:
                new_ids.append(note_id)
                changed_ids.append(note_id)
            elif before_item != after_item:
                changed_ids.append(note_id)

        ordered_ids = []
        for note_id in target_ids + new_ids + changed_ids:
            if note_id in after_by_id and note_id not in ordered_ids:
                ordered_ids.append(note_id)

        return ordered_ids

    # ==== 履歴トランザクション管理 ====

    def start_edit_transaction(
        self,
        operation: Optional[str] = None,
        *,
        meta: Optional[dict] = None,
        candidate_meta=None
        ):
        """
        1つのユーザー操作（ドラッグ・挿入・削除・結合・分割など）開始時に呼ぶ。
        現在の PTPR をスナップショットとして保持。
        """
        if self.transaction_active:
            return  # すでに開始されている場合は何もしない

        # 現在の描画状態をモデルに反映しておく（ずれていると困るため）
        self.sync_back_to_model()

        meta_with_domain = dict(meta or {})
        meta_with_domain.setdefault("domain", self.domain)

        # DTPR全体を1つの編集単位として保存する
        self.transaction_snapshot = self.dtpr_adapter.snapshot_dtpr_state(include_checked=False)
        self.transaction_log_snapshot = self.dtpr_adapter.snapshot_edit_log_state()
        self.transaction_active = True

        self.transaction_operation = operation
        self.transaction_meta = meta_with_domain
        self.transaction_candidate_meta = candidate_meta

    def commit_edit_transaction(self):
        """
        編集終了時に呼ぶ。transaction_snapshot を undo_stack に積む。
        """
        if not self.transaction_active:
            return

        # 編集後の状態をモデルに反映
        self.sync_back_to_model()

        before_notes = self.transaction_snapshot or {"ptpr": [], "mtpr": []}
        after_notes = self.dtpr_adapter.snapshot_dtpr_state(include_checked=False)
        before_log_notes = self.transaction_log_snapshot or []
        after_log_notes = self.dtpr_adapter.snapshot_edit_log_state()
        audition_note_ids = self._audition_note_ids_from_transaction(before_notes, after_notes)

        # No-op click (selection only): keep selection state and avoid needless refresh.
        if after_notes == before_notes:
            self.transaction_snapshot = None
            self.transaction_log_snapshot = None
            self.transaction_active = False
            self.transaction_operation = None
            self.transaction_meta = None
            self.transaction_candidate_meta = None
            return

        op_id = None
        if self.edit_logger is not None:
            try:
                if str(self.transaction_operation or "").startswith("beat_"):
                    op_id = self.edit_logger.record_note_patches_from_before_after(
                        before_notes=before_log_notes,
                        after_notes=after_log_notes,
                        operation_hint=self.transaction_operation,
                        meta=self.transaction_meta,
                        candidate_meta=self.transaction_candidate_meta,
                    )
                else:
                    op_id = self.edit_logger.record_from_before_after(
                        before_notes=before_log_notes,
                        after_notes=after_log_notes,
                        operation_hint=self.transaction_operation,  # fallback
                        meta=self.transaction_meta,
                        candidate_meta=self.transaction_candidate_meta,
                    )
            except Exception as e:
                print(f"[edit logger] record edit operation failed: {e}", file=sys.stderr)
                traceback.print_exc()

        self.undo_stack.append({
            "state": before_notes,
            "op_id": op_id,
            "domain": self.transaction_meta.get("domain", self.domain) if self.transaction_meta else self.domain,
        })
        self.redo_stack.clear()

        self.transaction_snapshot = None
        self.transaction_log_snapshot = None
        self.transaction_active = False
        self.transaction_operation = None
        self.transaction_meta = None
        self.transaction_candidate_meta = None

        self.notesChanged.emit()
        if audition_note_ids:
            self.notesAuditionRequested.emit(audition_note_ids)

    # ==== Undo / Redo 本体 ====

    def undo(self):
        if not self.undo_stack:
            return

        # 現在状態を Redo 用に保存
        self.sync_back_to_model()
        current_state = self.dtpr_adapter.snapshot_dtpr_state(include_checked=False)

        entry = self.undo_stack.pop()
        prev_state = entry["state"]
        target_op_id = entry.get("op_id")
        entry_domain = entry.get("domain", self.domain)

        # redo にも op_id を持ち回る
        self.redo_stack.append({"state": current_state, "op_id": target_op_id, "domain": entry_domain})

        self.dtpr_adapter.restore_dtpr_state(prev_state, preserve_checked=True)
        notes = self._get_display_notes()
        if notes:
            self.next_note_id = max(self._note_id(n) for n in notes) + 1
        else:
            self.next_note_id = 0

        self.populate_notes()
        self.notesChanged.emit()

        if self.edit_logger is not None:
            self.edit_logger.record_undo(target_op_id=target_op_id, meta={"domain": entry_domain})

    def redo(self):
        if not self.redo_stack:
            return

        # 現在状態を Undo 用に保存
        self.sync_back_to_model()
        current_state = self.dtpr_adapter.snapshot_dtpr_state(include_checked=False)
        
        entry = self.redo_stack.pop()
        next_state = entry["state"]
        target_op_id = entry.get("op_id")
        entry_domain = entry.get("domain", self.domain)

        # undo にも op_id を持ち回る
        self.undo_stack.append({"state": current_state, "op_id": target_op_id, "domain": entry_domain})

        self.dtpr_adapter.restore_dtpr_state(next_state, preserve_checked=True)
        notes = self._get_display_notes()
        if notes:
            self.next_note_id = max(self._note_id(n) for n in notes) + 1
        else:
            self.next_note_id = 0

        self.populate_notes()
        self.notesChanged.emit()

        if self.edit_logger is not None:
            self.edit_logger.record_redo(target_op_id=target_op_id, meta={"domain": entry_domain})

    def set_background_spec(self, spec_2d: np.ndarray, hop_sec: float, pitch_min: int, pitch_max: int):
        # spec_2d: (F, T) float (大きいほど明るい、など)
        u8 = normalize_to_u8(spec_2d)        # ここは min-max 正規化など
        self._spec_u8 = u8
        self._spec_img = _to_grayscale_qimage(u8)
        self._spec_hop_sec = float(hop_sec)
        self._spec_x_edges = None
        self._spec_pitch_min = int(pitch_min)
        self._spec_pitch_max = int(pitch_max)

    def set_background_spec_mtpr(
        self,
        spec_2d: np.ndarray,
        pitch_min: int,
        pitch_max: int,
        x_edges: np.ndarray,
    ):
        u8 = normalize_to_u8(spec_2d)
        self._spec_u8 = u8
        self._spec_img = _to_grayscale_qimage(u8)
        self._spec_hop_sec = None
        self._spec_x_edges = np.asarray(x_edges, dtype=np.float32)
        self._spec_pitch_min = int(pitch_min)
        self._spec_pitch_max = int(pitch_max)

    def clear_background_spec(self):
        self._spec_u8 = None
        self._spec_img = None
        self._spec_hop_sec = None
        self._spec_x_edges = None

    def set_background_chroma(self, chroma_2d: np.ndarray, hop_sec: float, pitch_min: int, pitch_max: int):
        u8 = normalize_to_u8(chroma_2d)      # (12, T) を想定
        self._chroma_u8 = u8
        self._chroma_img = _to_grayscale_qimage(u8)
        self._chroma_hop_sec = float(hop_sec)
        self._chroma_x_edges = None
        self._chroma_pitch_min = int(pitch_min)
        self._chroma_pitch_max = int(pitch_max)

    def set_background_chroma_mtpr(
        self,
        chroma_2d: np.ndarray,
        pitch_min: int,
        pitch_max: int,
        x_edges: np.ndarray,
    ):
        u8 = normalize_to_u8(chroma_2d)
        self._chroma_u8 = u8
        self._chroma_img = _to_grayscale_qimage(u8)
        self._chroma_hop_sec = None
        self._chroma_x_edges = np.asarray(x_edges, dtype=np.float32)
        self._chroma_pitch_min = int(pitch_min)
        self._chroma_pitch_max = int(pitch_max)

    def clear_background_chroma(self):
        self._chroma_u8 = None
        self._chroma_img = None
        self._chroma_hop_sec = None
        self._chroma_x_edges = None

    def set_background_pitch_prob(self, prob_2d: np.ndarray, hop_sec: float, pitch_min: int, pitch_max: int):
        # prob_2d: (F, T) float (0..1 想定)
        u8 = normalize_to_u8(prob_2d)
        self._pitch_prob_u8 = u8
        self._pitch_prob_img = _to_grayscale_qimage(u8)
        self._pitch_prob_hop_sec = float(hop_sec)
        self._pitch_prob_pitch_min = int(pitch_min)
        self._pitch_prob_pitch_max = int(pitch_max)
        self._pitch_prob_x_edges = None

    def set_background_pitch_prob_mtpr(
        self,
        prob_2d: np.ndarray,
        pitch_min: int,
        pitch_max: int,
        x_edges: np.ndarray,
    ):
        u8 = normalize_to_u8(prob_2d)
        self._pitch_prob_u8 = u8
        self._pitch_prob_img = _to_grayscale_qimage(u8)
        self._pitch_prob_hop_sec = None
        self._pitch_prob_pitch_min = int(pitch_min)
        self._pitch_prob_pitch_max = int(pitch_max)
        self._pitch_prob_x_edges = np.asarray(x_edges, dtype=np.float32)

    def clear_background_pitch_prob(self):
        self._pitch_prob_u8 = None
        self._pitch_prob_img = None
        self._pitch_prob_hop_sec = None
        self._pitch_prob_x_edges = None

    def _draw_feature_background(self, painter: QPainter, rect: QRectF):
        # スペクトログラム
        if self.show_spectrogram and self._spec_img is not None:
            if self._spec_hop_sec:
                self._draw_time_pitch_image(
                    painter, rect,
                    img=self._spec_img,
                    hop_sec=self._spec_hop_sec,
                    pitch_min=self._spec_pitch_min,
                    pitch_max=self._spec_pitch_max,
                )
            elif self._spec_x_edges is not None:
                self._draw_tick_warped_pitch_image(
                    painter, rect,
                    img=self._spec_img,
                    x_edges=self._spec_x_edges,
                    pitch_min=self._spec_pitch_min,
                    pitch_max=self._spec_pitch_max,
                )

        # クロマ（縦方向は 12 クラスなので、どこに置くかは設計次第）
        # まずは「ピアノロール全高に引き伸ばして薄く敷く」方式にするなら同様に描画可能
        if self.show_chroma and self._chroma_img is not None:
            if self._chroma_hop_sec:
                self._draw_time_pitch_image(
                    painter, rect,
                    img=self._chroma_img,
                    hop_sec=self._chroma_hop_sec,
                    pitch_min=self._chroma_pitch_min,
                    pitch_max=self._chroma_pitch_max,
                )
            elif self._chroma_x_edges is not None:
                self._draw_tick_warped_pitch_image(
                    painter, rect,
                    img=self._chroma_img,
                    x_edges=self._chroma_x_edges,
                    pitch_min=self._chroma_pitch_min,
                    pitch_max=self._chroma_pitch_max,
                )

        # ピッチ確率(0..127)
        if self.show_pitch_prob and self._pitch_prob_img is not None:
            if self._pitch_prob_hop_sec:
                self._draw_time_pitch_image(
                    painter, rect,
                    img=self._pitch_prob_img,
                    hop_sec=self._pitch_prob_hop_sec,
                    pitch_min=self._pitch_prob_pitch_min,
                    pitch_max=self._pitch_prob_pitch_max,
                )
            elif self._pitch_prob_x_edges is not None:
                self._draw_tick_warped_pitch_image(
                    painter, rect,
                    img=self._pitch_prob_img,
                    x_edges=self._pitch_prob_x_edges,
                    pitch_min=self._pitch_prob_pitch_min,
                    pitch_max=self._pitch_prob_pitch_max,
                )

    def _draw_time_pitch_image(
        self,
        painter: QPainter,
        rect: QRectF,
        img: QImage,
        hop_sec: float,
        pitch_min: int,
        pitch_max: int,
    ):
        """
        img: (H=freqbins, W=frames) のグレースケール画像を、
        時間(x)とピッチ(y)に対応させて背景に描画する。
        """
        if hop_sec <= 0:
            return

        # rect(シーン座標) → 可視時間範囲(sec)
        t0 = max(0.0, rect.left() * self.seconds_per_pixel)
        t1 = max(t0, rect.right() * self.seconds_per_pixel)

        # 画像フレーム範囲
        col0 = int(math.floor(t0 / hop_sec))
        col1 = int(math.ceil(t1 / hop_sec))
        col0 = max(0, min(col0, img.width()))
        col1 = max(col0 + 1, min(col1, img.width()))
        if col1 <= col0:
            return

        # ソース矩形（画像座標）
        src = QRectF(col0, 0, col1 - col0, img.height())

        # 描画先（シーン座標）
        x0 = self.time_to_x(col0 * hop_sec)
        x1 = self.time_to_x(col1 * hop_sec)

        # pitch_to_y は「高いピッチほど上」なので、pitch_max が上端
        y_top = self.pitch_to_y(pitch_max)
        y_bottom = self.pitch_to_y(pitch_min) + self.px_per_pitch

        dst = QRectF(x0, y_top, x1 - x0, y_bottom - y_top)

        painter.save()
        painter.setOpacity(0.35)  # 背景なので薄く
        painter.drawImage(dst, img, src)
        painter.restore()

    def _draw_tick_warped_pitch_image(
        self,
        painter: QPainter,
        rect: QRectF,
        img: QImage,
        x_edges: np.ndarray,
        pitch_min: int,
        pitch_max: int,
    ):
        if x_edges is None or len(x_edges) < 2 or img.width() <= 0:
            return

        left = float(rect.left())
        right = float(rect.right())
        col0 = int(max(0, np.searchsorted(x_edges, left, side="right") - 1))
        col1 = int(min(img.width(), np.searchsorted(x_edges, right, side="left") + 1))
        if col1 <= col0:
            return

        y_top = self.pitch_to_y(pitch_max)
        y_bottom = self.pitch_to_y(pitch_min) + self.px_per_pitch
        height = y_bottom - y_top
        if height <= 0:
            return

        painter.save()
        painter.setOpacity(0.35)
        for col in range(col0, col1):
            x0 = float(x_edges[col])
            x1 = float(x_edges[col + 1])
            if x1 <= left or x0 >= right:
                continue
            if x1 <= x0:
                x1 = x0 + 1.0
            src = QRectF(float(col), 0.0, 1.0, float(img.height()))
            dst = QRectF(x0, y_top, x1 - x0, height)
            painter.drawImage(dst, img, src)
        painter.restore()


    def _draw_time_only_image(
        self,
        painter: QPainter,
        rect: QRectF,
        img: QImage,
        hop_sec: float,
    ):
        """
        クロマなど、縦方向の意味付けを簡略化して
        「時間にだけ対応させて全高に敷く」描画。
        """
        if hop_sec <= 0:
            return

        t0 = max(0.0, rect.left() * self.seconds_per_pixel)
        t1 = max(t0, rect.right() * self.seconds_per_pixel)

        col0 = int(math.floor(t0 / hop_sec))
        col1 = int(math.ceil(t1 / hop_sec))
        col0 = max(0, min(col0, img.width()))
        col1 = max(col0 + 1, min(col1, img.width()))
        if col1 <= col0:
            return

        src = QRectF(col0, 0, col1 - col0, img.height())

        x0 = self.time_to_x(col0 * hop_sec)
        x1 = self.time_to_x(col1 * hop_sec)

        # ピアノロール全体（ピッチ範囲）に敷く
        y_top = self.pitch_to_y(self.max_pitch)
        y_bottom = self.pitch_to_y(self.min_pitch) + self.px_per_pitch
        dst = QRectF(x0, y_top, x1 - x0, y_bottom - y_top)

        painter.save()
        painter.setOpacity(0.20)
        painter.drawImage(dst, img, src)
        painter.restore()

class MTPRGridScene(PianoRollScene):
    def __init__(
        self,
        parent=None,
        edit_logger: Optional[EditOperationLogger] = None,
        dtpr_adapter: Optional[DTPRAdapter] = None,
        subdivisions_per_beat: int = 4,
    ):
        self.px_per_tick = 0.2
        self.max_tick = TPQN * 8
        self.subdivisions_per_beat = max(1, int(subdivisions_per_beat))
        self._beat_ticks: list[int] = []
        self._downbeat_ticks: set[int] = set()
        super().__init__(
            parent=parent,
            edit_logger=edit_logger,
            dtpr_adapter=dtpr_adapter,
            domain="mtpr",
        )
        self.seconds_per_pixel = 1.0 / float(self.px_per_tick)
        self.set_tick_range(self.max_tick)
        self.populate_notes()

    def _time_grid_step(self) -> float:
        return float(max(1, TPQN // max(1, self.subdivisions_per_beat)))

    def tick_to_x(self, tick: float) -> float:
        return self.domain_time_to_x(float(max(0.0, float(tick))))

    def dur_to_w(self, d: float) -> float:
        return max(0.1, float(d) / float(self.seconds_per_pixel))

    def set_px_per_tick(self, px_per_tick: float) -> None:
        px = max(0.005, float(px_per_tick))
        self.px_per_tick = px
        self.seconds_per_pixel = 1.0 / px
        self.set_tick_range(self.max_tick)
        self.populate_notes()

    def set_playhead_time(self, t: float):
        if t < 0:
            t = 0.0

        x = 0.0
        adapter = getattr(self, "dtpr_adapter", None)
        bdb = getattr(getattr(adapter, "dtpr", None), "BDB", None)
        if bdb is not None and hasattr(bdb, "index") and len(getattr(bdb, "index", [])) >= 2:
            beat_times = [float(bt) for bt, _beat_id in list(bdb.index)]
            if t < beat_times[0]:
                idx = 0
                t0 = beat_times[0]
                t1 = beat_times[1]
            elif t >= beat_times[-1]:
                idx = max(0, len(beat_times) - 2)
                t0 = beat_times[idx]
                t1 = beat_times[idx + 1]
            else:
                idx = max(0, bisect_right(beat_times, t) - 1)
                idx = min(idx, len(beat_times) - 2)
                t0 = beat_times[idx]
                t1 = beat_times[idx + 1]

            denom = max(1e-9, t1 - t0)
            frac = (t - t0) / denom
            abs_tick = float(idx * TPQN + frac * TPQN)
            x = float(self.tick_to_x(abs_tick))

        top = self.pitch_to_y(self.max_pitch)
        bottom = self.pitch_to_y(self.min_pitch) + self.px_per_pitch

        if self.playhead_item is None or self.playhead_item.scene() is None:
            self.playhead_item = PlayheadLineItem(self)
            pen = QPen(QColor(255, 200, 0))
            pen.setWidth(1)
            self.playhead_item.setPen(pen)
            self.addItem(self.playhead_item)

        self.playhead_item.setLine(x, top, x, bottom)

    def set_tick_range(self, max_tick: int) -> None:
        self.max_tick = max(int(max_tick), TPQN)
        self.set_total_time(float(self.max_tick + TPQN))

    def set_mtpr_notes(self, notes) -> None:
        # dtpr_adapter 側の現状態を描画するため、引数は互換のためのみ受け取る
        self.populate_notes()

    def populate_notes(self):
        for it in list(self.note_items.values()):
            try:
                self.removeItem(it)
            except Exception:
                pass
        self.note_items.clear()

        for note in self._get_display_notes():
            x0 = self.tick_to_x(self._note_onset(note))
            x1 = self.tick_to_x(self._note_offset(note))
            x = x0
            w = max(0.1, x1 - x0)
            y = self.pitch_to_y(self._note_pitch(note))
            h = self.px_per_pitch
            item = NoteItem(note, x, y, w, h)
            self.addItem(item)
            self.note_items[self._note_id(note)] = item

    def set_bdb_grid(self, bdb, subdivisions_per_beat: int = 4) -> None:
        self.subdivisions_per_beat = max(1, int(subdivisions_per_beat))
        self._beat_ticks = []
        self._downbeat_ticks = set()

        if bdb is None or not hasattr(bdb, "index") or not hasattr(bdb, "beats"):
            return

        for beat_idx, (_, beat_id) in enumerate(list(bdb.index)):
            tick = int(beat_idx * TPQN)
            self._beat_ticks.append(tick)
            beat_obj = bdb.beats.get(beat_id)
            if beat_obj is not None and bool(getattr(beat_obj, "is_downbeat", False)):
                self._downbeat_ticks.add(tick)

    def drawBackground(self, painter, rect: QRectF):
        QGraphicsScene.drawBackground(self, painter, rect)
        painter.save()

        self._draw_feature_background(painter, rect)
        self._draw_black_key_rows(painter, rect)

        pitch_pen = QPen(QColor(230, 230, 230), 0)
        for p in range(self.min_pitch, self.max_pitch + 1):
            y = self.pitch_to_y(p)
            if y < rect.top() - self.px_per_pitch or y > rect.bottom() + self.px_per_pitch:
                continue
            painter.setPen(pitch_pen)
            painter.drawLine(rect.left(), y, rect.right(), y)

        left_tick = max(0, int(math.floor(self.x_to_domain_time(rect.left()))))
        right_tick = max(left_tick, int(math.ceil(self.x_to_domain_time(rect.right()))))

        minor_step = max(1, TPQN // max(1, self.subdivisions_per_beat))
        major_step = TPQN
        start_tick = (left_tick // minor_step) * minor_step

        views = self.views()
        scale_x = float(views[0].transform().m11()) if views else 1.0
        scene_px = 1.0 / max(1e-9, abs(scale_x))

        beat_color = QColor(30, 30, 30)
        downbeat_color = QColor(0, 0, 0)
        minor_color = QColor(235, 235, 235)

        tick = start_tick
        while tick <= right_tick + minor_step:
            x = self.tick_to_x(float(tick))
            if (tick % major_step) == 0:
                if tick in self._downbeat_ticks:
                    color = downbeat_color
                    width = 2.5 * scene_px
                else:
                    color = beat_color
                    width = 1.5 * scene_px
            else:
                color = minor_color
                width = scene_px
            painter.fillRect(QRectF(x, rect.top(), width, rect.height()), color)
            tick += minor_step

        painter.restore()
