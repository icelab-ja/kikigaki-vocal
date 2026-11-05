from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import QGraphicsItem, QGraphicsLineItem


@dataclass(frozen=True)
class PeakCandidate:
    index: int
    score: float
    time_sec: float


@dataclass(frozen=True)
class PitchCandidate:
    pitch: int
    score: float


def _local_maxima_indices(x: np.ndarray) -> np.ndarray:
    """
    局所最大の index を返す（端は除外）。
    x[i-1] < x[i] >= x[i+1]
    """
    if x.size < 3:
        return np.array([], dtype=np.int64)
    return np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1


def topk_peaks_around(
    conf: np.ndarray,
    *,
    center_time_sec: float,
    hop_sec: float = 0.01,
    window_sec: float = 0.15,
    topk: int = 3,
    min_separation_sec: float = 0.02,
    exclude_center_sec: float = 0.02,
) -> List[PeakCandidate]:
    """
    center_time_sec ± window_sec の範囲でピーク候補を抽出する。
    - conf: (T,) 1次元系列（確率でもlog確率でも可）
    - min_separation_sec: 近接ピーク抑制（NMS）
    - exclude_center_sec: 現在オンセット近傍は候補から除外（見た目の冗長さ回避）
    """
    a = np.asarray(conf, dtype=np.float32).reshape(-1)
    t_len = int(a.size)
    if t_len == 0:
        return []

    center_idx = int(round(center_time_sec / hop_sec))
    radius = int(round(window_sec / hop_sec))
    lo = max(0, center_idx - radius)
    hi = min(t_len - 1, center_idx + radius)

    seg = a[lo:hi + 1]
    if seg.size == 0:
        return []

    rel_peaks = _local_maxima_indices(seg)
    peaks = (rel_peaks + lo).astype(np.int64)

    if peaks.size == 0:
        peaks = np.argsort(seg)[::-1][: max(topk * 4, topk)] + lo

    excl = int(round(exclude_center_sec / hop_sec))
    if excl > 0:
        peaks = peaks[np.abs(peaks - center_idx) > excl]

    min_sep = int(max(1, round(min_separation_sec / hop_sec)))
    order = peaks[np.argsort(a[peaks])[::-1]]

    selected: List[int] = []
    for idx in order:
        if all(abs(idx - s) >= min_sep for s in selected):
            selected.append(int(idx))
        if len(selected) >= topk:
            break

    selected.sort()
    return [
        PeakCandidate(index=i, score=float(a[i]), time_sec=float(i * hop_sec))
        for i in selected
    ]


def topk_peaks_in_interval(
    conf: np.ndarray,
    *,
    onset_sec: float,
    offset_sec: float,
    hop_sec: float,
    min_margin_sec: float,
    topk: int = 3,
    min_separation_sec: float = 0.02,
) -> List[PeakCandidate]:
    """
    [onset_sec, offset_sec] の区間内（ただし両端 min_margin_sec を除外）で
    conf（発音確率; log確率でもOK）の局所最大を取り、上位 topk を返す。
    """
    if conf is None or getattr(conf, "size", 0) == 0:
        return []

    search_start = onset_sec + float(min_margin_sec)
    search_end = offset_sec - float(min_margin_sec)
    if search_end <= search_start:
        return []

    t_len = int(conf.shape[0])
    s0 = int(np.floor(search_start / hop_sec))
    s1 = int(np.ceil(search_end / hop_sec))
    s0 = max(0, min(t_len - 1, s0))
    s1 = max(0, min(t_len, s1))
    if s1 <= s0 + 2:
        return []

    segment = conf[s0:s1]

    peaks = []
    for i in range(1, len(segment) - 1):
        if segment[i - 1] < segment[i] and segment[i] >= segment[i + 1]:
            peaks.append(i)
    if not peaks:
        return []

    peaks = sorted(peaks, key=lambda i: float(segment[i]), reverse=True)

    min_sep_frames = max(1, int(round(min_separation_sec / hop_sec)))
    picked: List[int] = []
    for i in peaks:
        if all(abs(i - j) >= min_sep_frames for j in picked):
            picked.append(i)
            if len(picked) >= topk:
                break

    out: List[PeakCandidate] = []
    for i in picked:
        frame = s0 + i
        t_sec = float(frame * hop_sec)
        out.append(PeakCandidate(index=frame, score=float(segment[i]), time_sec=t_sec))
    return out


def _logsumexp(a: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    a: log-prob array
    returns: log(sum(exp(a))) along axis
    """
    if a.size == 0:
        return np.array([], dtype=np.float64)
    m = np.max(a, axis=axis, keepdims=True)
    m2 = np.where(np.isfinite(m), m, 0.0)
    s = np.sum(np.exp(a - m2), axis=axis, keepdims=True)
    out = m2 + np.log(np.maximum(s, 1e-30))
    return np.squeeze(out, axis=axis)


def topk_pitch_candidates_for_note(
    logp_130: np.ndarray,
    onset_sec: float,
    offset_sec: float,
    *,
    hop_sec: float,
    exclude_pitch: Optional[int],
    topk: int = 3,
    pitch_min: Optional[int] = None,
    pitch_max: Optional[int] = None,
) -> List[PitchCandidate]:
    """
    logp_130: (T,130) うち pitch は 0..127 を使用（128=restは使わない）
    集計: 各pitch pについて sum_t exp(logp[t,p]) を取り、その大小で上位を返す
          実装は log-sum-exp で安定化（score = log(sum exp)）
    """
    if logp_130 is None or getattr(logp_130, "ndim", 0) != 2 or logp_130.shape[1] < 128:
        return []

    t_len = int(logp_130.shape[0])

    start = int(np.floor(onset_sec / hop_sec))
    end = int(np.ceil(offset_sec / hop_sec))

    start = max(0, min(t_len, start))
    end = max(0, min(t_len, end))
    if end <= start:
        return []

    seg = logp_130[start:end, 0:128].astype(np.float64, copy=False)
    scores = _logsumexp(seg, axis=0)

    if pitch_min is not None and pitch_max is not None:
        lo = max(0, int(pitch_min))
        hi = min(127, int(pitch_max))
        if lo > 0:
            scores[:lo] = -np.inf
        if hi < 127:
            scores[hi + 1:] = -np.inf

    if exclude_pitch is not None and 0 <= int(exclude_pitch) <= 127:
        scores[int(exclude_pitch)] = -np.inf

    if topk <= 0:
        return []

    idx = np.argsort(scores)[::-1]
    out: List[PitchCandidate] = []
    for p in idx[:topk]:
        sc = float(scores[int(p)])
        if not np.isfinite(sc):
            continue
        out.append(PitchCandidate(pitch=int(p), score=sc))
    return out


class PitchCandidateHLineItem(QGraphicsLineItem):
    """
    ピッチ候補: onset〜offset区間に水平線を描画し、クリックで即適用。
    """

    def __init__(
        self,
        x0: float,
        x1: float,
        y: float,
        *,
        pitch: int,
        owner_note_id: int,
        on_click: Callable[["PitchCandidateHLineItem", object], None],
    ):
        super().__init__(x0, y, x1, y)
        self.pitch = int(pitch)
        self.owner_note_id = int(owner_note_id)
        self._on_click = on_click

        pen = QPen(QColor(80, 180, 255))
        pen.setWidth(6)
        pen.setCosmetic(True)
        self.setPen(pen)

        self.setZValue(10_500)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if callable(self._on_click):
                self._on_click(self, event.scenePos())
                event.accept()
                return
        super().mousePressEvent(event)
