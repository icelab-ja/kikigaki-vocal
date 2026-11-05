# note_confidence.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple, Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QBrush


@dataclass
class ConfidenceConfig:
    hop_sec: float = 0.01
    start_sec: float = 0.0
    mode: str = "point"          # "point" | "max" | "mean"
    window_frames: int = 0       # point=0, max/meanなら1以上を推奨
    v_min: int = 90              # 明度の下限（0-255）
    v_max: int = 255             # 明度の上限（0-255）
    conf_full_scale: float = 0.25
    gb_min_scale: float = 0.15


class ConfidenceSeries:
    def __init__(self, config: ConfidenceConfig):
        self.cfg = config
        self.data: Optional[np.ndarray] = None

    def set_data(self, conf_values: np.ndarray) -> None:
        """
        conf_values: (T,) の 1次元配列を想定（0.0〜1.0の確率スケール推奨）
        """
        a = np.asarray(conf_values, dtype=np.float32).reshape(-1)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        self.data = a

    def _time_to_index(self, t_sec: float) -> int:
        # start_sec=0.0 固定でも、将来の拡張を見据えて一般形
        x = (t_sec - self.cfg.start_sec) / self.cfg.hop_sec
        return int(np.round(x))

    def conf_at_time(self, t_sec: float) -> float:
        """
        t_sec に対応する confidence を返す。
        範囲外（±1サンプル誤差含む）は 0.0 を返す方針。
        """
        if self.data is None or self.data.size == 0:
            return 0.0

        idx = self._time_to_index(t_sec)

        # 範囲外は 0（要件）
        if idx < 0 or idx >= int(self.data.size):
            return 0.0

        if self.cfg.mode == "point" or self.cfg.window_frames <= 0:
            return float(self.data[idx])

        k = int(self.cfg.window_frames)
        lo = max(0, idx - k)
        hi = min(int(self.data.size), idx + k + 1)
        seg = self.data[lo:hi]
        if seg.size == 0:
            return 0.0

        if self.cfg.mode == "max":
            return float(np.max(seg))
        if self.cfg.mode == "mean":
            return float(np.mean(seg))

        # 不明モードは保守的に point
        return float(self.data[idx])


def _clamp01(x: float) -> float:
    if x != x:  # NaN
        return 0.0
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _apply_value_to_color(base: QColor, conf: float, v_min: int, v_max: int, conf_full_scale: float, gb_min_scale: float = 0.15) -> QColor:
    """
    旧：HSVのV(明度)で表現
    新：RGBで「低信頼ほど赤く」表現（明度は変えない方針）
      - conf が低いほど R を 255 に近づける
      - 同時に G/B を gb_min_scale まで減衰（紫っぽくなるのを避ける）
    """
    conf = _clamp01(conf)

    # 0..1 にスケーリング（conf_full_scale 以上を「最大」扱い）
    if conf_full_scale <= 0:
        conf_scaled = conf
    else:
        conf_scaled = _clamp01(conf / float(conf_full_scale))

    # 低信頼ほど大きい重み（0:高信頼 → 1:低信頼）
    w = 1.0 - conf_scaled

    r0, g0, b0, a0 = base.red(), base.green(), base.blue(), base.alpha()

    # Rは255へ寄せる（低信頼ほど赤く）
    r = int(round(r0 + w * (255 - r0)))

    # G/Bは減衰（低信頼ほど小さく）しつつ、完全に0にしない
    gb_min_scale = float(gb_min_scale)
    if gb_min_scale < 0.0:
        gb_min_scale = 0.0
    if gb_min_scale > 1.0:
        gb_min_scale = 1.0

    scale = (1.0 - w) + w * gb_min_scale   # conf_scaled=1 → 1.0, conf_scaled=0 → gb_min_scale
    g = int(round(g0 * scale))
    b = int(round(b0 * scale))

    # clamp
    r = 0 if r < 0 else 255 if r > 255 else r
    g = 0 if g < 0 else 255 if g > 255 else g
    b = 0 if b < 0 else 255 if b > 255 else b

    return QColor(r, g, b, a0)


class NoteConfidenceStyler:
    """
    NoteItem の brush を「明度」で上書きする。
    - 既存の fill 色（base）を尊重し、明度だけを confidence で変える。
    - 選択中の視認性は枠線側で担保する想定（fillは維持）。
    """
    BASE_COLOR_KEY = Qt.UserRole + 910  # item.data(key) に base QColor を保存

    def __init__(self, series: ConfidenceSeries):
        self.series = series

    def _get_base_color(self, note_item: Any) -> QColor:
        stored = note_item.data(self.BASE_COLOR_KEY)
        if isinstance(stored, QColor):
            return stored

        # 初回：現在の brush 色を base として記録
        b = note_item.brush()
        c = b.color() if isinstance(b, QBrush) else QColor(120, 120, 120)
        if not isinstance(c, QColor) or not c.isValid():
            c = QColor(120, 120, 120)
        note_item.setData(self.BASE_COLOR_KEY, c)
        return c

    def apply_to_item(self, note_item: Any, onset_sec: float) -> float:
        """
        onset_sec に基づき confidence を計算し、note_item の brush を更新。
        戻り値：適用した confidence（0..1）
        """
        conf = self.series.conf_at_time(onset_sec)
        base = self._get_base_color(note_item)
        c2 = _apply_value_to_color(base, conf, self.series.cfg.v_min, self.series.cfg.v_max, self.series.cfg.conf_full_scale, self.series.cfg.gb_min_scale)
        note_item.setBrush(QBrush(c2))
        return conf


class LowConfidenceNavigator:
    """
    閾値 τ 未満のノートを onset 時刻順にスキャンして「次へジャンプ」する。
    - 現在選択ノートがあればその onset を基準に “次”
    - 無ければ view の表示範囲左端（scene x=0 の点）を基準
    """
    def __init__(self, series: ConfidenceSeries):
        self.series = series
        self.threshold: float = 0.3
        self._cache: List[Tuple[float, int]] = []  # (onset_sec, note_id)

    def set_threshold(self, thr: float) -> None:
        self.threshold = float(thr)

    def rebuild_cache(self, notes: List[Any]) -> None:
        """
        notes: 実時間を持つノート列（.real_ontime / .onset / .ideal_ontime, .id）
        """
        thr = float(self.threshold)
        out: List[Tuple[float, int]] = []
        for n in notes:
            onset = _note_real_onset_sec(n)
            nid = int(getattr(n, "id"))
            conf = self.series.conf_at_time(onset)
            if conf < thr:
                out.append((onset, nid))
        out.sort(key=lambda x: x[0])
        self._cache = out

    def next_note_id_after(self, t_ref: float) -> Optional[int]:
        """
        t_ref より後ろの最初の低信頼 note_id を返す。なければ None。
        """
        for onset, nid in self._cache:
            if onset > t_ref:
                return nid
        return None

    def first_note_id(self) -> Optional[int]:
        return self._cache[0][1] if self._cache else None


def get_reference_time_sec(scene: Any, view: Any) -> float:
    """
    仕様：
    - 選択中 NoteItem があればそのノートの onset を基準（次へ）
    - 無ければ view 表示範囲左端 x を秒へ変換
    """
    # 1) 選択中ノート
    selected = []
    try:
        selected = scene.selectedItems()
    except Exception:
        selected = []

    # NoteItem 判定は "note_id" 属性で簡易に行う（依存を増やさない）
    for it in selected:
        if hasattr(it, "note_id"):
            nid = int(getattr(it, "note_id"))
            note = _scene_ptpr_note_by_id(scene, nid)
            if note is not None:
                return _note_real_onset_sec(note)

    # 2) 表示範囲の左端（viewport左上のscene座標 x）
    try:
        vp = view.viewport()
        p = vp.rect().topLeft()
        p_scene = view.mapToScene(p)
        x_scene = float(p_scene.x())
        return x_scene * float(scene.seconds_per_pixel)
    except Exception:
        return 0.0


def _note_real_onset_sec(note: Any) -> float:
    if note is None:
        return 0.0
    for attr in ("real_ontime", "onset", "ideal_ontime"):
        value = getattr(note, attr, None)
        if value is not None:
            return float(value)
    return 0.0


def _scene_ptpr_note_by_id(scene: Any, note_id: int) -> Any:
    adapter = getattr(scene, "dtpr_adapter", None)
    if adapter is not None and hasattr(adapter, "get_ptpr_note_by_id"):
        try:
            return adapter.get_ptpr_note_by_id(int(note_id))
        except Exception:
            return None
    return None


def _scene_ptpr_notes(scene: Any) -> List[Any]:
    adapter = getattr(scene, "dtpr_adapter", None)
    if adapter is not None and hasattr(adapter, "get_ptpr_notes_sorted"):
        try:
            return list(adapter.get_ptpr_notes_sorted())
        except Exception:
            return []
    return []


class ConfidenceController:
    """
    MainWindowから利用する信頼度処理の統合クラス。
    UI以外（ロード、反映、閾値、ジャンプ、キャッシュ再構築）をまとめる。
    """
    def __init__(self, hop_sec: float = 0.01, start_sec: float = 0.0):
        cfg = ConfidenceConfig(hop_sec=hop_sec, start_sec=start_sec, mode="point", window_frames=0)
        self.series = ConfidenceSeries(cfg)
        self.styler = NoteConfidenceStyler(self.series)
        self.nav = LowConfidenceNavigator(self.series)

        self.enabled: bool = True  # 表示ON/OFF用（後でチェックボックスを付けるなら使う）
        self.model_path: Optional[str] = None
        self.model_logprobs: Optional[np.ndarray] = None
        self.pitch_rest_logprobs: Optional[np.ndarray] = None
        self.note_conf_logprob: Optional[np.ndarray] = None

    # --- public API ---
    def load_model_output_npy(self, path: str) -> None:
        arr = np.asarray(np.load(path), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 130:
            raise ValueError(f"Invalid npy shape: expected (T,130), got {arr.shape}")

        self.model_path = str(path)
        self.model_logprobs = arr
        self.pitch_rest_logprobs = arr[:, :129]
        self.note_conf_logprob = arr[:, 129]

        conf_logp = np.nan_to_num(
            self.note_conf_logprob,
            nan=-np.inf,
            posinf=-np.inf,
            neginf=-np.inf,
        )
        self.set_confidence_array(np.exp(conf_logp).astype(np.float32))

    def clear_model_output(self) -> None:
        self.model_path = None
        self.model_logprobs = None
        self.pitch_rest_logprobs = None
        self.note_conf_logprob = None
        self.series.data = None
        self.nav._cache = []

    def snapshot_model_output(self) -> dict:
        def copied(value):
            return None if value is None else np.array(value, copy=True)

        return {
            "model_npy_path": self.model_path,
            "model_logprobs": copied(self.model_logprobs),
            "pitch_rest_logprobs": copied(self.pitch_rest_logprobs),
            "note_conf_logprob": copied(self.note_conf_logprob),
            "conf_data": copied(self.series.data),
        }

    def restore_model_output(self, state: dict) -> None:
        def copied(key: str):
            value = state.get(key)
            return None if value is None else np.array(value, copy=True)

        self.model_path = state.get("model_npy_path")
        self.model_logprobs = copied("model_logprobs")
        self.pitch_rest_logprobs = copied("pitch_rest_logprobs")
        self.note_conf_logprob = copied("note_conf_logprob")
        self.series.data = copied("conf_data")
        self.nav._cache = []

    def has_confidence_data(self) -> bool:
        data = self.series.data
        return data is not None and getattr(data, "size", 0) > 0

    def set_confidence_array(self, conf_values: np.ndarray) -> None:
        """
        main 側でロード・整形した confidence 系列を受け取る。
        conf_values: (T,) 1次元（0〜1推奨）
        """
        self.series.set_data(conf_values)

    def set_mode_point(self) -> None:
        self.series.cfg.mode = "point"
        self.series.cfg.window_frames = 0

    def set_mode_max(self, window_frames: int) -> None:
        self.series.cfg.mode = "max"
        self.series.cfg.window_frames = int(window_frames)

    def set_mode_mean(self, window_frames: int) -> None:
        self.series.cfg.mode = "mean"
        self.series.cfg.window_frames = int(window_frames)

    def set_threshold(self, thr: float) -> None:
        self.nav.set_threshold(thr)

    def apply_to_all_items(self, scene: Any) -> None:
        if not self.enabled or self.series.data is None:
            return

        # NoteItem は scene.items() から拾う。note_id 属性で判定する。
        for it in scene.items():
            if not hasattr(it, "note_id"):
                continue

            if not (hasattr(it, "brush") and hasattr(it, "setBrush")):
                continue

            nid = int(getattr(it, "note_id"))
            note = _scene_ptpr_note_by_id(scene, nid)
            if note is None:
                continue
            self.styler.apply_to_item(it, _note_real_onset_sec(note))

    def rebuild_low_conf_cache(self, scene: Any) -> None:
        if self.series.data is None:
            self.nav._cache = []
            return
        self.nav.rebuild_cache(_scene_ptpr_notes(scene))

    def jump_next_low_conf(self, scene: Any, view: Any) -> bool:
        """
        次の低信頼ノートへジャンプして選択＋中央寄せ。成功なら True。
        見つからない場合は先頭へラップ（要件に未指定だが実運用向け）し、
        それも無ければ False。
        """
        if self.series.data is None:
            return False

        t_ref = get_reference_time_sec(scene, view)
        nid = self.nav.next_note_id_after(t_ref)
        if nid is None:
            nid = self.nav.first_note_id()
            if nid is None:
                return False

        # note_id -> item を探す（規模が小さければ線形探索で十分）
        target_item = None
        for it in scene.items():
            if hasattr(it, "note_id") and int(getattr(it, "note_id")) == nid:
                target_item = it
                break
        if target_item is None:
            return False

        # 選択してジャンプ
        scene.clearSelection()
        target_item.setSelected(True)
        view.centerOn(target_item)
        return True
