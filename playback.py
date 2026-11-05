import copy
import sys
import time
from typing import Optional

import librosa
import mido
import numpy as np
import sounddevice as sd
from PySide6.QtCore import QObject, QThread, QTimer, Signal

from fluidsynth_backend import open_default_midi_output


METRONOME_CHANNEL = 9
METRONOME_DOWNBEAT_NOTE = 76
METRONOME_BEAT_NOTE = 77
METRONOME_NOTE_DURATION_SEC = 0.04
DEFAULT_MIDI_PROGRAM = 71

VOLUME_SLIDER_DEFAULT = 50
VOLUME_SLIDER_BOOST_DB = 12.0
VOLUME_SLIDER_CUT_DB = 48.0
AUDIO_TARGET_ACTIVE_RMS = 0.08
AUDIO_NORMALIZE_MIN_GAIN = 0.05
AUDIO_NORMALIZE_MAX_GAIN = 8.0


def load_audio_file(path: str, target_sr: int) -> tuple[np.ndarray, int]:
    data, sample_rate = librosa.load(
        path,
        sr=int(target_sr),
        mono=False,
    )
    if data.ndim == 1:
        data = data[:, None]
    else:
        data = np.ascontiguousarray(data.T)
    return data.astype(np.float32, copy=False), int(sample_rate)


def build_stereo_mix(
    tracks: list[tuple[np.ndarray, float, bool, bool]],
) -> Optional[np.ndarray]:
    if not tracks:
        return None

    max_len = max(data.shape[0] for data, _volume, _left, _right in tracks)
    mix = np.zeros((max_len, 2), dtype=np.float32)
    for data, volume, left_enabled, right_enabled in tracks:
        mono = data[:, 0] if data.shape[1] == 1 else np.mean(data, axis=1)
        signal = np.asarray(mono, dtype=np.float32) * float(volume)
        if left_enabled:
            mix[:data.shape[0], 0] += signal
        if right_enabled:
            mix[:data.shape[0], 1] += signal

    peak = float(np.max(np.abs(mix)))
    if peak > 1.0:
        mix /= peak
    return mix


def max_audio_duration_seconds(
    tracks: list[Optional[np.ndarray]],
    sample_rate: Optional[int],
) -> float:
    if sample_rate is None or sample_rate <= 0:
        return 0.0
    max_len = max(
        (int(data.shape[0]) for data in tracks if data is not None),
        default=0,
    )
    return max_len / float(sample_rate)


def stop_audio_output() -> None:
    sd.stop()


def _volume_slider_to_gain(value: int) -> float:
    value = max(0, min(100, int(value)))
    if value <= 0:
        return 0.0
    if value >= VOLUME_SLIDER_DEFAULT:
        ratio = (value - VOLUME_SLIDER_DEFAULT) / float(100 - VOLUME_SLIDER_DEFAULT)
        db = ratio * VOLUME_SLIDER_BOOST_DB
    else:
        ratio = (VOLUME_SLIDER_DEFAULT - value) / float(VOLUME_SLIDER_DEFAULT)
        db = -ratio * VOLUME_SLIDER_CUT_DB
    return float(10.0 ** (db / 20.0))


def _metronome_slider_to_gain(value: int) -> float:
    value = max(0, min(100, int(value)))
    if value <= 0:
        return 0.0
    return float(value) / float(VOLUME_SLIDER_DEFAULT)


def _estimate_audio_playback_gain(data: np.ndarray) -> float:
    if data is None or data.size == 0:
        return 1.0

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        mono = arr
    else:
        mono = arr[:, 0] if arr.shape[1] == 1 else np.mean(arr, axis=1)

    if mono.size == 0:
        return 1.0

    step = max(1, mono.size // 1_000_000)
    sample = np.asarray(mono[::step], dtype=np.float32)
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return 1.0

    abs_sample = np.abs(sample)
    peak = float(np.max(abs_sample))
    if peak <= 1e-8:
        return 1.0

    threshold = max(1e-5, peak * 0.01)
    active = sample[abs_sample >= threshold]
    if active.size < max(1024, sample.size // 100):
        active = sample

    rms = float(np.sqrt(np.mean(np.square(active, dtype=np.float32), dtype=np.float32)))
    if rms <= 1e-8:
        return 1.0

    gain = AUDIO_TARGET_ACTIVE_RMS / rms
    return float(np.clip(gain, AUDIO_NORMALIZE_MIN_GAIN, AUDIO_NORMALIZE_MAX_GAIN))


def _time_stretch_audio(mix: np.ndarray, speed: float) -> np.ndarray:
    if speed <= 0 or abs(speed - 1.0) < 1e-6:
        return mix

    a = np.asarray(mix, dtype=np.float32)
    if a.ndim == 1:
        a = a[:, None]

    channels = a.shape[1]
    stretched = []
    for ch in range(channels):
        y = a[:, ch]
        y_st = librosa.effects.time_stretch(y, rate=speed)
        stretched.append(y_st)

    min_len = min(len(s) for s in stretched)
    if min_len <= 0:
        return mix
    out = np.stack([s[:min_len] for s in stretched], axis=1)
    return np.ascontiguousarray(out, dtype=np.float32)


class MidiClickPlayer:
    """
    鍵盤クリックなどの短いノートを鳴らすためのシンプルな MIDI 出力クラス。
    """

    def __init__(self, midi_out=None):
        self.out = midi_out  # mido output port
        self._owns_port = (midi_out is None)
        self.midi_program = DEFAULT_MIDI_PROGRAM

    def _ensure_open(self) -> bool:
        if self.out is not None:
            return True
        if not self._owns_port:
            return False
        self.out, _ = open_default_midi_output()
        return self.out is not None

    def play_note(self, pitch: int, velocity: int = 100, duration_ms: int = 300, channel: int = 0):
        """
        指定した pitch を短く鳴らす。
        """
        if not self._ensure_open():
            return

        pitch = int(pitch)
        velocity = int(velocity)

        if int(channel) == 0:
            self._send_program_change()

        # note_on を送信
        msg_on = mido.Message("note_on", note=pitch, velocity=velocity, channel=channel)
        self.out.send(msg_on)

        # duration_ms 後に note_off を送信
        QTimer.singleShot(duration_ms, lambda: self._note_off(pitch, channel))

    def _note_off(self, pitch: int, channel: int = 0):
        if self.out is None:
            return
        msg_off = mido.Message("note_off", note=int(pitch), velocity=0, channel=channel)
        self.out.send(msg_off)

    def set_program(self, program: int) -> None:
        self.midi_program = max(0, min(127, int(program)))
        self._send_program_change()

    def _send_program_change(self) -> None:
        if self.out is None:
            return
        self.out.send(
            mido.Message("program_change", program=int(self.midi_program), channel=0)
        )


class SyncAudioMidiPlayer(QThread):
    """
    2つの音源とMIDIノート列を、同じ時間軸で同期再生するスレッド。

    - mix_audio: (N, C) float32 の波形（None なら音声なし）
    - sample_rate: サンプリング周波数
    - notes: 開始時点のノートスナップショットを使用
    - midi_enabled: 再生中も変更可能な MIDI ON/OFF フラグ
    - midi_volume: MIDI velocity の音量倍率（再生中も変更可能）
    """
    progress = Signal(float)  # 再生経過時間 [sec] を通知
    def __init__(
        self,
        mix_audio: Optional[np.ndarray],
        sample_rate: int,
        notes,
        start_offset: float = 0.0,
        playback_speed: float = 1.0,
        total_duration_override: Optional[float] = None,
        parent: Optional[QObject] = None,
        midi_out=None,
        midi_port_name=None,
        beat_events=None,
        metronome_enabled: bool = False,
        metronome_volume: float = 1.0,
        midi_time_mode: str = "real",
        midi_program: int = DEFAULT_MIDI_PROGRAM,
    ):
        super().__init__(parent)
        self.mix_audio = mix_audio
        self.sample_rate = sample_rate
        self.playback_speed = max(0.01, float(playback_speed))
        self.note_snapshot = [copy.deepcopy(n) for n in notes]
        self.midi_time_mode = str(midi_time_mode).lower()
        if self.midi_time_mode not in ("real", "ideal"):
            raise ValueError(f"Unsupported MIDI time mode: {midi_time_mode}")

        # 再生開始オフセット（秒）
        self.start_offset = max(0.0, float(start_offset))

        # ノートオン・オフイベント（時刻[s], Noteオブジェクト）のリスト
        ontime_attr = f"{self.midi_time_mode}_ontime"
        offtime_attr = f"{self.midi_time_mode}_offtime"
        self.events: list[tuple[float, bool, object]] = []
        note_timings: list[tuple[float, float, object]] = []
        for n in sorted(
            self.note_snapshot,
            key=lambda nn: float(getattr(nn, ontime_attr, getattr(nn, "onset", 0.0))),
        ):
            on_t = float(getattr(n, ontime_attr, getattr(n, "onset", 0.0)))
            off_t = float(getattr(n, offtime_attr, getattr(n, "offset", on_t)))
            note_timings.append((on_t, off_t, n))
            self.events.append((on_t, True, n))
            self.events.append((off_t, False, n))
        self.events.sort(key=lambda x: x[0])

        # 再開時に、start_offset 以前のイベントはスキップ
        self.start_event_index = 0
        for idx, (t, _, _) in enumerate(self.events):
            if t >= self.start_offset:
                self.start_event_index = idx
                break
        else:
            # すべて start_offset より前なら末尾から開始（実質何も送らない）
            self.start_event_index = len(self.events)

        self.beat_events: list[tuple[float, bool]] = []
        for ev in list(beat_events or []):
            try:
                if isinstance(ev, tuple):
                    t, is_downbeat = ev[0], ev[1]
                else:
                    t = getattr(ev, "time_sec", getattr(ev, "beat_time", None))
                    is_downbeat = getattr(ev, "is_downbeat", False)
                if t is None:
                    continue
                self.beat_events.append((float(t), bool(is_downbeat)))
            except Exception:
                continue
        self.beat_events.sort(key=lambda item: float(item[0]))
        self.start_beat_event_index = 0
        for idx, (t, _is_downbeat) in enumerate(self.beat_events):
            if float(t) >= self.start_offset:
                self.start_beat_event_index = idx
                break
        else:
            self.start_beat_event_index = len(self.beat_events)

        # 再生全体の長さ（音声とMIDIの最大値）
        audio_duration = 0.0
        if self.mix_audio is not None:
            audio_duration = self.mix_audio.shape[0] / float(self.sample_rate)
        audio_duration_song = audio_duration * self.playback_speed
        midi_duration = max((t for t, _, _ in self.events), default=0.0)
        self.total_duration = max(audio_duration_song, midi_duration)
        if total_duration_override is not None and total_duration_override > 0:
            self.total_duration = float(total_duration_override)

        # start_offset が終端を超えている場合はクランプ
        if self.start_offset > self.total_duration:
            self.start_offset = self.total_duration

        # 音符の途中から再生する場合は、開始時点で発音中の音符を補完する。
        # オンセットちょうどは通常のイベント処理に任せ、note_on の重複を避ける。
        self.active_notes_at_start = [
            note
            for on_t, off_t, note in note_timings
            if on_t < self.start_offset < off_t
        ]

        # MIDI 出力ポート
        if midi_out is not None:
            self.midi_out = midi_out
            self.midi_port_name = midi_port_name
            self._owns_midi_port = False
        else:
            self.midi_out, self.midi_port_name = open_default_midi_output()
            self._owns_midi_port = True

        if self.midi_out is None:
            print('MIDI output is None (cannot open MIDI device).')

        # 制御用フラグ
        self._stop_flag = False
        self._pause_flag = False
        self.paused_time: float = 0.0

        # ★ 再生中に UI から変更されるフラグ・音量
        self.midi_enabled: bool = True       # チェックボックスと連動
        self.midi_volume: float = 1.0        # 50を基準音量とするスライダと連動
        self.midi_pan_value: int = 64        # 0=left, 64=center, 127=right
        self.midi_program: int = max(0, min(127, int(midi_program)))
        self.metronome_enabled: bool = bool(metronome_enabled)
        self.metronome_volume: float = max(0.0, float(metronome_volume))

    def _send_all_notes_off(self) -> None:
        if self.midi_out is None:
            return
        for channel in (0, METRONOME_CHANNEL):
            self.midi_out.send(
                mido.Message('control_change', control=123, value=0, channel=int(channel))
            )

    def _send_note_on(self, note) -> None:
        if self.midi_out is None:
            return
        velocity = int(round(int(getattr(note, "velocity", 64)) * self.midi_volume))
        velocity = max(0, min(127, velocity))
        self.midi_out.send(
            mido.Message("note_on", note=int(note.pitch), velocity=velocity, channel=0)
        )

    def _send_note_off(self, note) -> None:
        if self.midi_out is None:
            return
        self.midi_out.send(
            mido.Message("note_off", note=int(note.pitch), velocity=0, channel=0)
        )

    def _send_metronome_note_on(self, is_downbeat: bool) -> Optional[int]:
        if self.midi_out is None:
            return None
        note = METRONOME_DOWNBEAT_NOTE if bool(is_downbeat) else METRONOME_BEAT_NOTE
        base_velocity = 112 if bool(is_downbeat) else 84
        velocity = int(round(base_velocity * max(0.0, float(self.metronome_volume))))
        if velocity <= 0:
            return None
        velocity = min(127, velocity)
        self.midi_out.send(
            mido.Message("note_on", note=int(note), velocity=velocity, channel=METRONOME_CHANNEL)
        )
        return int(note)

    def _send_metronome_note_off(self, note: int) -> None:
        if self.midi_out is None:
            return
        self.midi_out.send(
            mido.Message("note_off", note=int(note), velocity=0, channel=METRONOME_CHANNEL)
        )

    def _send_midi_pan(self) -> None:
        if self.midi_out is None:
            return
        pan = max(0, min(127, int(self.midi_pan_value)))
        self.midi_out.send(
            mido.Message("control_change", control=10, value=pan, channel=0)
        )

    def _send_midi_program(self) -> None:
        if self.midi_out is None:
            return
        program = max(0, min(127, int(self.midi_program)))
        self.midi_out.send(mido.Message("program_change", program=program, channel=0))

    def set_midi_program(self, program: int) -> None:
        self.midi_program = max(0, min(127, int(program)))
        if self.midi_out is None or not self.midi_enabled:
            return
        try:
            self._send_midi_program()
        except Exception as e:
            print(f"MIDI program update error: {e}", file=sys.stderr)

    def set_midi_routing(self, enabled: bool, pan_value: int) -> None:
        self.midi_enabled = bool(enabled)
        self.midi_pan_value = max(0, min(127, int(pan_value)))
        if self.midi_out is None:
            return
        try:
            if self.midi_enabled:
                self._send_midi_program()
                self._send_midi_pan()
            else:
                self._send_all_notes_off()
        except Exception as e:
            print(f"MIDI routing update error: {e}", file=sys.stderr)

    # ---- 外部から呼ぶ制御メソッド ----

    def request_stop(self):
        """再生停止を要求（非同期）。"""
        self._stop_flag = True
        self._pause_flag = False  # 明示的に「停止」扱い
        try:
            sd.stop()
        except Exception:
            pass
        # 全ノートオフ
        if self.midi_out is not None:
            try:
                self._send_all_notes_off()
            except Exception:
                pass

    def request_pause(self):
        """一時停止を要求。内部では stop と同様ループを抜けるが、停止時刻を保存する。"""
        self._stop_flag = True
        self._pause_flag = True
        try:
            sd.stop()
        except Exception:
            pass
        if self.midi_out is not None:
            try:
                self._send_all_notes_off()
            except Exception:
                pass

    # ---- スレッド本体 ----
    def run(self):
        # 音声再生開始：start_offset から再生
        play_audio = False

        if self.mix_audio is not None:
            try:
                sd.stop()
                # start_offset に対応するサンプル位置から再生（スロー時は引き伸ばし分を考慮）
                start_sample = int((self.start_offset / self.playback_speed) * self.sample_rate)
                start_sample = max(0, start_sample)

                if start_sample < int(self.mix_audio.shape[0]):
                    sd.play(self.mix_audio[start_sample:], self.sample_rate, blocking=False)
                    play_audio = True
                else:
                    play_audio = False
            except Exception as e:
                play_audio = False
                print(f"Audio playback error: {e}", file=sys.stderr)

        start_time_perf = time.perf_counter()
        event_index = self.start_event_index
        num_events = len(self.events)
        beat_event_index = self.start_beat_event_index
        num_beat_events = len(self.beat_events)
        pending_metronome_note_offs: list[tuple[float, int]] = []

        if self.midi_out is not None and self.midi_enabled:
            try:
                self._send_midi_program()
                self._send_midi_pan()
                for note in self.active_notes_at_start:
                    self._send_note_on(note)
            except Exception as e:
                print(f"MIDI setup error: {e}", file=sys.stderr)

        last_emit = 0.0
        emit_interval = 0.03  # 約30msごとに UI 更新

        while True:
            now_perf = time.perf_counter() - start_time_perf
            now_global = self.start_offset + now_perf * self.playback_speed  # ★ 絶対時間（0からトータルまで）

            # 停止・一時停止チェック
            if self._stop_flag:
                if self._pause_flag:
                    # 一時停止時刻を記録
                    self.paused_time = now_global
                break

            if now_global > self.total_duration + 1.0:
                break

            # 進行状況のシグナル（絶対時間）
            if now_global - last_emit >= emit_interval:
                self.progress.emit(now_global)
                last_emit = now_global

            if self.midi_out is not None:
                keep_pending = []
                for off_time, note in pending_metronome_note_offs:
                    if float(off_time) <= now_global:
                        try:
                            self._send_metronome_note_off(int(note))
                        except Exception as e:
                            print(f"MIDI send error (metronome note_off): {e}", file=sys.stderr)
                    else:
                        keep_pending.append((float(off_time), int(note)))
                pending_metronome_note_offs = keep_pending

            while (
                self.midi_out is not None
                and beat_event_index < num_beat_events
                and self.beat_events[beat_event_index][0] <= now_global
            ):
                beat_time, is_downbeat = self.beat_events[beat_event_index]
                if self.metronome_enabled:
                    try:
                        note = self._send_metronome_note_on(bool(is_downbeat))
                        if note is not None:
                            pending_metronome_note_offs.append((
                                float(beat_time) + METRONOME_NOTE_DURATION_SEC,
                                int(note),
                            ))
                    except Exception as e:
                        print(f"MIDI send error (metronome note_on): {e}", file=sys.stderr)
                beat_event_index += 1

            # 到達済みのイベントを処理
            while event_index < num_events and self.events[event_index][0] <= now_global:
                target_time, is_on, note = self.events[event_index]

                # MIDI送信
                if self.midi_out is not None and self.midi_enabled:
                    if is_on:
                        try:
                            self._send_note_on(note)
                        except Exception as e:
                            print(f"MIDI send error (note_on): {e}", file=sys.stderr)
                    else:
                        try:
                            self._send_note_off(note)
                        except Exception as e:
                            print(f"MIDI send error (note_off): {e}", file=sys.stderr)

                event_index += 1

            # 次のイベントまたは次フレームまで待つ
            if event_index < num_events:
                wait_sec = self.events[event_index][0] - now_global
                if wait_sec > 0:
                    time.sleep(min(0.001, wait_sec))
            else:
                time.sleep(0.001)

        # ループを抜けたあと、停止が「自然終了」か「一時停止／停止」かにかかわらず、
        # audio 側をブロックしない程度に待つ（stop() 済みなら即 return のはず）
        if play_audio:
            try:
                sd.wait()
            except Exception:
                pass

        # 最後にノートを切っておく（pause 時は既に止めているが二重でも問題なし）
        if self.midi_out is not None:
            try:
                self._send_all_notes_off()
            except Exception:
                pass
