from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from playback import SyncAudioMidiPlayer, _time_stretch_audio, stop_audio_output


@dataclass(frozen=True)
class PlaybackSessionCallbacks:
    on_progress: Callable[[float], None]
    on_finished: Callable[[bool, float], None]


@dataclass(frozen=True)
class PlaybackStartResult:
    start_offset: float
    total_duration: float
    was_paused: bool
    reset_to_start: bool


class PlaybackSessionController:
    def __init__(self, callbacks: PlaybackSessionCallbacks):
        self.callbacks = callbacks
        self.player: Optional[SyncAudioMidiPlayer] = None
        self.is_paused = False
        self.paused_time = 0.0
        self.is_slider_dragging = False
        self.total_seconds = 0.0
        self.last_total_seconds = 0.0
        self._last_stretch_speed = 1.0
        self._last_stretch_signature = None
        self._last_stretched_audio: Optional[np.ndarray] = None

    @property
    def is_playing(self) -> bool:
        return self.player is not None

    @property
    def display_total_seconds(self) -> float:
        if self.total_seconds > 0.0:
            return float(self.total_seconds)
        return float(self.last_total_seconds)

    def _prepare_audio(
        self,
        mix_audio: Optional[np.ndarray],
        speed: float,
        mix_signature,
    ) -> Optional[np.ndarray]:
        if mix_audio is None or abs(float(speed) - 1.0) < 1e-6:
            return mix_audio
        if (
            self._last_stretched_audio is not None
            and self._last_stretch_signature == mix_signature
            and abs(self._last_stretch_speed - float(speed)) < 1e-6
        ):
            return self._last_stretched_audio

        stretched = _time_stretch_audio(mix_audio, float(speed))
        self._last_stretched_audio = stretched
        self._last_stretch_speed = float(speed)
        self._last_stretch_signature = mix_signature
        return stretched

    def start(
        self,
        *,
        mix_audio: Optional[np.ndarray],
        mix_signature,
        sample_rate: int,
        notes,
        playback_speed: float,
        total_duration: float,
        parent,
        midi_out,
        midi_port_name,
        beat_events,
        metronome_enabled: bool,
        metronome_volume: float,
        midi_time_mode: str,
        midi_program: int,
        midi_enabled: bool,
        midi_pan_value: int,
        midi_volume: float,
    ) -> PlaybackStartResult:
        if self.player is not None:
            self.dispose_player()

        speed = float(playback_speed)
        mix_audio = self._prepare_audio(mix_audio, speed, mix_signature)
        was_paused = bool(self.is_paused)
        start_offset = float(self.paused_time) if was_paused else 0.0
        reset_to_start = bool(
            total_duration > 0.0 and start_offset >= float(total_duration) - 1e-3
        )
        if reset_to_start:
            start_offset = 0.0
            was_paused = False
            self.is_paused = False
            self.paused_time = 0.0

        self.total_seconds = float(total_duration)
        self.last_total_seconds = self.total_seconds

        player = SyncAudioMidiPlayer(
            mix_audio=mix_audio,
            sample_rate=int(sample_rate),
            notes=notes,
            start_offset=start_offset,
            playback_speed=speed,
            total_duration_override=total_duration,
            parent=parent,
            midi_out=midi_out,
            midi_port_name=midi_port_name,
            beat_events=beat_events,
            metronome_enabled=metronome_enabled,
            metronome_volume=metronome_volume,
            midi_time_mode=midi_time_mode,
            midi_program=midi_program,
        )
        player.set_midi_routing(midi_enabled, midi_pan_value)
        player.midi_volume = float(midi_volume)
        player.progress.connect(self.callbacks.on_progress)
        player.finished.connect(lambda current=player: self._handle_finished(current))

        self.player = player
        self.is_paused = False
        player.start()
        return PlaybackStartResult(
            start_offset=start_offset,
            total_duration=float(player.total_duration),
            was_paused=was_paused,
            reset_to_start=reset_to_start,
        )

    def _handle_finished(self, player: SyncAudioMidiPlayer) -> None:
        if player is not self.player:
            player.deleteLater()
            return

        paused = bool(player._pause_flag)
        paused_time = float(getattr(player, "paused_time", 0.0))
        player.wait()
        player.deleteLater()
        self.player = None

        if paused:
            self.is_paused = True
            self.paused_time = paused_time
        else:
            self.is_paused = False
            self.paused_time = 0.0
        self.callbacks.on_finished(paused, self.paused_time)

    def request_stop(self) -> None:
        if self.player is not None:
            try:
                self.player.request_stop()
            except Exception:
                pass
        else:
            try:
                stop_audio_output()
            except Exception:
                pass
        self.is_paused = False
        self.paused_time = 0.0

    def request_pause(self) -> None:
        if self.player is not None:
            self.player.request_pause()

    def dispose_player(self) -> None:
        player = self.player
        if player is None:
            return
        player.request_stop()
        player.wait()
        player.deleteLater()
        self.player = None

    def set_paused_position(self, time_sec: float) -> None:
        self.is_paused = True
        self.paused_time = float(time_sec)

    def begin_slider_drag(self) -> bool:
        self.is_slider_dragging = not self.is_playing
        return self.is_slider_dragging

    def end_slider_drag(self) -> None:
        self.is_slider_dragging = False

    def update_midi_routing(self, enabled: bool, pan_value: int) -> None:
        if self.player is not None:
            self.player.set_midi_routing(enabled, pan_value)

    def update_midi_volume(self, volume: float) -> None:
        if self.player is not None:
            self.player.midi_volume = float(volume)

    def update_midi_program(self, program: int) -> None:
        if self.player is not None:
            self.player.set_midi_program(int(program))

    def update_metronome(self, enabled: bool, volume: float) -> None:
        if self.player is not None:
            self.player.metronome_enabled = bool(enabled)
            self.player.metronome_volume = float(volume)
