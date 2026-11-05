import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

import mido


FLUIDSYNTH_SF2_PATH = Path(__file__).with_name("MuseScore_General.sf2")
FLUIDSYNTH_GAIN = 2.0
_FLUIDSYNTH_PROCESS: Optional[subprocess.Popen] = None


class FluidSynthShellPort:
    """
    FluidSynth のシェルコマンド(stdin)経由で MIDI イベントを送る簡易ポート。
    mido output port と同じ send() 形状だけを満たす。
    """

    def __init__(self, process: subprocess.Popen):
        self._process = process
        self._lock = threading.Lock()
        self.name = "FluidSynth (shell)"

    def _write_command(self, command: str) -> None:
        proc = self._process
        if proc is None or proc.poll() is not None:
            raise RuntimeError("FluidSynth process is not running.")
        if proc.stdin is None:
            raise RuntimeError("FluidSynth stdin is not available.")
        with self._lock:
            proc.stdin.write((command + "\n").encode("utf-8"))
            proc.stdin.flush()

    def send(self, msg) -> None:
        msg_type = getattr(msg, "type", "")
        ch = int(getattr(msg, "channel", 0))

        if msg_type == "note_on":
            note = int(getattr(msg, "note", 0))
            vel = int(getattr(msg, "velocity", 0))
            if vel <= 0:
                self._write_command(f"noteoff {ch} {note}")
            else:
                self._write_command(f"noteon {ch} {note} {vel}")
            return

        if msg_type == "note_off":
            note = int(getattr(msg, "note", 0))
            self._write_command(f"noteoff {ch} {note}")
            return

        if msg_type == "control_change":
            control = int(getattr(msg, "control", 0))
            value = int(getattr(msg, "value", 0))
            self._write_command(f"cc {ch} {control} {value}")
            return

        if msg_type == "program_change":
            program = int(getattr(msg, "program", 0))
            self._write_command(f"prog {ch} {program}")
            return

    def close(self) -> None:
        # プロセス管理は stop_fluidsynth_server() が担当する。
        return


def open_default_midi_output():
    """
    FluidSynth MIDI 出力ポートを開く。
    見つからない場合はエラーを出力して None を返す。
    """
    try:
        names = mido.get_output_names()
    except Exception as e:
        print(f"Failed to list MIDI output ports: {e}", file=sys.stderr)
        return None, None

    preferred_name = _find_fluidsynth_output_name(names)

    if preferred_name is None:
        if not start_fluidsynth_server():
            return None, None

        # Windows では -n で起動し、シェル経由送信を使う。
        if _use_fluidsynth_shell_mode():
            if _FLUIDSYNTH_PROCESS is not None and _FLUIDSYNTH_PROCESS.poll() is None:
                return FluidSynthShellPort(_FLUIDSYNTH_PROCESS), "FluidSynth (shell)"
            return None, None

        try:
            names = mido.get_output_names()
        except Exception as e:
            print(f"Failed to list MIDI output ports after starting FluidSynth: {e}", file=sys.stderr)
            return None, None
        preferred_name = _find_fluidsynth_output_name(names)

        if preferred_name is None:
            available = ", ".join(names) if names else "(none)"
            print(
                f"FluidSynth MIDI output was not found. Available MIDI outputs: {available}",
                file=sys.stderr,
            )
            return None, None

    try:
        port = mido.open_output(preferred_name)
        return port, preferred_name
    except Exception as e:
        print(f"Failed to open FluidSynth MIDI output '{preferred_name}': {e}", file=sys.stderr)
        return None, None


def _find_fluidsynth_output_name(names) -> Optional[str]:
    for name in names:
        normalized = "".join(ch for ch in str(name).lower() if ch.isalnum())
        if "fluidsynth" in normalized:
            return name
    return None


def _use_fluidsynth_shell_mode() -> bool:
    return sys.platform.lower().startswith("win")


def _fluidsynth_driver_candidates() -> List[tuple[str, str]]:
    platform = sys.platform.lower()
    if platform.startswith("win"):
        return [("wasapi", "winmidi"), ("dsound", "winmidi"), ("waveout", "winmidi")]
    if platform == "darwin":
        return [("coreaudio", "coremidi")]
    if platform.startswith("linux"):
        return [("pulseaudio", "alsa"), ("alsa", "alsa"), ("sdl2", "alsa")]
    return [("coreaudio", "coremidi"), ("wasapi", "winmidi"), ("pulseaudio", "alsa")]


def start_fluidsynth_server(timeout_sec: float = 5.0) -> bool:
    global _FLUIDSYNTH_PROCESS

    if _FLUIDSYNTH_PROCESS is not None:
        if _FLUIDSYNTH_PROCESS.poll() is None:
            return True
        _FLUIDSYNTH_PROCESS = None

    if not FLUIDSYNTH_SF2_PATH.exists():
        print(f"FluidSynth soundfont was not found: {FLUIDSYNTH_SF2_PATH}", file=sys.stderr)
        return False

    fluidsynth_path = shutil.which("fluidsynth")
    if fluidsynth_path is None:
        print("FluidSynth executable was not found. Please install FluidSynth and ensure it is on PATH.", file=sys.stderr)
        return False

    deadline = time.monotonic() + float(timeout_sec)
    last_names = []
    failures = []
    shell_mode = _use_fluidsynth_shell_mode()

    for audio_driver, midi_driver in _fluidsynth_driver_candidates():
        cmd = [
            fluidsynth_path,
            "-a",
            audio_driver,
            "-g",
            f"{FLUIDSYNTH_GAIN:.3g}",
        ]
        if shell_mode:
            # MIDI input デバイスなし環境向け: shell(stdin)から直接イベント送信する。
            cmd += ["-n"]
        else:
            cmd += ["-m", midi_driver]
        cmd += [str(FLUIDSYNTH_SF2_PATH)]

        try:
            _FLUIDSYNTH_PROCESS = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        except Exception as e:
            failures.append(f"{audio_driver}/{midi_driver}: start failed ({e})")
            _FLUIDSYNTH_PROCESS = None
            continue

        if shell_mode:
            # 起動直後に落ちなければ shell モードとして成功扱い。
            time.sleep(0.3)
            if _FLUIDSYNTH_PROCESS.poll() is None:
                return True
            code = _FLUIDSYNTH_PROCESS.returncode
            failures.append(f"{audio_driver}/{midi_driver}: exited early (exit code {code})")
            _FLUIDSYNTH_PROCESS = None
            continue

        while time.monotonic() < deadline:
            if _FLUIDSYNTH_PROCESS.poll() is not None:
                code = _FLUIDSYNTH_PROCESS.returncode
                failures.append(f"{audio_driver}/{midi_driver}: exited early (exit code {code})")
                _FLUIDSYNTH_PROCESS = None
                break
            try:
                last_names = mido.get_output_names()
            except Exception:
                last_names = []
            if _find_fluidsynth_output_name(last_names) is not None:
                return True
            time.sleep(0.1)

        # Total timeout reached while process is still running but no MIDI port appeared.
        if _FLUIDSYNTH_PROCESS is not None and _FLUIDSYNTH_PROCESS.poll() is None:
            stop_fluidsynth_server()
            break

    available = ", ".join(last_names) if last_names else "(none)"
    attempted = ", ".join(f"{a}/{m}" for a, m in _fluidsynth_driver_candidates())
    failure_text = "; ".join(failures) if failures else "no start attempt was completed"
    print(
        (
            f"FluidSynth MIDI output did not appear within {timeout_sec:.1f}s. "
            f"Attempted driver pairs: {attempted}. "
            f"Failures: {failure_text}. "
            f"Available MIDI outputs: {available}"
        ),
        file=sys.stderr,
    )
    return False


def stop_fluidsynth_server() -> None:
    global _FLUIDSYNTH_PROCESS

    proc = _FLUIDSYNTH_PROCESS
    _FLUIDSYNTH_PROCESS = None
    if proc is None or proc.poll() is not None:
        return

    try:
        proc.terminate()
        proc.wait(timeout=2.0)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
