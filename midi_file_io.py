from bisect import bisect_right
from typing import List

import mido

from DataRepresentation import TPQN
from dtpr_to_qpr import infer_last_stime, infer_meter_events, infer_spqn_events, sorted_beats


def _build_tempo_map(mid: mido.MidiFile):
    tempo_changes = [(0, 500000)]  # (abs_tick, tempo)
    abs_tick = 0

    merged = mido.merge_tracks(mid.tracks)
    for msg in merged:
        abs_tick += msg.time
        if msg.type == "set_tempo":
            if abs_tick == tempo_changes[-1][0]:
                tempo_changes[-1] = (abs_tick, msg.tempo)
            else:
                tempo_changes.append((abs_tick, msg.tempo))

    ticks = []
    tempos = []
    secs = []
    acc_sec = 0.0
    for i, (tick, tempo) in enumerate(tempo_changes):
        if i > 0:
            prev_tick = ticks[-1]
            prev_tempo = tempos[-1]
            acc_sec += mido.tick2second(tick - prev_tick, mid.ticks_per_beat, prev_tempo)
        ticks.append(int(tick))
        tempos.append(int(tempo))
        secs.append(float(acc_sec))

    return ticks, tempos, secs


def _ticks_to_seconds(tick: int, ticks, tempos, secs, ticks_per_beat: int) -> float:
    idx = bisect_right(ticks, tick) - 1
    if idx < 0:
        idx = 0
    base_tick = ticks[idx]
    base_sec = secs[idx]
    tempo = tempos[idx]
    return base_sec + mido.tick2second(tick - base_tick, ticks_per_beat, tempo)


def _compute_midi_max_tick(mid: mido.MidiFile) -> int:
    max_tick = 0
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += int(msg.time)
        if tick > max_tick:
            max_tick = tick
    return int(max_tick)


def _extract_beat_times_from_midi(path: str) -> List[float]:
    mid = mido.MidiFile(path)
    ticks, tempos, secs = _build_tempo_map(mid)
    ticks_per_beat = int(mid.ticks_per_beat)

    max_tick = _compute_midi_max_tick(mid)
    last_beat_tick = ((max_tick // ticks_per_beat) + 1) * ticks_per_beat
    beat_ticks = list(range(0, last_beat_tick + ticks_per_beat, ticks_per_beat))
    beat_times = [
        float(_ticks_to_seconds(bt, ticks, tempos, secs, ticks_per_beat))
        for bt in beat_ticks
    ]
    return beat_times


def read_midi_notes(path: str) -> list[tuple[int, float, float]]:
    mid = mido.MidiFile(path)
    ticks, tempos, secs = _build_tempo_map(mid)
    notes: list[tuple[int, float, float]] = []

    for track in mid.tracks:
        current_tick = 0
        active_notes: dict[int, tuple[int, int]] = {}
        for msg in track:
            current_tick += int(msg.time)
            if msg.is_meta:
                continue
            if msg.type == "note_on" and int(msg.velocity) > 0:
                active_notes[int(msg.note)] = (current_tick, int(msg.velocity))
            elif msg.type == "note_off" or (msg.type == "note_on" and int(msg.velocity) == 0):
                if int(msg.note) not in active_notes:
                    continue
                onset_tick, _ = active_notes.pop(int(msg.note))
                onset_sec = _ticks_to_seconds(
                    onset_tick,
                    ticks,
                    tempos,
                    secs,
                    mid.ticks_per_beat,
                )
                offset_sec = _ticks_to_seconds(
                    current_tick,
                    ticks,
                    tempos,
                    secs,
                    mid.ticks_per_beat,
                )
                if float(offset_sec) <= float(onset_sec):
                    continue
                notes.append((int(msg.note), float(onset_sec), float(offset_sec)))

    return notes


def export_notes_to_midi(notes, path: str, time_mode: str = "real") -> None:
    mode = str(time_mode).lower()
    if mode not in ("real", "ideal"):
        raise ValueError(f"Unsupported MIDI time mode: {time_mode}")

    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    ticks_per_beat = 480
    mid.ticks_per_beat = ticks_per_beat
    tempo = mido.bpm2tempo(120)

    min_dur_sec = 0.02

    def sec_to_tick(sec: float) -> int:
        return int(round(mido.second2tick(sec, ticks_per_beat, tempo)))

    current_tick = 0

    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    skipped = 0

    onset_attr = f"{mode}_ontime"
    offset_attr = f"{mode}_offtime"

    for n in notes:
        onset_sec = float(getattr(n, onset_attr))
        offset_sec = float(getattr(n, offset_attr))
        dur_sec = offset_sec - onset_sec

        # --- 不正ノートはスキップ ---
        if offset_sec <= onset_sec:
            skipped += 1
            continue
        if dur_sec < min_dur_sec:
            skipped += 1
            continue

        pitch = int(n.pitch)
        velocity = 64

        # MIDI仕様に合わせてクランプ
        pitch = max(0, min(127, pitch))
        velocity = max(1, min(127, velocity))

        onset_tick = sec_to_tick(onset_sec)
        offset_tick = sec_to_tick(offset_sec)

        # tick の単調増加を保証（丸め誤差や編集の並び崩れ対策）
        if onset_tick < current_tick:
            onset_tick = current_tick

        # ここまで来たノートは dur_sec >= min_dur_sec のはずだが、
        # tick変換の丸めで同tickになる可能性はあるので +1tick は保証
        if offset_tick <= onset_tick:
            skipped += 1
            continue  # 「短すぎる」扱いで捨てる（要件：不正は無視）

        delta_on = onset_tick - current_tick
        track.append(mido.Message("note_on", note=pitch, velocity=velocity, time=delta_on))

        delta_off = offset_tick - onset_tick
        track.append(mido.Message("note_off", note=pitch, velocity=velocity, time=delta_off))

        current_tick = offset_tick

    mid.save(path)

    # 任意：デバッグ用（不要なら削除）
    if skipped:
        print(f"[export_to_midi] skipped invalid notes: {skipped}")


def _scale_dtpr_tick(tick: int, ticks_per_beat: int) -> int:
    if int(ticks_per_beat) == TPQN:
        return int(tick)
    return int(round(int(tick) * int(ticks_per_beat) / float(TPQN)))


def _append_score_midi_meta_events(track: mido.MidiTrack, bdb, notes, ticks_per_beat: int) -> None:
    meta_events = []

    if bdb is None:
        meta_events.append(
            (0, 0, mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
        )
        meta_events.append((0, 1, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0)))
    else:
        beats = sorted_beats(bdb)
        last_stime = infer_last_stime(beats, notes)

        for stime, numer, denom, _barlen in infer_meter_events(beats, last_stime):
            if int(numer) <= 0 or int(denom) <= 0:
                continue
            meta_events.append(
                (
                    _scale_dtpr_tick(stime, ticks_per_beat),
                    0,
                    mido.MetaMessage(
                        "time_signature",
                        numerator=int(numer),
                        denominator=int(denom),
                        time=0,
                    ),
                )
            )

        for stime, spqn, _bpm in infer_spqn_events(beats):
            if float(spqn) <= 0:
                continue
            tempo = int(round(float(spqn) * 1_000_000))
            tempo = max(1, min(16_777_215, tempo))
            meta_events.append(
                (
                    _scale_dtpr_tick(stime, ticks_per_beat),
                    1,
                    mido.MetaMessage("set_tempo", tempo=tempo, time=0),
                )
            )

    if not any(msg.type == "time_signature" for _tick, _priority, msg in meta_events):
        meta_events.append(
            (0, 0, mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
        )
    if not any(msg.type == "set_tempo" for _tick, _priority, msg in meta_events):
        meta_events.append((0, 1, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0)))

    current_tick = 0
    for abs_tick, _priority, msg in sorted(meta_events, key=lambda item: (item[0], item[1])):
        abs_tick = int(abs_tick)
        msg.time = max(0, abs_tick - current_tick)
        track.append(msg)
        current_tick = max(current_tick, abs_tick)


def export_tick_notes_to_midi(notes, path: str, ticks_per_beat: int = 1440, bdb=None) -> None:
    notes = list(notes)
    mid = mido.MidiFile(type=1)
    meta_track = mido.MidiTrack()
    track = mido.MidiTrack()
    mid.tracks.append(meta_track)
    mid.tracks.append(track)

    mid.ticks_per_beat = int(ticks_per_beat)

    current_tick = 0
    skipped = 0

    _append_score_midi_meta_events(meta_track, bdb, notes, int(ticks_per_beat))

    for n in notes:
        onset_tick = int(getattr(n, "abs_ontick"))
        offset_tick = int(getattr(n, "abs_offtick"))

        if offset_tick <= onset_tick:
            skipped += 1
            continue

        pitch = int(n.pitch)
        velocity = 64

        pitch = max(0, min(127, pitch))
        velocity = max(1, min(127, velocity))

        if onset_tick < current_tick:
            onset_tick = current_tick
        if offset_tick <= onset_tick:
            skipped += 1
            continue

        delta_on = onset_tick - current_tick
        track.append(mido.Message("note_on", note=pitch, velocity=velocity, time=delta_on))

        delta_off = offset_tick - onset_tick
        track.append(mido.Message("note_off", note=pitch, velocity=velocity, time=delta_off))

        current_tick = offset_tick

    mid.save(path)

    if skipped:
        print(f"[export_tick_notes_to_midi] skipped invalid notes: {skipped}")
