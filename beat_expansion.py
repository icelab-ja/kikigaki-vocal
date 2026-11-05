from __future__ import annotations

import itertools
from dataclasses import dataclass

from DataRepresentation import BDB


@dataclass(frozen=True)
class BeatExpansionEntry:
    beat_time: float
    is_downbeat: bool


def _sorted_entries_from_bdb(bdb: BDB) -> list[BeatExpansionEntry]:
    return [
        BeatExpansionEntry(
            beat_time=float(beat_time),
            is_downbeat=bool(getattr(bdb.beats[beat_id], "is_downbeat", False)),
        )
        for beat_time, beat_id in sorted(list(bdb.index), key=lambda item: float(item[0]))
    ]


def _infer_edge_beat_interval(beats: list[BeatExpansionEntry], *, side: str) -> float:
    if len(beats) < 2:
        return 0.5
    if side == "front":
        interval = beats[1].beat_time - beats[0].beat_time
    elif side == "back":
        interval = beats[-1].beat_time - beats[-2].beat_time
    else:
        raise ValueError("side must be 'front' or 'back'")
    return max(float(interval), 1e-6)


def _infer_downbeat_cycle_from_indices(indices: list[int]) -> int:
    if len(indices) < 2:
        return 4
    spans = [right - left for left, right in zip(indices, indices[1:]) if right > left]
    return max(1, spans[0]) if spans else 4


def _infer_front_downbeat_phase(beats: list[BeatExpansionEntry]) -> tuple[int, int]:
    downbeat_indices = [idx for idx, beat in enumerate(beats) if beat.is_downbeat]
    if not downbeat_indices:
        return 4, 0

    front_bar_len = _infer_downbeat_cycle_from_indices(downbeat_indices)
    first_downbeat_idx = downbeat_indices[0]
    if first_downbeat_idx == 0:
        return front_bar_len, 0
    first_phase = (front_bar_len - (first_downbeat_idx % front_bar_len)) % front_bar_len
    return front_bar_len, first_phase


def _infer_back_downbeat_info(beats: list[BeatExpansionEntry]) -> tuple[int, int]:
    downbeat_indices = [idx for idx, beat in enumerate(beats) if beat.is_downbeat]
    if not downbeat_indices:
        return 4, len(beats) - 1

    last_downbeat_idx = downbeat_indices[-1]
    if len(downbeat_indices) >= 2:
        back_bar_len = max(1, downbeat_indices[-1] - downbeat_indices[-2])
    else:
        back_bar_len = 4
    return back_bar_len, last_downbeat_idx


def expand_beat_entries_to_cover_times(
    beats: list[BeatExpansionEntry],
    *,
    note_start: float | None = None,
    note_end: float | None = None,
    end_time: float | None = None,
) -> list[BeatExpansionEntry]:
    if not beats:
        beats = [BeatExpansionEntry(0.0, True), BeatExpansionEntry(0.5, False)]

    expanded = sorted(list(beats), key=lambda beat: float(beat.beat_time))
    if len(expanded) < 2:
        last_time = float(expanded[-1].beat_time)
        expanded.append(BeatExpansionEntry(last_time + 0.5, False))

    front_interval = _infer_edge_beat_interval(expanded, side="front")
    back_interval = _infer_edge_beat_interval(expanded, side="back")
    front_bar_len, first_phase = _infer_front_downbeat_phase(expanded)
    back_bar_len, last_downbeat_idx = _infer_back_downbeat_info(expanded)

    prepended_count = 0
    while True:
        first = expanded[0]
        new_time = first.beat_time - front_interval
        if new_time < 0.0:
            break
        prepended_count += 1
        new_phase = (first_phase - prepended_count) % front_bar_len
        expanded.insert(0, BeatExpansionEntry(new_time, new_phase == 0))
        if new_time == 0.0:
            break

    if expanded[0].beat_time > 0.0:
        prepended_count += 1
        new_phase = (first_phase - prepended_count) % front_bar_len
        expanded.insert(0, BeatExpansionEntry(0.0, new_phase == 0))

    expanded[0] = BeatExpansionEntry(0.0, True)

    target_time = end_time
    if target_time is None:
        target_time = note_end
    if target_time is None:
        target_time = expanded[-1].beat_time
    if note_start is not None:
        target_time = max(float(target_time), float(note_start))
    target_time = float(target_time)

    appended_count = 0
    original_len = len(beats)
    while expanded[-1].beat_time <= target_time:
        last = expanded[-1]
        appended_count += 1
        new_index = (original_len - 1) + appended_count
        beat_count_from_last_downbeat = new_index - last_downbeat_idx
        expanded.append(
            BeatExpansionEntry(
                beat_time=last.beat_time + back_interval,
                is_downbeat=(beat_count_from_last_downbeat % back_bar_len == 0),
            )
        )

    return expanded


def replace_bdb_with_expanded_entries(bdb: BDB, beats: list[BeatExpansionEntry]) -> None:
    bdb.beats = {}
    bdb.index = []
    bdb._id_counter = itertools.count(0)
    for beat in beats:
        bdb.add_beat(float(beat.beat_time), bool(beat.is_downbeat))
    bdb.index.sort(key=lambda item: float(item[0]))


def expand_bdb_to_cover_times(
    bdb: BDB,
    *,
    note_start: float | None = None,
    note_end: float | None = None,
    end_time: float | None = None,
) -> None:
    expanded = expand_beat_entries_to_cover_times(
        _sorted_entries_from_bdb(bdb),
        note_start=note_start,
        note_end=note_end,
        end_time=end_time,
    )
    replace_bdb_with_expanded_entries(bdb, expanded)
