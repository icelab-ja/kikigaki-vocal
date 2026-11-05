from __future__ import annotations

import argparse
import itertools
import json
from bisect import insort
from pathlib import Path
from typing import Any, Optional, Sequence

from DataRepresentation import Beat, DTPR
from DualTimePianoRoll import dtpr_class_to_dtpr_file, dtpr_file_to_dtpr_class


class ReplayError(RuntimeError):
    pass


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ReplayError(f"Invalid JSON at line {line_no}: {exc}") from exc
            if isinstance(record, dict):
                record["_lineNo"] = line_no
                records.append(record)
    return records


def _note_id(snapshot: dict[str, Any]) -> int:
    if "noteId" in snapshot:
        return int(snapshot["noteId"])
    if "id" in snapshot:
        return int(snapshot["id"])
    raise ReplayError(f"Note snapshot has no noteId/id: {snapshot}")


def _get_first(snapshot: dict[str, Any], *keys: str):
    for key in keys:
        if key in snapshot:
            return snapshot[key]
    return None


def _current_note_snapshot(dtpr: DTPR, note_id: int) -> dict[str, Any]:
    ptpr_note = dtpr.PTPR.notes.get(int(note_id))
    mtpr_note = dtpr.MTPR.notes.get(int(note_id))
    if ptpr_note is None and mtpr_note is None:
        raise ReplayError(f"Cannot patch missing note {note_id}; patch is not a full insertion.")

    pitch = None
    if ptpr_note is not None:
        pitch = int(ptpr_note.pitch)
    elif mtpr_note is not None:
        pitch = int(mtpr_note.pitch)

    snapshot: dict[str, Any] = {
        "noteId": int(note_id),
        "pitch": pitch,
    }
    if ptpr_note is not None:
        snapshot.update(
            {
                "realOnsetSec": float(ptpr_note.real_ontime),
                "realOffsetSec": float(ptpr_note.real_offtime),
                "idealOnsetSec": float(ptpr_note.ideal_ontime),
                "idealOffsetSec": float(ptpr_note.ideal_offtime),
            }
        )
    if mtpr_note is not None:
        snapshot.update(
            {
                "absOnTick": int(mtpr_note.abs_ontick),
                "absOffTick": int(mtpr_note.abs_offtick),
                "onBeatId": int(mtpr_note.on_beat_id),
                "offBeatId": int(mtpr_note.off_beat_id),
                "relOnTick": int(mtpr_note.on_metrical_pos),
                "relOffTick": int(mtpr_note.off_metrical_pos),
            }
        )
    return snapshot


def _merge_note_patch(dtpr: DTPR, note_id: int, patch: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if patch is None:
        return None
    if "noteId" in patch or "id" in patch:
        base = dict(patch)
        base.setdefault("noteId", note_id)
        return base

    base = _current_note_snapshot(dtpr, note_id)
    base.update(patch)
    return base


def _rebuild_note_indices(dtpr: DTPR) -> None:
    dtpr.PTPR.index = sorted(
        (
            (
                float(note.ideal_ontime),
                float(note.ideal_offtime),
                int(note.id),
            )
            for note in dtpr.PTPR.notes.values()
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )
    dtpr.MTPR.index = sorted(
        (
            (
                int(note.abs_ontick),
                int(note.abs_offtick),
                int(note.id),
            )
            for note in dtpr.MTPR.notes.values()
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )


def _refresh_note_id_counter(dtpr: DTPR) -> None:
    max_id = -1
    for note_id in dtpr.PTPR.notes.keys():
        max_id = max(max_id, int(note_id))
    for note_id in dtpr.MTPR.notes.keys():
        max_id = max(max_id, int(note_id))
    dtpr._note_id_counter = itertools.count(max_id + 1)


def _delete_note(dtpr: DTPR, note_id: int) -> None:
    dtpr.PTPR.notes.pop(int(note_id), None)
    dtpr.MTPR.notes.pop(int(note_id), None)
    _rebuild_note_indices(dtpr)
    _refresh_note_id_counter(dtpr)


def _upsert_note(dtpr: DTPR, snapshot: dict[str, Any]) -> None:
    note_id = _note_id(snapshot)
    pitch = _get_first(snapshot, "pitch")
    if pitch is None:
        current = _current_note_snapshot(dtpr, note_id)
        pitch = current.get("pitch")
    if pitch is None:
        raise ReplayError(f"Note {note_id} has no pitch.")

    real_on = _get_first(snapshot, "realOnsetSec", "real_ontime", "onsetSec", "onset")
    real_off = _get_first(snapshot, "realOffsetSec", "real_offtime", "offsetSec", "offset")
    ideal_on = _get_first(snapshot, "idealOnsetSec", "ideal_ontime")
    ideal_off = _get_first(snapshot, "idealOffsetSec", "ideal_offtime")

    if real_on is not None or real_off is not None or ideal_on is not None or ideal_off is not None:
        if real_on is None and ideal_on is None:
            raise ReplayError(f"Note {note_id} has no onset time.")
        if real_off is None and ideal_off is None:
            raise ReplayError(f"Note {note_id} has no offset time.")
        dtpr.PTPR.notes.pop(note_id, None)
        dtpr.PTPR.add_ptpr_note(
            note_id,
            int(pitch),
            None if ideal_on is None else float(ideal_on),
            None if real_on is None else float(real_on),
            None if ideal_off is None else float(ideal_off),
            None if real_off is None else float(real_off),
        )

    abs_on = _get_first(snapshot, "absOnTick", "abs_ontick")
    abs_off = _get_first(snapshot, "absOffTick", "abs_offtick")
    on_beat_id = _get_first(snapshot, "onBeatId", "on_beat_id")
    off_beat_id = _get_first(snapshot, "offBeatId", "off_beat_id")
    rel_on = _get_first(snapshot, "relOnTick", "on_metrical_pos")
    rel_off = _get_first(snapshot, "relOffTick", "off_metrical_pos")
    mtpr_values = [abs_on, abs_off, on_beat_id, off_beat_id, rel_on, rel_off]
    if any(value is not None for value in mtpr_values):
        if any(value is None for value in mtpr_values):
            raise ReplayError(f"Note {note_id} has incomplete MTPR fields: {snapshot}")
        dtpr.MTPR.notes.pop(note_id, None)
        dtpr.MTPR.add_mtpr_note(
            note_id,
            int(pitch),
            int(abs_on),
            int(on_beat_id),
            int(rel_on),
            int(abs_off),
            int(off_beat_id),
            int(rel_off),
        )

    _rebuild_note_indices(dtpr)
    _refresh_note_id_counter(dtpr)


def _apply_note_list_diff(
    dtpr: DTPR,
    *,
    original_notes: Sequence[dict[str, Any]],
    revised_notes: Sequence[dict[str, Any]],
    forward: bool,
) -> None:
    original_by_id = {_note_id(note): dict(note) for note in original_notes}
    revised_by_id = {_note_id(note): dict(note) for note in revised_notes}

    if forward:
        for note_id in sorted(set(original_by_id) - set(revised_by_id)):
            _delete_note(dtpr, note_id)
        for note_id in sorted(revised_by_id):
            _upsert_note(dtpr, revised_by_id[note_id])
    else:
        for note_id in sorted(set(revised_by_id) - set(original_by_id)):
            _delete_note(dtpr, note_id)
        for note_id in sorted(original_by_id):
            _upsert_note(dtpr, original_by_id[note_id])


def _apply_note_patches(
    dtpr: DTPR,
    patches: Sequence[dict[str, Any]],
    *,
    forward: bool,
) -> None:
    for patch in patches:
        note_id = _note_id(patch)
        payload = patch.get("after") if forward else patch.get("before")
        snapshot = _merge_note_patch(dtpr, note_id, payload)
        if snapshot is None:
            _delete_note(dtpr, note_id)
        else:
            _upsert_note(dtpr, snapshot)


def _refresh_beat_counter(dtpr: DTPR) -> None:
    max_id = max((int(beat_id) for beat_id in dtpr.BDB.beats.keys()), default=-1)
    dtpr.BDB._id_counter = itertools.count(max_id + 1)


def _sort_beats(dtpr: DTPR) -> None:
    dtpr.BDB.index.sort(key=lambda item: (float(item[0]), int(item[1])))


def _upsert_beat(dtpr: DTPR, beat_id: int, beat_time: float, is_downbeat: bool) -> None:
    beat_id = int(beat_id)
    if beat_id in dtpr.BDB.beats:
        beat = dtpr.BDB.beats[beat_id]
        beat.set_beat_time(float(beat_time))
        beat.set_downbeat(bool(is_downbeat))
        dtpr.BDB.index = [
            (float(time), int(existing_id))
            for time, existing_id in dtpr.BDB.index
            if int(existing_id) != beat_id
        ]
        insort(dtpr.BDB.index, (float(beat_time), beat_id))
    else:
        beat = Beat(beat_id, float(beat_time), bool(is_downbeat))
        dtpr.BDB.beats[beat_id] = beat
        insort(dtpr.BDB.index, (float(beat_time), beat_id))
    _sort_beats(dtpr)
    _refresh_beat_counter(dtpr)


def _delete_beat(dtpr: DTPR, beat_id: int) -> None:
    beat_id = int(beat_id)
    dtpr.BDB.beats.pop(beat_id, None)
    dtpr.BDB.index = [
        (float(time), int(existing_id))
        for time, existing_id in dtpr.BDB.index
        if int(existing_id) != beat_id
    ]
    _sort_beats(dtpr)
    _refresh_beat_counter(dtpr)


def _apply_beat_change(dtpr: DTPR, record: dict[str, Any], *, forward: bool) -> None:
    operation = str(record.get("operation") or "")
    meta = dict(record.get("meta") or {})
    beat_id = meta.get("beatId")
    if beat_id is None:
        raise ReplayError(f"{operation} at line {record.get('_lineNo')} has no meta.beatId.")

    if operation == "beat_insert":
        if forward:
            _upsert_beat(
                dtpr,
                int(beat_id),
                float(meta["afterTimeSec"]),
                bool(meta.get("isDownbeat", False)),
            )
        else:
            _delete_beat(dtpr, int(beat_id))
        return

    if operation == "beat_delete":
        if forward:
            _delete_beat(dtpr, int(beat_id))
        else:
            _upsert_beat(
                dtpr,
                int(beat_id),
                float(meta["beforeTimeSec"]),
                bool(meta.get("isDownbeat", False)),
            )
        return

    if operation == "beat_move":
        key = "afterTimeSec" if forward else "beforeTimeSec"
        beat_time = meta.get(key)
        if beat_time is None:
            raise ReplayError(f"{operation} at line {record.get('_lineNo')} has no meta.{key}.")
        beat = dtpr.BDB.beats.get(int(beat_id))
        is_downbeat = bool(meta.get("isDownbeat", getattr(beat, "is_downbeat", False)))
        _upsert_beat(dtpr, int(beat_id), float(beat_time), is_downbeat)
        return

    if operation == "beat_downbeat_toggle":
        beat = dtpr.BDB.beats.get(int(beat_id))
        if beat is None:
            raise ReplayError(f"beat_downbeat_toggle references missing beat {beat_id}.")
        value_key = "isDownbeat" if forward else "beforeIsDownbeat"
        if value_key not in meta:
            raise ReplayError(f"beat_downbeat_toggle at line {record.get('_lineNo')} has no meta.{value_key}.")
        beat.set_downbeat(bool(meta[value_key]))
        return


def _apply_record_payload(dtpr: DTPR, record: dict[str, Any], *, forward: bool) -> None:
    operation = str(record.get("operation") or "")
    if operation.startswith("checked_"):
        return

    if operation.startswith("beat_"):
        _apply_beat_change(dtpr, record, forward=forward)

    meta = dict(record.get("meta") or {})
    note_patches = meta.get("notePatches")
    original_notes = list(record.get("originalNotes") or [])
    revised_notes = list(record.get("revisedNotes") or [])

    if note_patches:
        _apply_note_patches(dtpr, list(note_patches), forward=forward)
        return

    if original_notes or revised_notes:
        _apply_note_list_diff(
            dtpr,
            original_notes=original_notes,
            revised_notes=revised_notes,
            forward=forward,
        )
        return

    if operation in {"beat_insert", "beat_delete"}:
        raise ReplayError(
            f"{operation} at line {record.get('_lineNo')} has no notePatches or note snapshots; "
            "this old log format is not enough to restore affected notes."
        )


def replay_records(dtpr: DTPR, records: Sequence[dict[str, Any]]) -> DTPR:
    records_by_op_id: dict[int, dict[str, Any]] = {}

    for record in records:
        if record.get("recordType") == "session":
            continue
        operation = str(record.get("operation") or "")
        op_id = record.get("opId")

        if operation == "undo":
            target_op_id = record.get("targetOpId")
            if target_op_id is None or int(target_op_id) not in records_by_op_id:
                raise ReplayError(f"undo at line {record.get('_lineNo')} references unknown targetOpId={target_op_id}.")
            _apply_record_payload(dtpr, records_by_op_id[int(target_op_id)], forward=False)
            if op_id is not None:
                records_by_op_id[int(op_id)] = record
            continue

        if operation == "redo":
            target_op_id = record.get("targetOpId")
            if target_op_id is None or int(target_op_id) not in records_by_op_id:
                raise ReplayError(f"redo at line {record.get('_lineNo')} references unknown targetOpId={target_op_id}.")
            _apply_record_payload(dtpr, records_by_op_id[int(target_op_id)], forward=True)
            if op_id is not None:
                records_by_op_id[int(op_id)] = record
            continue

        _apply_record_payload(dtpr, record, forward=True)
        if op_id is not None:
            records_by_op_id[int(op_id)] = record

    return dtpr


def replay_edit_log_to_dtpr(
    input_dtpr_path: str | Path,
    edit_log_path: str | Path,
    output_dtpr_path: str | Path,
) -> Path:
    dtpr = dtpr_file_to_dtpr_class(input_dtpr_path)
    records = load_jsonl_records(edit_log_path)
    replay_records(dtpr, records)
    return dtpr_class_to_dtpr_file(dtpr, output_dtpr_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay a Kikigaki edit log onto an initial DTPR file and export the restored DTPR."
    )
    parser.add_argument("input_dtpr", help="Initial DTPR text file path.")
    parser.add_argument("edit_log", help="Edit log JSONL file path.")
    parser.add_argument("output_dtpr", help="Output restored DTPR text file path.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output = replay_edit_log_to_dtpr(args.input_dtpr, args.edit_log, args.output_dtpr)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
