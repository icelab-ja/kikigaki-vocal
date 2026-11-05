"""edit_log.py

JSONL operation logger for the MIDI editor.

Key points
- 1 line (JSON) per operation (append-friendly).
- Stable identification via noteId (do NOT rely on index as the primary key).
- N->1 (merge) and 1->N (split) are represented by arrays: originalNotes / revisedNotes.
- Undo/Redo are explicit operations referencing targetOpId.
- No Qt/PySide dependency.

Notes: this module works with either NoteSnapshot or "note-like" objects that have
attributes: id (or note_id), onset, offset, pitch (and optionally velocity).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import json
import uuid


# -------------------------
# Data structures
# -------------------------


@dataclass(frozen=True)
class NoteSnapshot:
    """Serializable snapshot of a single note state."""

    noteId: Union[int, str]
    onsetSec: float
    offsetSec: float
    pitch: int
    index: Optional[int] = None
    velocity: Optional[int] = None
    realOnsetSec: Optional[float] = None
    realOffsetSec: Optional[float] = None
    idealOnsetSec: Optional[float] = None
    idealOffsetSec: Optional[float] = None
    absOnTick: Optional[int] = None
    absOffTick: Optional[int] = None
    onBeatId: Optional[int] = None
    offBeatId: Optional[int] = None
    relOnTick: Optional[int] = None
    relOffTick: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class OperationRecord:
    opId: int
    timestamp: str
    operation: str

    originalNotes: List[Dict[str, Any]]
    revisedNotes: List[Dict[str, Any]]

    targetOpId: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None
    candidateMeta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


# -------------------------
# Helpers
# -------------------------


def _now_iso(tz: timezone) -> str:
    return datetime.now(tz).isoformat(timespec="milliseconds")


def _get_note_id(note: Any) -> Union[int, str]:
    if isinstance(note, dict):
        if "noteId" in note:
            return note["noteId"]
        if "note_id" in note:
            return note["note_id"]
        if "id" in note:
            return note["id"]
    if hasattr(note, "noteId"):
        return getattr(note, "noteId")
    if hasattr(note, "note_id"):
        return getattr(note, "note_id")
    if hasattr(note, "id"):
        return getattr(note, "id")
    raise AttributeError("Note-like object must have id (or note_id/noteId).")


def snapshot_note(note: Any, *, index: Optional[int] = None) -> NoteSnapshot:
    """Convert a note-like object (or NoteSnapshot) into NoteSnapshot."""
    if isinstance(note, NoteSnapshot):
        if index is None or note.index == index:
            return note
        return NoteSnapshot(
            noteId=note.noteId,
            index=index,
            onsetSec=note.onsetSec,
            offsetSec=note.offsetSec,
            pitch=note.pitch,
            velocity=note.velocity,
            realOnsetSec=note.realOnsetSec,
            realOffsetSec=note.realOffsetSec,
            idealOnsetSec=note.idealOnsetSec,
            idealOffsetSec=note.idealOffsetSec,
            absOnTick=note.absOnTick,
            absOffTick=note.absOffTick,
            onBeatId=note.onBeatId,
            offBeatId=note.offBeatId,
            relOnTick=note.relOnTick,
            relOffTick=note.relOffTick,
        )

    if isinstance(note, dict):
        note_id = _get_note_id(note)
        onset = note.get("onsetSec")
        if onset is None:
            onset = note.get("realOnsetSec")
        if onset is None:
            onset = note.get("real_ontime")
        if onset is None:
            onset = note.get("idealOnsetSec")
        if onset is None:
            onset = note.get("ideal_ontime")
        if onset is None:
            onset = note.get("onset")

        offset = note.get("offsetSec")
        if offset is None:
            offset = note.get("realOffsetSec")
        if offset is None:
            offset = note.get("real_offtime")
        if offset is None:
            offset = note.get("idealOffsetSec")
        if offset is None:
            offset = note.get("ideal_offtime")
        if offset is None:
            offset = note.get("offset")

        pitch = note.get("pitch")
        if onset is None or offset is None or pitch is None:
            raise KeyError("Dict note snapshot must contain id/pitch and onset/offset-compatible keys.")

        velocity = note.get("velocity")
        velocity_i = int(velocity) if velocity is not None else None

        return NoteSnapshot(
            noteId=note_id,
            index=index,
            onsetSec=float(onset),
            offsetSec=float(offset),
            pitch=int(pitch),
            velocity=velocity_i,
            realOnsetSec=None if note.get("realOnsetSec") is None and note.get("real_ontime") is None else float(note.get("realOnsetSec", note.get("real_ontime"))),
            realOffsetSec=None if note.get("realOffsetSec") is None and note.get("real_offtime") is None else float(note.get("realOffsetSec", note.get("real_offtime"))),
            idealOnsetSec=None if note.get("idealOnsetSec") is None and note.get("ideal_ontime") is None else float(note.get("idealOnsetSec", note.get("ideal_ontime"))),
            idealOffsetSec=None if note.get("idealOffsetSec") is None and note.get("ideal_offtime") is None else float(note.get("idealOffsetSec", note.get("ideal_offtime"))),
            absOnTick=None if note.get("absOnTick") is None and note.get("abs_ontick") is None else int(note.get("absOnTick", note.get("abs_ontick"))),
            absOffTick=None if note.get("absOffTick") is None and note.get("abs_offtick") is None else int(note.get("absOffTick", note.get("abs_offtick"))),
            onBeatId=None if note.get("onBeatId") is None and note.get("on_beat_id") is None else int(note.get("onBeatId", note.get("on_beat_id"))),
            offBeatId=None if note.get("offBeatId") is None and note.get("off_beat_id") is None else int(note.get("offBeatId", note.get("off_beat_id"))),
            relOnTick=None if note.get("relOnTick") is None and note.get("on_metrical_pos") is None else int(note.get("relOnTick", note.get("on_metrical_pos"))),
            relOffTick=None if note.get("relOffTick") is None and note.get("off_metrical_pos") is None else int(note.get("relOffTick", note.get("off_metrical_pos"))),
        )

    note_id = _get_note_id(note)
    onset = float(getattr(note, "onset"))
    offset = float(getattr(note, "offset"))
    pitch = int(getattr(note, "pitch"))
    velocity = getattr(note, "velocity", None)
    velocity_i = int(velocity) if velocity is not None else None

    return NoteSnapshot(
        noteId=note_id,
        index=index,
        onsetSec=onset,
        offsetSec=offset,
        pitch=pitch,
        velocity=velocity_i,
        realOnsetSec=onset,
        realOffsetSec=offset,
    )


def snapshot_notes(
    notes: Sequence[Any],
    *,
    with_index: bool = True,
    sort_by_onset: bool = True,
) -> List[NoteSnapshot]:
    """Snapshot notes into NoteSnapshot list."""
    snaps = [snapshot_note(n) for n in notes]

    if sort_by_onset:
        snaps.sort(key=lambda s: (s.onsetSec, s.pitch, str(s.noteId)))

    if with_index:
        snaps = [
            NoteSnapshot(
                noteId=s.noteId,
                index=i,
                onsetSec=s.onsetSec,
                offsetSec=s.offsetSec,
                pitch=s.pitch,
                velocity=s.velocity,
                realOnsetSec=s.realOnsetSec,
                realOffsetSec=s.realOffsetSec,
                idealOnsetSec=s.idealOnsetSec,
                idealOffsetSec=s.idealOffsetSec,
                absOnTick=s.absOnTick,
                absOffTick=s.absOffTick,
                onBeatId=s.onBeatId,
                offBeatId=s.offBeatId,
                relOnTick=s.relOnTick,
                relOffTick=s.relOffTick,
            )
            for i, s in enumerate(snaps)
        ]

    return snaps


def _build_note_map(snaps: Sequence[NoteSnapshot]) -> Dict[Union[int, str], NoteSnapshot]:
    return {s.noteId: s for s in snaps}


def _note_snapshot_patch_dict(snapshot: NoteSnapshot) -> Dict[str, Any]:
    data = snapshot.to_dict()
    data.pop("noteId", None)
    data.pop("index", None)
    return data


def _note_snapshot_changed_patch(
    before: NoteSnapshot,
    after: NoteSnapshot,
) -> Optional[Dict[str, Any]]:
    before_data = _note_snapshot_patch_dict(before)
    after_data = _note_snapshot_patch_dict(after)
    keys = sorted((set(before_data.keys()) | set(after_data.keys())) - {"noteId"})
    before_patch: Dict[str, Any] = {}
    after_patch: Dict[str, Any] = {}
    for key in keys:
        before_value = before_data.get(key)
        after_value = after_data.get(key)
        if before_value != after_value:
            before_patch[key] = before_value
            after_patch[key] = after_value

    if not before_patch and not after_patch:
        return None

    return {
        "noteId": before.noteId,
        "before": before_patch,
        "after": after_patch,
    }


def diff_notes(
    before: Sequence[NoteSnapshot],
    after: Sequence[NoteSnapshot],
) -> Tuple[List[NoteSnapshot], List[NoteSnapshot], List[Tuple[NoteSnapshot, NoteSnapshot]]]:
    """Compute (inserted, deleted, modified) by noteId."""
    bmap = _build_note_map(before)
    amap = _build_note_map(after)

    inserted = [amap[k] for k in amap.keys() - bmap.keys()]
    deleted = [bmap[k] for k in bmap.keys() - amap.keys()]

    modified: List[Tuple[NoteSnapshot, NoteSnapshot]] = []
    for k in amap.keys() & bmap.keys():
        b = bmap[k]
        a = amap[k]
        if (
            b.onsetSec != a.onsetSec
            or b.offsetSec != a.offsetSec
            or b.pitch != a.pitch
            or b.velocity != a.velocity
            or b.realOnsetSec != a.realOnsetSec
            or b.realOffsetSec != a.realOffsetSec
            or b.idealOnsetSec != a.idealOnsetSec
            or b.idealOffsetSec != a.idealOffsetSec
            or b.absOnTick != a.absOnTick
            or b.absOffTick != a.absOffTick
            or b.onBeatId != a.onBeatId
            or b.offBeatId != a.offBeatId
            or b.relOnTick != a.relOnTick
            or b.relOffTick != a.relOffTick
        ):
            modified.append((b, a))

    inserted.sort(key=lambda s: (s.onsetSec, s.pitch, str(s.noteId)))
    deleted.sort(key=lambda s: (s.onsetSec, s.pitch, str(s.noteId)))
    modified.sort(key=lambda p: (p[1].onsetSec, p[1].pitch, str(p[1].noteId)))
    return inserted, deleted, modified


def infer_operation_from_diff(
    inserted: Sequence[NoteSnapshot],
    deleted: Sequence[NoteSnapshot],
    modified: Sequence[Tuple[NoteSnapshot, NoteSnapshot]],
) -> str:
    """Heuristic fallback when a caller doesn't provide an operation hint."""
    if not inserted and not deleted and not modified:
        return "noop"

    if inserted and not deleted and not modified:
        return "insert"

    if deleted and not inserted and not modified:
        return "delete"

    if len(deleted) >= 2 and len(inserted) == 1 and not modified:
        return "merge"

    if len(deleted) == 1 and len(inserted) >= 2 and not modified:
        return "split"

    # single-note modifications
    if len(modified) == 1 and not inserted and not deleted:
        b, a = modified[0]
        onset_changed = (b.onsetSec != a.onsetSec)
        offset_changed = (b.offsetSec != a.offsetSec)
        pitch_changed = (b.pitch != a.pitch)

        if pitch_changed:
            return "move"  # vertical or combined

        if onset_changed and offset_changed:
            dur_b = b.offsetSec - b.onsetSec
            dur_a = a.offsetSec - a.onsetSec
            if dur_b == dur_a:
                return "move"  # pure time shift
            return "edit"  # both ends changed (not a translation)

        if onset_changed and not offset_changed:
            return "resizeOnset"

        if offset_changed and not onset_changed:
            return "resizeOffset"

    return "edit"


# -------------------------
# Logger
# -------------------------


class EditOperationLogger:
    """In-memory operation logger producing JSONL."""

    def __init__(
        self,
        log_path: Optional[Union[str, Path]] = None,
        *,
        timezone_offset_minutes: int = 540,  # JST
        session_id: Optional[str] = None,
        initial_op_id: int = 1,
        auto_flush: bool = True,
    ) -> None:
        self._log_path: Optional[Path] = Path(log_path) if log_path is not None else None
        self._tz = timezone(timedelta(minutes=timezone_offset_minutes))
        self.sessionId = session_id or str(uuid.uuid4())
        self._next_op_id = int(initial_op_id)
        self._records: List[OperationRecord] = []
        self._auto_flush = bool(auto_flush)
        # self._undo_stack: List[int] = []
        # self._redo_stack: List[int] = []

    def reset_session(self, *, session_id: Optional[str] = None) -> None:
        self.sessionId = session_id or str(uuid.uuid4())
        self._next_op_id = 1
        self._records.clear()

    def next_op_id(self) -> int:
        op_id = self._next_op_id
        self._next_op_id += 1
        return op_id
    
    def _now_iso(self) -> str:
        return datetime.now(self._tz).isoformat(timespec="milliseconds")
    
    def _append_jsonl_dict(self, d: Dict[str, Any]) -> None:
        if self._log_path is None:
            return
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            f.flush()

    def append_session(self, *, meta: Optional[Dict[str, Any]] = None, timestamp: Optional[str] = None) -> None:
        d = {
            "recordType": "session",
            "timestamp": timestamp or self._now_iso(),
            "sessionId": self.sessionId,
            "meta": meta or {},
        }
        self._append_jsonl_dict(d)

    @property
    def records(self) -> List[OperationRecord]:
        return list(self._records)

    def record(
        self,
        *,
        operation: str,
        original_notes: Sequence[Any],
        revised_notes: Sequence[Any],
        meta: Optional[Dict[str, Any]] = None,
        candidate_meta: Optional[Dict[str, Any]] = None,
        target_op_id: Optional[int] = None,
        timestamp: Optional[str] = None,
    ) -> int:
        op_id = self.next_op_id()
        ts = timestamp or self._now_iso()

        orig_snaps = snapshot_notes(original_notes, with_index=True, sort_by_onset=True)
        rev_snaps = snapshot_notes(revised_notes, with_index=True, sort_by_onset=True)

        rec = OperationRecord(
            opId=op_id,
            timestamp=ts,
            operation=operation,
            originalNotes=[s.to_dict() for s in orig_snaps],
            revisedNotes=[s.to_dict() for s in rev_snaps],
            targetOpId=target_op_id,
            meta=meta,
            candidateMeta=candidate_meta,
        )
        self._records.append(rec)

        if self._auto_flush:
            self._append_jsonl_dict(rec.to_dict())

        return op_id

    def record_undo(
        self,
        *,
        target_op_id: int,
        meta: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> int:
        return self.record(
            operation="undo",
            original_notes=[],
            revised_notes=[],
            target_op_id=target_op_id,
            meta=meta,
            candidate_meta=None,
            timestamp=timestamp,
        )

    def record_redo(
        self,
        *,
        target_op_id: int,
        meta: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> int:
        return self.record(
            operation="redo",
            original_notes=[],
            revised_notes=[],
            target_op_id=target_op_id,
            meta=meta,
            candidate_meta=None,
            timestamp=timestamp,
        )

    def record_from_before_after(
        self,
        *,
        before_notes: Sequence[Any],
        after_notes: Sequence[Any],
        operation_hint: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        candidate_meta: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> int:
        """Convenience: diff before/after and record changed notes only."""
        b_snaps = snapshot_notes(before_notes, with_index=False, sort_by_onset=False)
        a_snaps = snapshot_notes(after_notes, with_index=False, sort_by_onset=False)

        inserted, deleted, modified = diff_notes(b_snaps, a_snaps)
        operation = operation_hint or infer_operation_from_diff(inserted, deleted, modified)

        orig_changed: List[NoteSnapshot] = []
        rev_changed: List[NoteSnapshot] = []

        orig_changed.extend(deleted)
        rev_changed.extend(inserted)
        for b, a in modified:
            orig_changed.append(b)
            rev_changed.append(a)

        # add indices within the subset for readability
        orig_changed = snapshot_notes(orig_changed, with_index=True, sort_by_onset=True)
        rev_changed = snapshot_notes(rev_changed, with_index=True, sort_by_onset=True)

        return self.record(
            operation=operation,
            original_notes=orig_changed,
            revised_notes=rev_changed,
            meta=meta,
            candidate_meta=candidate_meta,
            timestamp=timestamp,
        )

    def record_note_ids_from_before_after(
        self,
        *,
        before_notes: Sequence[Any],
        after_notes: Sequence[Any],
        operation_hint: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        candidate_meta: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> int:
        """Record changed note IDs in meta.affectedNoteIds."""
        b_snaps = snapshot_notes(before_notes, with_index=False, sort_by_onset=False)
        a_snaps = snapshot_notes(after_notes, with_index=False, sort_by_onset=False)

        inserted, deleted, modified = diff_notes(b_snaps, a_snaps)
        operation = operation_hint or infer_operation_from_diff(inserted, deleted, modified)

        affected_note_ids = []
        affected_note_ids.extend(s.noteId for s in deleted)
        affected_note_ids.extend(s.noteId for s in inserted)
        for b, a in modified:
            affected_note_ids.append(b.noteId)
            affected_note_ids.append(a.noteId)

        affected_note_ids = sorted(
            set(affected_note_ids),
            key=lambda note_id: (isinstance(note_id, str), note_id),
        )
        meta_with_ids = dict(meta or {})
        meta_with_ids["affectedNoteIds"] = affected_note_ids

        op_id = self.next_op_id()
        ts = timestamp or self._now_iso()
        rec = OperationRecord(
            opId=op_id,
            timestamp=ts,
            operation=operation,
            originalNotes=[],
            revisedNotes=[],
            meta=meta_with_ids,
            candidateMeta=candidate_meta,
        )
        self._records.append(rec)

        if self._auto_flush:
            self._append_jsonl_dict(rec.to_dict())

        return op_id

    def record_note_patches_from_before_after(
        self,
        *,
        before_notes: Sequence[Any],
        after_notes: Sequence[Any],
        operation_hint: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        candidate_meta: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> int:
        """Record changed notes as compact meta.notePatches entries."""
        b_snaps = snapshot_notes(before_notes, with_index=False, sort_by_onset=False)
        a_snaps = snapshot_notes(after_notes, with_index=False, sort_by_onset=False)

        inserted, deleted, modified = diff_notes(b_snaps, a_snaps)
        operation = operation_hint or infer_operation_from_diff(inserted, deleted, modified)

        note_patches: List[Dict[str, Any]] = []

        for note in deleted:
            note_patches.append(
                {
                    "noteId": note.noteId,
                    "before": _note_snapshot_patch_dict(note),
                    "after": None,
                }
            )
        for note in inserted:
            note_patches.append(
                {
                    "noteId": note.noteId,
                    "before": None,
                    "after": _note_snapshot_patch_dict(note),
                }
            )
        for before, after in modified:
            patch = _note_snapshot_changed_patch(before, after)
            if patch is not None:
                note_patches.append(patch)

        note_patches.sort(key=lambda patch: (isinstance(patch.get("noteId"), str), patch.get("noteId")))

        meta_with_patches = dict(meta or {})
        if note_patches:
            meta_with_patches["notePatches"] = note_patches

        op_id = self.next_op_id()
        ts = timestamp or self._now_iso()
        rec = OperationRecord(
            opId=op_id,
            timestamp=ts,
            operation=operation,
            originalNotes=[],
            revisedNotes=[],
            meta=meta_with_patches,
            candidateMeta=candidate_meta,
        )
        self._records.append(rec)

        if self._auto_flush:
            self._append_jsonl_dict(rec.to_dict())

        return op_id

    def to_jsonl(self) -> str:
        lines: List[str] = []
        for r in self._records:
            payload = r.to_dict()
            payload["sessionId"] = self.sessionId
            lines.append(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        return "\n".join(lines) + ("\n" if lines else "")

    def export_jsonl(self, path: Union[str, Path] = None) -> None:
        if path is None:
            if self._default_log_path is None:
                raise ValueError("export_jsonl(): path is required (no default log_path set).")
            path = self._default_log_path
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for rec in self._records:
                f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
        return path

    def export_json(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        arr: List[Dict[str, Any]] = []
        for r in self._records:
            d = r.to_dict()
            d["sessionId"] = self.sessionId
            arr.append(d)
        p.write_text(json.dumps(arr, ensure_ascii=False, indent=2), encoding="utf-8")

__all__ = [
    "NoteSnapshot",
    "OperationRecord",
    "EditOperationLogger",
    "snapshot_note",
    "snapshot_notes",
    "diff_notes",
    "infer_operation_from_diff",
]
