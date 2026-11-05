import itertools
import math
from bisect import bisect_right
from typing import List, Optional

from DataRepresentation import Beat, BDB, DTPR, MTPR, PTPR, TPQN
from beat_expansion import expand_bdb_to_cover_times
from beats_view import BeatEvent
from midi_file_io import read_midi_notes

class DTPRAdapter:
    def __init__(self, dtpr: DTPR):
        self.dtpr = dtpr

    @staticmethod
    def _normalize_domain(domain: str) -> str:
        d = str(domain or "ptpr").lower()
        if d not in ("ptpr", "mtpr"):
            raise ValueError(f"Unsupported domain: {domain}")
        return d

    @classmethod
    def from_beat_events(cls, beat_events: List[BeatEvent]) -> "DTPRAdapter":
        bdb = BDB()
        mtpr = MTPR()
        ptpr = PTPR()
        dtpr = DTPR(bdb, mtpr, ptpr)

        events = sorted(list(beat_events or []), key=lambda e: float(e.time_sec))
        if len(events) < 2:
            # DTPR側の変換で少なくとも次のbeat参照が必要になるため保険を入れる。
            if not events:
                events = [
                    BeatEvent(time_sec=0.0, is_downbeat=True),
                    BeatEvent(time_sec=0.5, is_downbeat=False),
                ]
            else:
                last_t = float(events[-1].time_sec)
                events.append(BeatEvent(time_sec=last_t + 0.5, is_downbeat=False))

        for ev in events:
            add_beat = getattr(dtpr, "add_beat", None)
            if callable(add_beat):
                add_beat(float(ev.time_sec), bool(ev.is_downbeat))
            else:
                dtpr.BDB.add_beat(float(ev.time_sec), bool(ev.is_downbeat))

        return cls(dtpr=dtpr)

    def replace_notes_from_midi_file(self, midi_path: str, *, use_language_model_quantization: bool = False) -> None:
        midi_notes = read_midi_notes(midi_path)
        self.dtpr.PTPR.notes = {}
        self.dtpr.PTPR.index = []
        self.dtpr.MTPR.notes = {}
        self.dtpr.MTPR.index = []
        self.dtpr._note_id_counter = itertools.count(0)

        for pitch, onset_sec, offset_sec in midi_notes:
            self.dtpr.add_note(
                pitch=int(pitch),
                real_ontime=float(onset_sec),
                real_offtime=float(offset_sec),
            )
        if self.dtpr.PTPR.index:
            ptpr_notes = list(self.dtpr.PTPR.notes.values())
            min_note_time = float(min(note.real_ontime for note in ptpr_notes))
            max_note_time = float(max(note.real_offtime for note in ptpr_notes))
            expand_bdb_to_cover_times(
                self.dtpr.BDB,
                note_start=min_note_time,
                note_end=max_note_time,
            )
        if use_language_model_quantization:
            self.dtpr.quantize_init_real_time_w_language_model()
        else:
            self.dtpr.quantize_init_real_time()

    def get_notes(self, domain: str = "ptpr"):
        d = self._normalize_domain(domain)
        if d == "ptpr":
            notes = list(self.dtpr.PTPR.notes.values())
            notes.sort(key=lambda n: (float(n.real_ontime), int(n.id)))
            return notes
        notes = list(self.dtpr.MTPR.notes.values())
        notes.sort(key=lambda n: (int(n.abs_ontick), int(n.id)))
        return notes

    def get_notes_sorted(self, domain: str = "ptpr"):
        return self.get_notes(domain=domain)

    def get_note_by_id(self, domain: str, id: int):
        d = self._normalize_domain(domain)
        if d == "ptpr":
            return self.dtpr.PTPR.notes.get(int(id))
        return self.dtpr.MTPR.notes.get(int(id))

    def note_delete_by_ids(self, note_ids: List[int], domain: str = "ptpr") -> None:
        d = self._normalize_domain(domain)
        note_map = self.dtpr.PTPR.notes if d == "ptpr" else self.dtpr.MTPR.notes
        for note_id in note_ids:
            nid = int(note_id)
            if nid in note_map:
                self.dtpr.note_delete(nid)

    def get_ptpr_notes(self):
        return self.get_notes(domain="ptpr")

    def get_mtpr_notes(self):
        return self.get_notes(domain="mtpr")

    def get_ptpr_note_by_id(self, id: int):
        return self.get_note_by_id("ptpr", int(id))

    def get_mtpr_note_by_id(self, id: int):
        return self.get_note_by_id("mtpr", int(id))

    def get_ptpr_notes_sorted(self):
        return self.get_notes_sorted(domain="ptpr")

    def get_mtpr_notes_sorted(self):
        return self.get_notes_sorted(domain="mtpr")

    def get_ptpr_duration_sec(self, time_mode: str = "real") -> float:
        notes = self.get_ptpr_notes()
        if not notes:
            return 0.0
        mode = str(time_mode).lower()
        if mode not in ("real", "ideal"):
            raise ValueError(f"Unsupported MIDI time mode: {time_mode}")
        offtime_attr = f"{mode}_offtime"
        return max(float(getattr(n, offtime_attr, 0.0) or 0.0) for n in notes)

    def get_mtpr_max_tick(self) -> int:
        notes = self.get_mtpr_notes()
        if not notes:
            return 0
        return max(int(getattr(n, "abs_offtick", 0) or 0) for n in notes)

    def get_mtpr_display_max_tick(self, default_beats: int = 8) -> int:
        default_tick = TPQN * max(1, int(default_beats))
        try:
            notes = self.get_mtpr_notes()
            bdb_max_tick = max(0, (len(self.dtpr.BDB.index) - 1) * TPQN)
            if notes:
                note_max_tick = max(
                    max(
                        int(getattr(note, "abs_offtick", 0)),
                        int(getattr(note, "abs_ontick", 0)),
                    )
                    for note in notes
                )
                return int(max(note_max_tick, bdb_max_tick))
            if len(self.dtpr.BDB.index) >= 2:
                return int(bdb_max_tick)
        except Exception:
            pass
        return int(default_tick)

    def get_median_beat_duration_sec(self) -> Optional[float]:
        beat_times = [float(time) for time, _beat_id in list(self.dtpr.BDB.index)]
        if len(beat_times) < 2:
            return None
        durations = sorted(
            duration
            for duration in (
                beat_times[index + 1] - beat_times[index]
                for index in range(len(beat_times) - 1)
            )
            if math.isfinite(duration) and duration > 1e-9
        )
        if not durations:
            return None
        middle = len(durations) // 2
        if len(durations) % 2:
            return float(durations[middle])
        return float((durations[middle - 1] + durations[middle]) / 2.0)

    def time_sec_to_abs_tick(self, time_sec: float) -> float:
        beat_times = [float(time) for time, _beat_id in list(self.dtpr.BDB.index)]
        if len(beat_times) < 2:
            return 0.0

        time_sec = float(time_sec)
        if time_sec < beat_times[0]:
            index = 0
            time0 = beat_times[0]
            time1 = beat_times[1]
        elif time_sec >= beat_times[-1]:
            index = max(0, len(beat_times) - 2)
            time0 = beat_times[index]
            time1 = beat_times[index + 1]
        else:
            index = max(0, bisect_right(beat_times, time_sec) - 1)
            index = min(index, len(beat_times) - 2)
            time0 = beat_times[index]
            time1 = beat_times[index + 1]

        fraction = (time_sec - time0) / max(1e-9, time1 - time0)
        return float(index * TPQN + fraction * TPQN)

    def abs_tick_to_time_sec(self, abs_tick: float) -> float:
        beat_times = [float(time) for time, _beat_id in list(self.dtpr.BDB.index)]
        if len(beat_times) < 2:
            return 0.0

        tick = max(0.0, float(abs_tick))
        beat_position = tick / float(TPQN)
        index = int(math.floor(beat_position))
        fraction = beat_position - index
        index = max(0, min(index, len(beat_times) - 2))
        time0 = beat_times[index]
        time1 = beat_times[index + 1]
        return float(time0 + fraction * (time1 - time0))

    def set_quantization_grid(self, divisions: int) -> None:
        divisions = max(1, int(divisions))
        self.dtpr.quantize_delta = max(1, int(TPQN // divisions))
        if hasattr(self.dtpr, "min_note_value"):
            self.dtpr.min_note_value = int(divisions * 4)

    @staticmethod
    def is_grid_divisions_change_allowed(current: int, new: int) -> bool:
        current = int(current)
        new = int(new)
        if new == current:
            return True
        if new < current:
            return False
        return new % current == 0

    def get_beats_sorted(self):
        out = []
        for beat_time, beat_id in list(self.dtpr.BDB.index):
            beat = self.dtpr.BDB.beats.get(int(beat_id))
            if beat is not None:
                out.append(beat)
        return out

    def get_beat_by_id(self, beat_id: int):
        return self.dtpr.BDB.beats.get(int(beat_id))

    def snapshot_bdb_state(self, *, include_checked: bool = True) -> list[dict]:
        out = []
        for beat in self.get_beats_sorted():
            item = {
                "id": int(beat.id),
                "beat_time": float(beat.beat_time),
                "is_downbeat": bool(beat.is_downbeat),
            }
            if include_checked:
                item["checked"] = bool(getattr(beat, "checked", False))
            out.append(item)
        return out

    def snapshot_ptpr_state(self, *, include_checked: bool = True) -> list[dict]:
        out = []
        for n in self.get_ptpr_notes():
            item = {
                "id": int(n.id),
                "pitch": int(n.pitch),
                "ideal_ontime": None if getattr(n, "ideal_ontime", None) is None else float(n.ideal_ontime),
                "real_ontime": None if getattr(n, "real_ontime", None) is None else float(n.real_ontime),
                "ideal_offtime": None if getattr(n, "ideal_offtime", None) is None else float(n.ideal_offtime),
                "real_offtime": None if getattr(n, "real_offtime", None) is None else float(n.real_offtime),
            }
            if include_checked:
                item["checked"] = bool(getattr(n, "checked", False))
            out.append(item)
        return out

    def snapshot_mtpr_state(self, *, include_checked: bool = True) -> list[dict]:
        out = []
        for n in self.get_mtpr_notes():
            item = {
                "id": int(n.id),
                "pitch": int(n.pitch),
                "abs_ontick": int(n.abs_ontick),
                "on_beat_id": int(n.on_beat_id),
                "on_metrical_pos": int(n.on_metrical_pos),
                "abs_offtick": int(n.abs_offtick),
                "off_beat_id": int(n.off_beat_id),
                "off_metrical_pos": int(n.off_metrical_pos),
            }
            if include_checked:
                item["checked"] = bool(getattr(n, "checked", False))
            out.append(item)
        return out

    def snapshot_state(self, domain: str = "ptpr") -> list[dict]:
        d = self._normalize_domain(domain)
        if d == "ptpr":
            return self.snapshot_ptpr_state()
        return self.snapshot_mtpr_state()

    def snapshot_dtpr_state(self, *, include_checked: bool = True) -> dict[str, list[dict]]:
        return {
            "bdb": self.snapshot_bdb_state(include_checked=include_checked),
            "ptpr": self.snapshot_ptpr_state(include_checked=include_checked),
            "mtpr": self.snapshot_mtpr_state(include_checked=include_checked),
        }

    def snapshot_edit_log_state(self) -> list[dict]:
        out = []
        ptpr_notes = list(self.get_ptpr_notes())
        ptpr_notes.sort(
            key=lambda n: (
                float(getattr(n, "real_ontime", 0.0) or 0.0),
                float(getattr(n, "ideal_ontime", 0.0) or 0.0),
                int(getattr(n, "id", 0)),
            )
        )
        for ptpr_note in ptpr_notes:
            note_id = int(ptpr_note.id)
            mtpr_note = self.get_mtpr_note_by_id(note_id)
            out.append(
                {
                    "id": note_id,
                    "pitch": int(ptpr_note.pitch),
                    "onsetSec": float(getattr(ptpr_note, "real_ontime", 0.0) or 0.0),
                    "offsetSec": float(getattr(ptpr_note, "real_offtime", 0.0) or 0.0),
                    "realOnsetSec": None if getattr(ptpr_note, "real_ontime", None) is None else float(ptpr_note.real_ontime),
                    "realOffsetSec": None if getattr(ptpr_note, "real_offtime", None) is None else float(ptpr_note.real_offtime),
                    "idealOnsetSec": None if getattr(ptpr_note, "ideal_ontime", None) is None else float(ptpr_note.ideal_ontime),
                    "idealOffsetSec": None if getattr(ptpr_note, "ideal_offtime", None) is None else float(ptpr_note.ideal_offtime),
                    "absOnTick": None if mtpr_note is None else int(mtpr_note.abs_ontick),
                    "absOffTick": None if mtpr_note is None else int(mtpr_note.abs_offtick),
                    "onBeatId": None if mtpr_note is None else int(mtpr_note.on_beat_id),
                    "offBeatId": None if mtpr_note is None else int(mtpr_note.off_beat_id),
                    "relOnTick": None if mtpr_note is None else int(mtpr_note.on_metrical_pos),
                    "relOffTick": None if mtpr_note is None else int(mtpr_note.off_metrical_pos),
                }
            )
        return out

    def restore_ptpr_state(
        self,
        state: list[dict],
        *,
        checked_by_id: Optional[dict[int, bool]] = None,
    ) -> None:
        self.dtpr.PTPR.notes = {}
        self.dtpr.PTPR.index = []
        max_id = -1
        for item in state:
            note_id = int(item["id"])
            max_id = max(max_id, note_id)
            self.dtpr.PTPR.add_ptpr_note(
                note_id,
                int(item["pitch"]),
                item.get("ideal_ontime"),
                item.get("real_ontime"),
                item.get("ideal_offtime"),
                item.get("real_offtime"),
            )
            checked = (
                bool(item.get("checked", False))
                if checked_by_id is None
                else bool(checked_by_id.get(note_id, False))
            )
            self.dtpr.PTPR.notes[note_id].set_checked(checked)
        self.dtpr.PTPR._id_counter = itertools.count(max_id + 1)

    def restore_mtpr_state(
        self,
        state: list[dict],
        *,
        checked_by_id: Optional[dict[int, bool]] = None,
    ) -> None:
        self.dtpr.MTPR.notes = {}
        self.dtpr.MTPR.index = []
        for item in state:
            self.dtpr.MTPR.add_mtpr_note(
                int(item["id"]),
                int(item["pitch"]),
                int(item["abs_ontick"]),
                int(item["on_beat_id"]),
                int(item["on_metrical_pos"]),
                int(item["abs_offtick"]),
                int(item["off_beat_id"]),
                int(item["off_metrical_pos"]),
            )
            note_id = int(item["id"])
            checked = (
                bool(item.get("checked", False))
                if checked_by_id is None
                else bool(checked_by_id.get(note_id, False))
            )
            self.dtpr.MTPR.notes[note_id].set_checked(checked)

    def restore_bdb_state(
        self,
        state: list[dict],
        *,
        checked_by_id: Optional[dict[int, bool]] = None,
    ) -> None:
        self.dtpr.BDB.beats = {}
        self.dtpr.BDB.index = []
        max_id = -1
        for item in state:
            beat_id = int(item["id"])
            max_id = max(max_id, beat_id)
            beat = Beat(
                beat_id,
                float(item["beat_time"]),
                bool(item["is_downbeat"]),
            )
            checked = (
                bool(item.get("checked", False))
                if checked_by_id is None
                else bool(checked_by_id.get(beat_id, False))
            )
            beat.set_checked(checked)
            self.dtpr.BDB.beats[beat_id] = beat
            self.dtpr.BDB.index.append((float(beat.beat_time), beat_id))
        self.dtpr.BDB.index.sort(key=lambda item: (float(item[0]), int(item[1])))
        self.dtpr.BDB._id_counter = itertools.count(max_id + 1)

    def restore_state(self, domain: str, state: list[dict]) -> None:
        d = self._normalize_domain(domain)
        if d == "ptpr":
            self.restore_ptpr_state(state)
            return
        self.restore_mtpr_state(state)

    def restore_dtpr_state(
        self,
        state: dict[str, list[dict]],
        *,
        preserve_checked: bool = False,
    ) -> None:
        bdb_checked = None
        ptpr_checked = None
        mtpr_checked = None
        if preserve_checked:
            bdb_checked = {
                int(beat.id): bool(getattr(beat, "checked", False))
                for beat in self.get_beats_sorted()
            }
            ptpr_checked = {
                int(note.id): bool(getattr(note, "checked", False))
                for note in self.get_ptpr_notes()
            }
            mtpr_checked = {
                int(note.id): bool(getattr(note, "checked", False))
                for note in self.get_mtpr_notes()
            }
        if "bdb" in (state or {}):
            self.restore_bdb_state(
                list((state or {}).get("bdb", [])),
                checked_by_id=bdb_checked,
            )
        self.restore_ptpr_state(
            list((state or {}).get("ptpr", [])),
            checked_by_id=ptpr_checked,
        )
        self.restore_mtpr_state(
            list((state or {}).get("mtpr", [])),
            checked_by_id=mtpr_checked,
        )

        max_id = -1
        for note_id in self.dtpr.PTPR.notes.keys():
            max_id = max(max_id, int(note_id))
        for note_id in self.dtpr.MTPR.notes.keys():
            max_id = max(max_id, int(note_id))
        self.dtpr._note_id_counter = itertools.count(max_id + 1)

    def _beat_edit_delta(self) -> int:
        delta = getattr(self.dtpr, "quantize_delta", None)
        if delta is None:
            delta = TPQN // 4
        return max(1, int(delta))

    def beat_insert(self, beat_time: float, is_downbeat: bool = False):
        before_ids = set(self.dtpr.BDB.beats.keys())
        self.dtpr.beat_insert(float(beat_time), bool(is_downbeat), self._beat_edit_delta())
        after_ids = set(self.dtpr.BDB.beats.keys())
        new_ids = list(after_ids - before_ids)
        if len(new_ids) != 1:
            return None
        return self.get_beat_by_id(int(new_ids[0]))

    def beat_delete(self, beat_id: int) -> None:
        bid = int(beat_id)
        if bid in self.dtpr.BDB.beats:
            self.dtpr.beat_delete(bid, self._beat_edit_delta())

    def beat_local_move(self, beat_id: int, beat_time: float) -> None:
        bid = int(beat_id)
        if bid in self.dtpr.BDB.beats:
            self.dtpr.beat_local_move(bid, float(beat_time), self._beat_edit_delta())

    def beat_set_downbeat(self, beat_id: int, is_downbeat: bool) -> None:
        bid = int(beat_id)
        if bid in self.dtpr.BDB.beats:
            self.dtpr.BDB.set_downbeat(bid, bool(is_downbeat))

    def get_checked(self, domain: str, id: int) -> bool:
        d = str(domain or "").lower()
        if d == "beat":
            item = self.get_beat_by_id(int(id))
        else:
            item = self.get_note_by_id(self._normalize_domain(d), int(id))
        return bool(getattr(item, "checked", False)) if item is not None else False

    def set_checked(self, domain: str, id: int, checked: bool) -> bool:
        d = str(domain or "").lower()
        checked_bool = bool(checked)
        if d == "beat":
            item = self.get_beat_by_id(int(id))
            if item is None:
                return False
            old = bool(getattr(item, "checked", False))
            self.dtpr.beat_set_checked(int(id), checked_bool)
            return old != checked_bool

        normalized = self._normalize_domain(d)
        item = self.get_note_by_id(normalized, int(id))
        if item is None:
            return False
        old = bool(getattr(item, "checked", False))
        if normalized == "ptpr":
            self.dtpr.ptpr_set_checked(int(id), checked_bool)
        else:
            self.dtpr.mtpr_set_checked(int(id), checked_bool)
        return old != checked_bool

    def note_delete(self, id: int, domain: str = "ptpr") -> None:
        d = self._normalize_domain(domain)
        nid = int(id)
        note_map = self.dtpr.PTPR.notes if d == "ptpr" else self.dtpr.MTPR.notes
        if nid in note_map:
            self.dtpr.note_delete(nid)

    def ptpr_note_insert(self, pitch: int, real_ontime: float, real_offtime: float):
        before_ids = set(self.dtpr.PTPR.notes.keys())
        self.dtpr.ptpr_note_insert(
            pitch=int(pitch),
            real_ontime=float(real_ontime),
            real_offtime=float(real_offtime),
        )
        after_ids = set(self.dtpr.PTPR.notes.keys())
        new_ids = list(after_ids - before_ids)
        if len(new_ids) != 1:
            return None
        return self.dtpr.PTPR.get_note(int(new_ids[0]))

    def mtpr_note_insert(self, pitch: int, abs_ontick: int, abs_offtick: int):
        before_ids = set(self.dtpr.MTPR.notes.keys())
        self.dtpr.mtpr_note_insert(
            pitch=int(pitch),
            abs_ontick=int(abs_ontick),
            abs_offtick=int(abs_offtick),
        )
        after_ids = set(self.dtpr.MTPR.notes.keys())
        new_ids = list(after_ids - before_ids)
        if len(new_ids) != 1:
            return None
        return self.dtpr.MTPR.get_note(int(new_ids[0]))

    def ptpr_note_move_by_id(
        self,
        note_id: int,
        pitch: int,
        real_ontime: float,
        real_offtime: float,
        *,
        update_ideal: bool = False,
    ) -> None:
        nid = int(note_id)
        if nid not in self.dtpr.PTPR.notes:
            return
        self.dtpr.ptpr_note_move(
            id=nid,
            pitch=int(pitch),
            real_ontime=float(real_ontime),
            real_offtime=float(real_offtime),
            update_ideal=bool(update_ideal),
        )

    def mtpr_note_move_by_id(
        self,
        note_id: int,
        pitch: int,
        abs_ontick: int,
        abs_offtick: int,
        *,
        keep_real: bool = False,
    ) -> None:
        nid = int(note_id)
        if nid not in self.dtpr.MTPR.notes:
            return
        self.dtpr.mtpr_note_move(
            id=nid,
            pitch=int(pitch),
            abs_ontick=int(abs_ontick),
            abs_offtick=int(abs_offtick),
            keep_real=bool(keep_real),
        )

    def ptpr_note_ontime_shift_by_id(
        self,
        note_id: int,
        real_ontime: float,
        *,
        update_ideal: bool = False,
    ) -> None:
        nid = int(note_id)
        if nid not in self.dtpr.PTPR.notes:
            return
        self.dtpr.ptpr_note_ontime_shift(
            id=nid,
            real_ontime=float(real_ontime),
            update_ideal=bool(update_ideal),
        )

    def mtpr_note_ontime_shift_by_id(
        self,
        note_id: int,
        abs_ontick: int,
        *,
        keep_real: bool = False,
    ) -> None:
        nid = int(note_id)
        if nid not in self.dtpr.MTPR.notes:
            return
        self.dtpr.mtpr_note_ontime_shift(
            id=nid,
            abs_ontick=int(abs_ontick),
            keep_real=bool(keep_real),
        )

    def ptpr_note_offtime_shift_by_id(
        self,
        note_id: int,
        real_offtime: float,
        *,
        update_ideal: bool = False,
    ) -> None:
        nid = int(note_id)
        if nid not in self.dtpr.PTPR.notes:
            return
        self.dtpr.ptpr_note_offtime_shift(
            id=nid,
            real_offtime=float(real_offtime),
            update_ideal=bool(update_ideal),
        )

    def mtpr_note_offtime_shift_by_id(
        self,
        note_id: int,
        abs_offtick: int,
        *,
        keep_real: bool = False,
    ) -> None:
        nid = int(note_id)
        if nid not in self.dtpr.MTPR.notes:
            return
        self.dtpr.mtpr_note_offtime_shift(
            id=nid,
            abs_offtick=int(abs_offtick),
            keep_real=bool(keep_real),
        )

    def note_join_by_ids(self, note_ids: List[int], domain: str = "ptpr") -> Optional[int]:
        d = self._normalize_domain(domain)
        note_map = self.dtpr.PTPR.notes if d == "ptpr" else self.dtpr.MTPR.notes
        valid_ids = []
        for nid in note_ids:
            i = int(nid)
            if i in note_map:
                valid_ids.append(i)
        if len(valid_ids) < 2:
            return None

        if d == "ptpr":
            key_fn = lambda i: (float(self.dtpr.PTPR.notes[i].real_ontime), int(i))
        else:
            key_fn = lambda i: (int(self.dtpr.MTPR.notes[i].abs_ontick), int(i))

        valid_ids = sorted(
            set(valid_ids),
            key=key_fn,
        )
        base_id = int(valid_ids[0])
        for nid in valid_ids[1:]:
            self.dtpr.note_join(base_id, int(nid))
        return base_id

    def ptpr_note_split_by_id(self, note_id: int, real_split_time: float) -> Optional[List[int]]:
        nid = int(note_id)
        if nid not in self.dtpr.PTPR.notes:
            return None

        before_ids = set(self.dtpr.PTPR.notes.keys())
        self.dtpr.ptpr_note_split(
            id=nid,
            real_split_time=float(real_split_time),
        )
        after_ids = set(self.dtpr.PTPR.notes.keys())
        if nid not in after_ids:
            return None

        new_ids = list(after_ids - before_ids)
        if len(new_ids) != 1:
            return [nid]

        split_ids = [nid, int(new_ids[0])]
        split_ids = sorted(
            split_ids,
            key=lambda i: (
                float(self.dtpr.PTPR.notes[int(i)].real_ontime),
                int(i),
            ),
        )
        return split_ids

    def mtpr_note_split_by_id(self, note_id: int, abs_split_tick: int) -> Optional[List[int]]:
        nid = int(note_id)
        if nid not in self.dtpr.MTPR.notes:
            return None

        before_ids = set(self.dtpr.MTPR.notes.keys())
        self.dtpr.mtpr_note_split(
            id=nid,
            abs_split_tick=int(abs_split_tick),
        )
        after_ids = set(self.dtpr.MTPR.notes.keys())
        if nid not in after_ids:
            return None

        new_ids = list(after_ids - before_ids)
        if len(new_ids) != 1:
            return [nid]

        split_ids = [nid, int(new_ids[0])]
        split_ids = sorted(
            split_ids,
            key=lambda i: (
                int(self.dtpr.MTPR.notes[int(i)].abs_ontick),
                int(i),
            ),
        )
        return split_ids

    # New method names (DataRepresentation_addmethods.md)
    def ptpr_note_move(self, id: int, pitch: int, real_ontime: float, real_offtime: float, *, update_ideal: bool = False) -> None:
        self.ptpr_note_move_by_id(int(id), pitch, real_ontime, real_offtime, update_ideal=update_ideal)

    def ptpr_note_ontime_shift(self, id: int, real_ontime: float, *, update_ideal: bool = False) -> None:
        self.ptpr_note_ontime_shift_by_id(int(id), real_ontime, update_ideal=update_ideal)

    def ptpr_note_offtime_shift(self, id: int, real_offtime: float, *, update_ideal: bool = False) -> None:
        self.ptpr_note_offtime_shift_by_id(int(id), real_offtime, update_ideal=update_ideal)

    def ptpr_note_split(self, id: int, real_split_time: float) -> Optional[List[int]]:
        return self.ptpr_note_split_by_id(int(id), real_split_time)

    def mtpr_note_move(self, id: int, pitch: int, abs_ontick: int, abs_offtick: int, *, keep_real: bool = False) -> None:
        self.mtpr_note_move_by_id(int(id), pitch, abs_ontick, abs_offtick, keep_real=keep_real)

    def mtpr_note_ontime_shift(self, id: int, abs_ontick: int, *, keep_real: bool = False) -> None:
        self.mtpr_note_ontime_shift_by_id(int(id), abs_ontick, keep_real=keep_real)

    def mtpr_note_offtime_shift(self, id: int, abs_offtick: int, *, keep_real: bool = False) -> None:
        self.mtpr_note_offtime_shift_by_id(int(id), abs_offtick, keep_real=keep_real)

    def mtpr_note_split(self, id: int, abs_split_tick: int) -> Optional[List[int]]:
        return self.mtpr_note_split_by_id(int(id), abs_split_tick)

    def note_join(self, note_ids: List[int], domain: str = "ptpr") -> Optional[int]:
        return self.note_join_by_ids(note_ids, domain=domain)
