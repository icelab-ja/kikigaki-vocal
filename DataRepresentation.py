from typing import Dict, List, Tuple
from bisect import insort, bisect_left, bisect_right
import itertools
import numpy as np
import os, sys, copy
import math

TPQN = 1440 # 4分音符

PTPR_TIME_GRID_SEC = 0.01

def snap_ptpr_time(time: float) -> float:
    grid = PTPR_TIME_GRID_SEC
    return math.floor(float(time) / grid + 0.5) * grid

class Beat:
    def __init__(self, id: int, beat_time: float, is_downbeat: bool):
        self.id = id
        self.beat_time = beat_time
        self.is_downbeat = is_downbeat
        self.checked = False

    def set_beat_time(self, beat_time: float):
        self.beat_time = beat_time

    def set_downbeat(self, is_downbeat: bool):
        self.is_downbeat = is_downbeat

    def set_checked(self, checked: bool):
        self.checked = bool(checked)

    def change_checked(self):
        self.checked = not self.checked

    def print(self):
        print('id: '+str(self.id)+', beat time: '+str(self.beat_time)+', downbeat: '+str(self.is_downbeat))

class MTPRNote:
    def __init__(self, id: int, pitch: int, abs_ontick: int, on_beat_id: int, on_metrical_pos: int, abs_offtick: int, off_beat_id: int, off_metrical_pos: int):
        self.id = id
        self.pitch = pitch
        self.abs_ontick = abs_ontick
        self.on_beat_id = on_beat_id
        self.on_metrical_pos = on_metrical_pos
        self.abs_offtick = abs_offtick
        self.off_beat_id = off_beat_id
        self.off_metrical_pos = off_metrical_pos
        self.checked = False

    def set_pitch(self, pitch: int):
        self.pitch = pitch

    def set_abs_ontick(self, abs_ontick: int):
        self.abs_ontick = abs_ontick

    def set_on_beat_id(self, on_beat_id: int):
        self.on_beat_id = on_beat_id

    def set_on_metrical_pos(self, on_metrical_pos: int):
        self.on_metrical_pos = on_metrical_pos

    def set_abs_offtick(self, abs_offtick: int):
        self.abs_offtick = abs_offtick

    def set_off_beat_id(self, off_beat_id: int):
        self.off_beat_id = off_beat_id

    def set_off_metrical_pos(self, off_metrical_pos: int):
        self.off_metrical_pos = off_metrical_pos

    def set_checked(self, checked: bool):
        self.checked = bool(checked)

    def change_checked(self):
        self.checked = not self.checked

    def print(self):
        print('id: '+str(self.id)+', pitch: '+str(self.pitch)+', abs ontick: '+str(self.abs_ontick)+', rel ontick: ('+str(self.on_beat_id)+', '+str(self.on_metrical_pos)+'), abs offtick: '+str(self.abs_offtick)+', rel offtick: ('+str(self.off_beat_id)+', '+str(self.off_metrical_pos)+')')

class PTPRNote:
    def __init__(self, id: int, pitch: int, ideal_ontime: float, real_ontime: float, ideal_offtime: float, real_offtime: float):
        self.id = id
        self.pitch = pitch
        self.ideal_ontime = ideal_ontime
        self.real_ontime = snap_ptpr_time(real_ontime)
        self.ideal_offtime = ideal_offtime
        self.real_offtime = snap_ptpr_time(real_offtime)
        self.checked = False

    def set_pitch(self, pitch: int):
        self.pitch = pitch

    def set_ideal_ontime(self, ideal_ontime: float):
        self.ideal_ontime = ideal_ontime

    def set_real_ontime(self, real_ontime: float):
        self.real_ontime = snap_ptpr_time(real_ontime)

    def set_ideal_offtime(self, ideal_offtime: float):
        self.ideal_offtime = ideal_offtime

    def set_real_offtime(self, real_offtime: float):
        self.real_offtime = snap_ptpr_time(real_offtime)

    def set_checked(self, checked: bool):
        self.checked = bool(checked)

    def change_checked(self):
        self.checked = not self.checked

    def print(self):
        print('id: '+str(self.id)+', pitch: '+str(self.pitch)+', ideal ontime: '+str(self.ideal_ontime)+', real ontime: '+str(self.real_ontime)+', ideal offtime: '+str(self.ideal_offtime)+', real offtime: '+str(self.real_offtime))

# BeatDownBeat
class BDB:
    def __init__(self):
        self.beats: Dict[int, Beat] = {}
        self.index: List[Tuple[float, int]] = [] #(beat_time, id)
        self._id_counter = itertools.count(0)

    def add_beat(self, beat_time: float, is_downbeat: bool):
        while True:
            new_id = next(self._id_counter)
            if new_id not in self.beats:
                break
        
        beat = Beat(new_id, beat_time, is_downbeat)
        self.beats[beat.id] = beat
        self.index.append((beat.beat_time, beat.id))

    def set_beat_time(self, id: int, beat_time: float):
        beat = self.get_beat(id)
        beat.set_beat_time(beat_time)

    def set_downbeat(self, id: int, is_downbeat: bool):
        beat = self.get_beat(id)
        beat.set_downbeat(is_downbeat)

    # ビート時刻更新用、indexの順番は変わらない
    def update_index(self, id: int, beat_time: float, idx: int = None):
        if idx is None:
            idx = self.get_beat_idx(id)
        self.index[idx] = (beat_time, id)

    def print_index(self):
        for i in range(len(self.index)):
            beat = self.get_beat_from_idx(i)
            beat.print()

    def get_beat(self, id: int):
        beat = self.beats[id]
        return beat

    def get_beat_time_from_id(self, id: int):
        beat = self.beats[id]
        return beat.beat_time
    
    def get_beat_idx(self, id: int):
        beat = self.beats[id]
        idx = bisect_left(self.index, beat.beat_time, key=lambda t: t[0])
        return idx
    
    def get_beat_from_idx(self, idx: int):
        beat_id = self.index[idx][1]
        beat = self.beats[beat_id]
        return beat
    
    def get_prev_beat(self, id: int):
        prev_beat_index = self.get_beat_idx(id) - 1
        if prev_beat_index < 0:
            return None
        prev_beat_id = self.index[prev_beat_index][1]
        return self.beats[prev_beat_id]
    
    def get_next_beat(self, id: int):
        next_beat_index = self.get_beat_idx(id) + 1
        if next_beat_index >= len(self.index):
            return None
        next_beat_id = self.index[next_beat_index][1]
        return self.beats[next_beat_id]
    
    def get_beat_id_from_time(self, time: float):
        idx = bisect_right(self.index, time, key=lambda t: t[0]) - 1
        beat_id = self.index[idx][1]
        return beat_id

    def insert(self, beat_time: float, is_downbeat: bool):
        first_beat_time = self.get_beat_from_idx(0).beat_time
        last_beat_time = self.get_beat_from_idx(len(self.index)-1).beat_time
        if beat_time <= first_beat_time or last_beat_time <= beat_time:
            return None, None
        while True:
            new_id = next(self._id_counter)
            if new_id not in self.beats:
                break

        beat = Beat(new_id, beat_time, is_downbeat)
        self.beats[beat.id] = beat
        insort(self.index, (beat.beat_time, beat.id))
        new_beat_idx = self.get_beat_idx(new_id)
        # beatとそのindexを返すようにしたほうが明確
        return beat, new_beat_idx

    def delete(self, id: int):
        beat = self.get_beat(id)
        prev_beat = self.get_prev_beat(beat.id)
        key = (beat.beat_time, beat.id)
        delete_idx = bisect_left(self.index, beat.beat_time, key=lambda t: t[0])
        if delete_idx == 0 or delete_idx == len(self.index) - 1: # 最初と最後のbeatは削除不可
            return None
        if delete_idx == len(self.index) or self.index[delete_idx] != key:
            raise RuntimeError("Index out of sync (delete)")
        beat = self.beats.pop(id)
        self.index.pop(delete_idx)
        # 影響のあるbeat idを取得して返す
        past_beat_idx = delete_idx
        return beat, past_beat_idx
    
    def local_move(self, id: int, beat_time: float):
        # local moveは、両隣のbeatを超えることはないと制約
        beat = self.get_beat(id)
        prev_beat = self.get_prev_beat(id)
        next_beat = self.get_next_beat(id)
        idx = bisect_left(self.index, beat.beat_time, key=lambda t: t[0])
        if idx == 0 or idx == len(self.index) - 1: # 最初と最後のbeatは移動不可
            return None
        assert (prev_beat is None or prev_beat.beat_time < beat_time) and (next_beat is None or beat_time < next_beat.beat_time)
        self.update_index(id, beat_time)
        self.set_beat_time(id, beat_time)
        update_beat = self.get_beat(id)
        return update_beat, idx

# MusicalTimePianoRoll
class MTPR:
    def __init__(self):
        self.notes: Dict[int, MTPRNote] = {}
        self.index: List[Tuple[int, int, int]] = [] # [abs_ontick, abs_offtick, id]

    def add_mtpr_note(self, id, pitch: int, abs_ontick: int, on_beat_id: int, on_metrical_pos: int, abs_offtick: int, off_beat_id: int, off_metrical_pos: int):
        note = MTPRNote(id, pitch, abs_ontick, on_beat_id, on_metrical_pos, abs_offtick, off_beat_id, off_metrical_pos)
        self.notes[note.id] = note
        insort(self.index, (abs_ontick, abs_offtick, id))

    def get_index(self):
        return self.index

    def print_index(self):
        for i in range(len(self.index)):
            note = self.get_note_from_idx(i)
            note.print()

    def get_note(self, id: int):
        note = self.notes[id]
        return note
    
    def get_note_from_idx(self, idx: int):
        id = self.index[idx][2]
        note = self.get_note(id)
        return note

    def set_tick(self, id: int, abs_ontick: int = None, on_beat_id: int = None, on_metrical_pos: int = None, abs_offtick: int = None, off_beat_id: int = None, off_metrical_pos: int = None):
        note = self.get_note(id)
        if abs_ontick is not None:
            note.set_abs_ontick(abs_ontick)
        if (on_beat_id is not None) and (on_metrical_pos is not None):
            note.set_on_beat_id(on_beat_id)
            note.set_on_metrical_pos(on_metrical_pos)
        if abs_offtick is not None:
            note.set_abs_offtick(abs_offtick)
        if (off_beat_id is not None) and (off_metrical_pos is not None):
            note.set_off_beat_id(off_beat_id)
            note.set_off_metrical_pos(off_metrical_pos)

    def set_pitch(self, id: int, pitch: int):
        note = self.get_note(id)
        note.set_pitch(pitch)

    def update_index(self, id: int, abs_ontick: int, abs_offtick: int, idx: int = None):
        if idx is None:
            idx = self.get_note_idx(id)
        self.index.pop(idx)
        insort(self.index, (abs_ontick, abs_offtick, id))

    def update_tick(self, id: int, abs_ontick: int, abs_offtick: int, idx: int = None):
        if idx is None:
            idx = self.get_note_idx(id)
        self.index[idx] = (abs_ontick, abs_offtick, id)

    def get_notes_from_beat_id(self, beat_id: int):
        target_notes = []
        for l in range(len(self.index)):
            note_id = self.index[l][2]
            note = self.notes[note_id]
            if beat_id == note.on_beat_id or beat_id == note.off_beat_id:
                target_notes.append(note)
        return target_notes

    def get_note_idx(self, id: int):
        for idx, item in enumerate(self.index):
            if int(item[2]) == int(id):
                return idx
        raise KeyError(f"MTPR index out of sync: id={id}")
    
    def get_next_note_idx_from_tick(self, abs_tick: int):
        idx = bisect_left(self.index, abs_tick, key=lambda t: t[0])
        if idx == len(self.index):
            return None
        return idx
    
    def recalc_abs_tick_beat_insert(self, id):
        if id is None:
            return
        note = self.get_note(id)
        self.update_tick(id, note.abs_ontick + TPQN, note.abs_offtick + TPQN)
        self.set_tick(id, note.abs_ontick + TPQN, None, None, note.abs_offtick + TPQN, None, None)

    def recalc_abs_tick_beat_delete(self, id: int):
        if id is None:
            return
        note = self.get_note(id)
        self.update_tick(id, note.abs_ontick - TPQN, note.abs_offtick - TPQN)
        self.set_tick(id, note.abs_ontick - TPQN, None, None, note.abs_offtick - TPQN, None, None)

    def insert(self, id: int, pitch: int, abs_ontick: int, on_beat_id: int, on_metrical_pos: int, abs_offtick: int, off_beat_id: int, off_metrical_pos: int):
        note = MTPRNote(id, pitch, abs_ontick, on_beat_id, on_metrical_pos, abs_offtick, off_beat_id, off_metrical_pos)
        self.notes[note.id] = note
        insort(self.index, (abs_ontick, abs_offtick, id))

    def delete(self, id: int):
        idx = self.get_note_idx(id)
        delete_note = self.notes.pop(id)
        self.index.pop(idx)

    def move(self, id: int, pitch: int, abs_ontick: int, on_beat_id: int, on_metrical_pos: int, abs_offtick: int, off_beat_id: int, off_metrical_pos: int):
        if (abs_ontick is not None) and (abs_offtick is not None):
            self.update_index(id, abs_ontick, abs_offtick)
        self.set_tick(id, abs_ontick, on_beat_id, on_metrical_pos, abs_offtick, off_beat_id, off_metrical_pos)
        self.set_pitch(id, pitch)
    
    def ontime_shift(self, id: int, abs_ontick: int, on_beat_id: int, on_metrical_pos: int):
        note = self.get_note(id)
        self.update_index(id, abs_ontick, note.abs_offtick)
        self.set_tick(id, abs_ontick, on_beat_id, on_metrical_pos, None, None, None)

    def offtime_shift(self, id: int, abs_offtick: int, off_beat_id: int, off_metrical_pos: int):
        note = self.get_note(id)
        self.update_index(id, note.abs_ontick, abs_offtick)
        self.set_tick(id, None, None, None, abs_offtick, off_beat_id, off_metrical_pos)

    def split(self, id: int, new_id: int, abs_split_tick: int, split_beat_id: int, split_metrical_pos: int):
        note = self.get_note(id)
        pitch_new_note = note.pitch
        abs_offtick_new_note = note.abs_offtick
        off_beat_id_new_note = note.off_beat_id
        off_metrical_pos_new_note = note.off_metrical_pos
        self.update_index(id, note.abs_ontick, abs_split_tick)
        self.set_tick(id, None, None, None, abs_split_tick, split_beat_id, split_metrical_pos)
        self.insert(new_id, pitch_new_note, abs_split_tick, split_beat_id, split_metrical_pos, abs_offtick_new_note, off_beat_id_new_note, off_metrical_pos_new_note)

    def join(self, id1: int, id2: int):
        note1 = self.get_note(id1)
        note2 = self.get_note(id2)
        pitch = note1.pitch # note1, note2のどっちにするか、引数にするかは要検討
        abs_ontick = note1.abs_ontick
        on_beat_id = note1.on_beat_id
        on_metrical_pos = note1.on_metrical_pos
        abs_offtick = note2.abs_offtick
        off_beat_id = note2.off_beat_id
        off_metrical_pos = note2.off_metrical_pos
        self.delete(id2)
        self.move(id1, pitch, abs_ontick, on_beat_id, on_metrical_pos, abs_offtick, off_beat_id, off_metrical_pos)

# PhysicalTimePianoRoll
class PTPR:
    def __init__(self):
        self.notes: Dict[int, PTPRNote] = {}
        self.index: List[Tuple[float, float, int]] = [] # (ideal_ontime, ideal_offtime, id)

    def add_ptpr_note(self, id: int, pitch: int, ideal_ontime: float = None, real_ontime: float = None, ideal_offtime: float = None, real_offtime: float = None):

        if real_ontime is None:
            real_ontime = ideal_ontime

        if ideal_ontime is None:
            ideal_ontime = real_ontime

        if real_offtime is None:
            real_offtime = ideal_offtime

        if ideal_offtime is None:
            ideal_offtime = real_offtime
        
        note = PTPRNote(id, pitch, ideal_ontime, real_ontime, ideal_offtime, real_offtime)
        self.notes[note.id] = note
        self.index.append((ideal_ontime, ideal_offtime, id))
        return note

    def get_index(self):
        return self.index

    def print_index(self):
        for i in range(len(self.index)):
            note = self.get_note_from_idx(i)
            note.print()
    
    def get_note(self, id: int):
        note = self.notes[id]
        return note
    
    def get_note_from_idx(self, idx: int):
        id = self.index[idx][2]
        note = self.get_note(id)
        return note

    def set_time(self, id: int, ideal_ontime: float = None, real_ontime: float = None, ideal_offtime: float = None, real_offtime: float = None):
        note = self.get_note(id)
        if ideal_ontime is not None:
            note.set_ideal_ontime(ideal_ontime)
        if real_ontime is not None:
            note.set_real_ontime(real_ontime)
        if ideal_offtime is not None:
            note.set_ideal_offtime(ideal_offtime)
        if real_offtime is not None:
            note.set_real_offtime(real_offtime)

    def set_pitch(self, id: int, pitch: int):
        note = self.get_note(id)
        note.set_pitch(pitch)

    def update_index(self, id: int, ideal_ontime: float, ideal_offtime: float, idx: int = None):
        if idx is None:
            idx = self.get_note_idx(id)
        self.index.pop(idx)
        insort(self.index, (ideal_ontime, ideal_offtime, id))

    def get_note_idx(self, id: int):
        for idx, item in enumerate(self.index):
            if int(item[2]) == int(id):
                return idx
        raise KeyError(f"PTPR index out of sync: id={id}")

    def insert(self, id: int, pitch: int, ideal_ontime: float, real_ontime: float, ideal_offtime: float, real_offtime: float):
        note = PTPRNote(id, pitch, ideal_ontime, real_ontime, ideal_offtime, real_offtime)
        self.notes[note.id] = note
        insort(self.index, (ideal_ontime, ideal_offtime, id))

    def delete(self, id: int):
        idx = self.get_note_idx(id)
        delete_note = self.notes.pop(id)
        self.index.pop(idx)

    def move(self, id: int, pitch: int, ideal_ontime: float, real_ontime: float, ideal_offtime: float, real_offtime: float):
        if (ideal_ontime is not None) and (ideal_offtime is not None):
            self.update_index(id, ideal_ontime, ideal_offtime)
        self.set_time(id, ideal_ontime, real_ontime, ideal_offtime, real_offtime)
        self.set_pitch(id, pitch)

    def ontime_shift(self, id: int, ideal_ontime: float, real_ontime: float):
        note = self.get_note(id)
        if ideal_ontime is not None:
            self.update_index(id, ideal_ontime, note.ideal_offtime)
        self.set_time(id, ideal_ontime, real_ontime, None, None)

    def offtime_shift(self, id: int, ideal_offtime: float, real_offtime: float):
        note = self.get_note(id)
        if ideal_offtime is not None:
            self.update_index(id, note.ideal_ontime, ideal_offtime)
        self.set_time(id, None, None, ideal_offtime, real_offtime)

    def split(self, id: int, new_id: int, ideal_split_time: float, real_split_time: float):
        note = self.get_note(id)
        pitch_new_note = note.pitch
        ideal_offtime_new_note = note.ideal_offtime
        real_offtime_new_note = note.real_offtime
        self.update_index(id, note.ideal_ontime, ideal_split_time)
        self.set_time(id, None, None, ideal_split_time, real_split_time)
        self.insert(new_id, pitch_new_note, ideal_split_time, real_split_time, ideal_offtime_new_note, real_offtime_new_note)

    def join(self, id1: int, id2: int):
        note1 = self.get_note(id1)
        note2 = self.get_note(id2)
        pitch = note1.pitch
        ideal_ontime = note1.ideal_ontime
        real_ontime = note1.real_ontime
        ideal_offtime = note2.ideal_offtime
        real_offtime = note2.real_offtime
        self.delete(id2)
        self.move(id1, pitch, ideal_ontime, real_ontime, ideal_offtime, real_offtime)

# DualTimePianoRoll
class DTPR:
    def __init__(self, BDB: BDB, MTPR: MTPR, PTPR: PTPR):
        self.BDB = BDB
        self.MTPR = MTPR
        self.PTPR = PTPR
        self.min_note_value = 16
        self.quantize_delta = self.set_min_note_value(self.min_note_value)
        self._note_id_counter = itertools.count(0)
        self.onset_uni_prob = None
        self.onset_metrical_markov = None
        self.offset_uni_prob = None
        self.offset_metrical_markov = None
        self.interp_onset = 0.9
        self.interp_offset = 0.8
        self.load_onset_uni_prob()
        self.load_onset_metrical_markov()
        self.load_offset_uni_prob()
        self.load_offset_metrical_markov()
        self.interpolation_onset_markov(interp_lambda=self.interp_onset)
        self.interpolation_offset_markov(interp_lambda=self.interp_offset)

    def set_min_note_value(self, min_note_value: int):
        self.min_note_value = min_note_value
        self.quantize_delta = TPQN // (min_note_value // 4)

    def beat_set_checked(self, id: int, checked: bool):
        self.BDB.beats[id].set_checked(checked)

    def mtpr_set_checked(self, id: int, checked: bool):
        self.MTPR.notes[id].set_checked(checked)

    def ptpr_set_checked(self, id: int, checked: bool):
        self.PTPR.notes[id].set_checked(checked)

    # 初期入力用
    def add_note(self, pitch: int, real_ontime: float, real_offtime: float):
        while True:
            new_id = next(self._note_id_counter)
            if new_id not in self.PTPR.notes:
                break

        # on_beat_id = self.BDB.get_beat_id_from_time(real_ontime)
        # off_beat_id = self.BDB.get_beat_id_from_time(real_offtime)

        # ideal_ontime, abs_ontick, on_beat_id, on_metrical_pos = self.real_time_to_ideal_time(real_ontime, on_beat_id)
        # ideal_offtime, abs_offtick, off_beat_id, off_metrical_pos = self.real_time_to_ideal_time(real_offtime, off_beat_id)

        self.PTPR.add_ptpr_note(new_id, pitch, None, real_ontime, None, real_offtime)
        
        # self.MTPR.add_mtpr_note(new_id, pitch, abs_ontick, on_beat_id, on_metrical_pos, abs_offtick, off_beat_id, off_metrical_pos)

    def print_notes(self):
        assert len(self.PTPR.index) == len(self.MTPR.index)
        num_notes = len(self.PTPR.index)
        for i in range(num_notes):
            ptpr_note = self.PTPR.notes[self.PTPR.index[i][2]]
            mtpr_note = self.MTPR.notes[self.MTPR.index[i][2]]
            ptpr_note.print()
            mtpr_note.print()

    def get_mtpr_note_by_id(self, id: int):
        note = self.MTPR.get_note(id)
        return note

    def get_ptpr_note_by_id(self, id: int):
        note = self.PTPR.get_note(id)
        return note

    def get_mtpr_notes_sorted(self):
        notes = []
        index = self.MTPR.get_index()
        for i in range(len(index)):
            note = self.MTPR.get_note_from_idx(i)
            notes.append(note)
        return notes

    def get_ptpr_notes_sorted(self):
        notes = []
        index = self.PTPR.get_index()
        for i in range(len(index)):
            note = self.PTPR.get_note_from_idx(i)
            notes.append(note)
        return notes

    def get_ptpr_duration_sec(self):
        index = self.PTPR.get_index()
        offtime_last_note = index[-1][1]
        return offtime_last_note

    def get_mtpr_max_tick(self):
        beats = self.BDB.beats
        max_tick = len(beats) * TPQN
        return max_tick

    def real_time_to_ideal_time(self, time: float, beat_id: int, delta: int = 360):
        # 時刻を指定したbeat区間内で量子化
        assert TPQN % delta == 0

        beat_time = self.BDB.get_beat_time_from_id(beat_id)
        beat_idx = self.BDB.get_beat_idx(beat_id)

        next_beat = self.BDB.get_next_beat(beat_id)
        next_beat_time = next_beat.beat_time
        next_beat_id = next_beat.id

        try:
            assert beat_time <= time and time < next_beat_time
        except AssertionError:
            print('beat time: '+str(beat_time)+', next beat time: '+str(next_beat_time)+', time: '+str(time))
            raise

        idx = np.arange(TPQN//delta + 1)
        ideal_time_grid = beat_time+(idx*delta*(next_beat_time-beat_time)/TPQN)
        diff_time = np.abs(time-ideal_time_grid)
        min_grid = np.argmin(diff_time)

        metrical_pos = int(delta * min_grid)
        abs_tick = int(TPQN * beat_idx + metrical_pos)
        ideal_time = float(beat_time + (metrical_pos * (next_beat_time - beat_time)/TPQN))
        if metrical_pos >= TPQN:
            return ideal_time, abs_tick, next_beat_id, 0
        return ideal_time, abs_tick, beat_id, metrical_pos

    def quantize_note(self, note_id: int, delta: int):
        mtpr_note = self.MTPR.get_note(note_id)
        ptpr_note = self.PTPR.get_note(note_id)
        note_idx = self.MTPR.get_note_idx(note_id)
        if note_idx - 1 < 0:
            pre_abs_ontick = -1
            pre_abs_offtick = -1
        else:
            pre_note = self.MTPR.get_note_from_idx(note_idx - 1)
            pre_abs_ontick = pre_note.abs_ontick
            pre_abs_offtick = pre_note.abs_offtick

        on_beat_id = self.BDB.get_beat_id_from_time(ptpr_note.ideal_ontime)
        off_beat_id = self.BDB.get_beat_id_from_time(ptpr_note.ideal_offtime)

        ideal_ontime, abs_ontick, on_beat_id, rel_ontick = self.real_time_to_ideal_time(ptpr_note.ideal_ontime, on_beat_id, delta)
        if pre_abs_ontick >= abs_ontick and pre_abs_offtick > abs_ontick:
            abs_ontick = pre_abs_offtick
            on_beat_id, rel_ontick = self.abs_tick_to_rel_tick(abs_ontick)
            ideal_ontime = self.abs_tick_to_time(abs_ontick)

        ideal_offtime, abs_offtick, off_beat_id, rel_offtick = self.real_time_to_ideal_time(ptpr_note.ideal_offtime, off_beat_id, delta)
        if abs_ontick >= abs_offtick:
            abs_offtick = abs_ontick + delta
            off_beat_id, rel_offtick = self.abs_tick_to_rel_tick(abs_offtick)
            ideal_offtime = self.abs_tick_to_time(abs_offtick)
        self.MTPR.update_index(note_id, abs_ontick, abs_offtick)
        self.PTPR.update_index(note_id, ideal_ontime, ideal_offtime)
        self.MTPR.set_tick(mtpr_note.id, abs_ontick, on_beat_id, rel_ontick, abs_offtick, off_beat_id, rel_offtick)
        self.PTPR.set_time(ptpr_note.id, ideal_ontime, None, ideal_offtime, None)

    def quantize_real_time(self, delta: int):
        pre_abs_ontick = -1
        pre_abs_offtick = -1
        for i in range(len(self.MTPR.index)):
            mtpr_note = self.MTPR.get_note_from_idx(i)
            note_id = mtpr_note.id
            ptpr_note = self.PTPR.get_note(note_id)
            ideal_ontime, abs_ontick, on_beat_id, rel_ontick = self.real_time_to_ideal_time(ptpr_note.real_ontime, mtpr_note.on_beat_id, delta)
            if pre_abs_ontick >= abs_ontick and pre_abs_offtick > abs_ontick:
                abs_ontick = pre_abs_offtick
                on_beat_id, rel_ontick = self.abs_tick_to_rel_tick(abs_ontick)
                ideal_ontime = self.abs_tick_to_time(abs_ontick)
            pre_abs_ontick = abs_ontick

            ideal_offtime, abs_offtick, off_beat_id, rel_offtick = self.real_time_to_ideal_time(ptpr_note.real_offtime, mtpr_note.off_beat_id, delta)
            if pre_abs_ontick >= abs_offtick:
                abs_offtick = abs_ontick + delta
                off_beat_id, rel_offtick = self.abs_tick_to_rel_tick(abs_offtick)
                ideal_offtime = self.abs_tick_to_time(abs_offtick)
            pre_abs_offtick = abs_offtick
            self.MTPR.set_tick(mtpr_note.id, abs_ontick, on_beat_id, rel_ontick, abs_offtick, off_beat_id, rel_offtick)
            self.PTPR.set_time(ptpr_note.id, ideal_ontime, None, ideal_offtime, None)

    def quantize_init_real_time(self, delta: int | None = None):
        if delta is None:
            delta = self.quantize_delta
        pre_abs_ontick = -1
        pre_abs_offtick = -1
        for i in range(len(self.PTPR.index)):
            ptpr_note = self.PTPR.get_note_from_idx(i)
            on_beat_id = self.BDB.get_beat_id_from_time(ptpr_note.real_ontime)
            off_beat_id = self.BDB.get_beat_id_from_time(ptpr_note.real_offtime)
            ideal_ontime, abs_ontick, on_beat_id, rel_ontick = self.real_time_to_ideal_time(ptpr_note.real_ontime, on_beat_id, delta)
            if pre_abs_ontick >= abs_ontick and pre_abs_offtick > abs_ontick:
                abs_ontick = pre_abs_offtick
                on_beat_id, rel_ontick = self.abs_tick_to_rel_tick(abs_ontick)
                ideal_ontime = self.abs_tick_to_time(abs_ontick)
            pre_abs_ontick = abs_ontick

            ideal_offtime, abs_offtick, off_beat_id, rel_offtick = self.real_time_to_ideal_time(ptpr_note.real_offtime, off_beat_id, delta)
            if abs_ontick >= abs_offtick:
                abs_offtick = abs_ontick + delta
                off_beat_id, rel_offtick = self.abs_tick_to_rel_tick(abs_offtick)
                ideal_offtime = self.abs_tick_to_time(abs_offtick)
            pre_abs_offtick = abs_offtick
            
            self.MTPR.add_mtpr_note(ptpr_note.id, ptpr_note.pitch, abs_ontick, on_beat_id, rel_ontick, abs_offtick, off_beat_id, rel_offtick)
            self.PTPR.set_time(ptpr_note.id, ideal_ontime, None, ideal_offtime, None)
        self.quantize_offset()


    def quantize_ideal_time(self, delta: int):
        pre_abs_ontick = -1
        pre_abs_offtick = -1
        for i in range(len(self.MTPR.index)):
            mtpr_note = self.MTPR.get_note_from_idx(i)
            note_id = mtpr_note.id
            ptpr_note = self.PTPR.get_note(note_id)
            ideal_ontime, abs_ontick, on_beat_id, rel_ontick = self.real_time_to_ideal_time(ptpr_note.ideal_ontime, mtpr_note.on_beat_id, delta)
            if pre_abs_ontick >= abs_ontick and pre_abs_offtick > abs_ontick:
                abs_ontick = pre_abs_offtick
                on_beat_id, rel_ontick = self.abs_tick_to_rel_tick(abs_ontick)
                ideal_ontime = self.abs_tick_to_time(abs_ontick)
            pre_abs_ontick = abs_ontick

            ideal_offtime, abs_offtick, off_beat_id, rel_offtick = self.real_time_to_ideal_time(ptpr_note.ideal_offtime, mtpr_note.off_beat_id, delta)
            if abs_ontick >= abs_offtick:
                abs_offtick = abs_ontick + delta
                off_beat_id, rel_offtick = self.abs_tick_to_rel_tick(abs_offtick)
                ideal_offtime = self.abs_tick_to_time(abs_offtick)
            pre_abs_offtick = abs_offtick
            self.MTPR.set_tick(mtpr_note.id, abs_ontick, on_beat_id, rel_ontick, abs_offtick, off_beat_id, rel_offtick)
            self.PTPR.set_time(ptpr_note.id, ideal_ontime, None, ideal_offtime, None)

    def quantize_offset(self):
        for i in range(len(self.MTPR.index)-1):
            mtpr_note = self.MTPR.get_note_from_idx(i)
            note_id = mtpr_note.id
            ptpr_note = self.PTPR.get_note(note_id)
            next_mtpr_note = self.MTPR.get_note_from_idx(i+1)
            next_note_id = next_mtpr_note.id
            next_ptpr_note = self.PTPR.get_note(next_note_id)
            # note_next_beat = self.BDB.get_next_beat(mtpr_note.off_beat_id)
            if (next_mtpr_note.abs_ontick - mtpr_note.abs_offtick) < TPQN/2:
                abs_offtick = next_mtpr_note.abs_ontick
                ideal_offtime = next_ptpr_note.ideal_ontime
                off_beat_id, rel_offtick = self.abs_tick_to_rel_tick(abs_offtick)
            elif mtpr_note.abs_offtick % TPQN == 0:
                abs_offtick = mtpr_note.abs_offtick
                off_beat_id = mtpr_note.off_beat_id
                rel_offtick = mtpr_note.off_metrical_pos
                ideal_offtime = ptpr_note.ideal_offtime
            else:
                abs_offtick = min(next_mtpr_note.abs_ontick, mtpr_note.abs_offtick - (mtpr_note.abs_offtick%TPQN) + TPQN)
                off_beat_id, rel_offtick = self.abs_tick_to_rel_tick(abs_offtick)
                beat = self.BDB.get_beat(off_beat_id)
                ideal_offtime = min(next_ptpr_note.ideal_ontime, beat.beat_time)
            self.MTPR.set_tick(note_id, None, None, None, abs_offtick, off_beat_id, rel_offtick)
            self.PTPR.set_time(note_id, None, None, ideal_offtime, None)

    def ln_P_uni(self, on_cur):
        return self.onset_uni_prob[on_cur]
    
    def ln_P_tr(self, on_ore, on_cur):
        return self.onset_metrical_markov[on_ore, on_cur]
        # return self.onset_uni_prob[on_cur]
    
    def ln_P_off_uni(self, off_cur):
        return self.offset_uni_prob[off_cur]

    def ln_P_off(self, on_cur, on_nex, off_cur):
        return self.offset_metrical_markov[on_cur, on_nex, off_cur]
        # return self.offset_uni_prob[off_cur]

    def quantize_init_real_time_w_language_model(self, delta: int | None = None):
        # ln_P_uni(on_cur)
        # ln_P_tr(on_ore, on_cur)
        # ln_P_off(on_cur, on_nex, off_cur)
        sig_t = 0.05 # 0.025 # = 50 ms, onset time deviation
        sig_off = 0.1 # = 100 ms, offset time deviation
        tpqn = 24

        if delta is None:
            delta = self.quantize_delta

        # ontick estimation

        #state (candidate set) construction
        num_tatum_per_beat = TPQN // delta
        ontick_cand_set_list = []
        ideal_ontime_cand_set_list = []
        real_ontime_list = []
        abs_ontick_list = []
        abs_offtick_list = []

        pre_abs_ontick = -1

        for i in range(len(self.PTPR.index)):
            ptpr_note = self.PTPR.get_note_from_idx(i)
            on_beat_id = self.BDB.get_beat_id_from_time(ptpr_note.real_ontime)
            ideal_ontime, abs_ontick, on_beat_id, rel_ontick = self.real_time_to_ideal_time(ptpr_note.real_ontime, on_beat_id, delta)
            while pre_abs_ontick >= abs_ontick:
                abs_ontick += delta
            real_ontime_list.append(ptpr_note.real_ontime)
            abs_ontick_list.append(abs_ontick)
            ontick_cand_set_this = [max(abs_ontick - (TPQN // 2) + j * delta, 0) for j in range(num_tatum_per_beat+1)]
            ontick_cand_set_list.append(ontick_cand_set_this)
            ideal_ontime_cand_set_this = [self.abs_tick_to_time(ontick_cand_set_this[j]) for j in range(num_tatum_per_beat+1)]
            ideal_ontime_cand_set_list.append(ideal_ontime_cand_set_this)
            pre_abs_ontick = abs_ontick

        #forward updates
        like = [-1E100 for _ in range(num_tatum_per_beat+1)]
        amax_hist = []
        for i in range(len(self.PTPR.index)):
            amax_hist.append([-1 for _ in range(num_tatum_per_beat+1)])
            if i == 0:
                for j in range(num_tatum_per_beat+1):
                    tau = ontick_cand_set_list[i][j]
                    like[j] = self.ln_P_uni((tau%TPQN)//(TPQN//tpqn)) - 0.5 * np.pow((real_ontime_list[i]-ideal_ontime_cand_set_list[i][j])/sig_t, 2.) #******************
                continue
            like_pre = copy.deepcopy(like)
            like = [-1E100 for _ in range(num_tatum_per_beat+1)]
            for j in range(num_tatum_per_beat+1):
                tau = ontick_cand_set_list[i][j]
                for jp in range(num_tatum_per_beat+1):
                    tau_pre = ontick_cand_set_list[i-1][jp]
                    if tau <= tau_pre:
                        continue
                    log_P = like_pre[jp] + self.ln_P_tr((tau_pre%TPQN)//(TPQN//tpqn), (tau%TPQN)//(TPQN//tpqn)) - 0.5 * np.pow((real_ontime_list[i]-ideal_ontime_cand_set_list[i][j])/sig_t, 2.) #******************
                    if log_P > like[j]:
                        like[j] = log_P
                        amax_hist[i][j] = jp

        #backtracking
        opt_state = None
        opt_state_pre = None
        for i in range(len(self.PTPR.index)-1, -1, -1):
            if i == len(self.PTPR.index)-1:
                opt_state = np.argmax(like)
                abs_ontick_list[i] = ontick_cand_set_list[i][opt_state]
            else:
                opt_state = amax_hist[i+1][opt_state_pre]
                abs_ontick_list[i] = ontick_cand_set_list[i][opt_state]                
            opt_state_pre = opt_state

        # offtick estimation

        for i in range(len(self.PTPR.index)):
            ptpr_note = self.PTPR.get_note_from_idx(i)
            off_beat_id = self.BDB.get_beat_id_from_time(ptpr_note.real_offtime)
            ideal_offtime, abs_offtick, off_beat_id, rel_offtick = self.real_time_to_ideal_time(ptpr_note.real_offtime, off_beat_id, delta)
            offtick_cand_set_this = None
            next_ontick = None

            if i < len(self.PTPR.index) - 1:
                next_ontick = abs_ontick_list[i+1]
            else:
                next_ontick = abs_offtick - abs_offtick%TPQN + TPQN
                if abs_ontick_list[i]+delta >= next_ontick:
                    next_ontick = abs_ontick_list[i]+delta
            offtick_cand_set_this = [tau_off for tau_off in range(abs_ontick_list[i]+delta, next_ontick+1, delta)]

            like = [-1E100 for j in range(len(offtick_cand_set_this))]

            for j in range(len(offtick_cand_set_this)):
                ideal_offtime = self.abs_tick_to_time(offtick_cand_set_this[j])
                # like[j] = self.ln_P_off((abs_ontick_list[i]%TPQN)//(TPQN//tpqn), (next_ontick%TPQN)//(TPQN//tpqn), (offtick_cand_set_this[j]%TPQN)//(TPQN//tpqn)) - 0.5 * np.pow((ptpr_note.real_offtime-ideal_offtime)/sig_off, 2.) #******************
                if offtick_cand_set_this[j] == next_ontick:
                    like[j] = -0.102629 - 0.5 * np.pow((ptpr_note.real_offtime-ideal_offtime)/sig_off, 2.)
                else:
                    like[j] = -2.32751 + self.ln_P_off((abs_ontick_list[i]%TPQN)//(TPQN//tpqn), (next_ontick%TPQN)//(TPQN//tpqn), (offtick_cand_set_this[j]%TPQN)//(TPQN//tpqn)) - 0.5 * np.pow((ptpr_note.real_offtime-ideal_offtime)/sig_off, 2.) #******************
            opt_state = np.argmax(like)
            abs_offtick_list.append(offtick_cand_set_this[opt_state])

            abs_ontick = abs_ontick_list[i]
            abs_offtick = abs_offtick_list[i]
            on_beat_id, rel_ontick = self.abs_tick_to_rel_tick(abs_ontick)
            off_beat_id, rel_offtick = self.abs_tick_to_rel_tick(abs_offtick)
            ideal_ontime = self.abs_tick_to_time(abs_ontick)
            ideal_offtime = self.abs_tick_to_time(abs_offtick)
            self.MTPR.add_mtpr_note(ptpr_note.id, ptpr_note.pitch, abs_ontick, on_beat_id, rel_ontick, abs_offtick, off_beat_id, rel_offtick)
            self.PTPR.set_time(ptpr_note.id, ideal_ontime, None, ideal_offtime, None)


    def abs_tick_to_time(self, abs_tick: int):
        beat_idx = abs_tick // TPQN
        metrical_pos = abs_tick % TPQN
        beat = self.BDB.get_beat_from_idx(beat_idx)
        if beat_idx < len(self.BDB.index) - 1:
            next_beat = self.BDB.get_next_beat(beat.id)
            time = beat.beat_time + (metrical_pos * (next_beat.beat_time - beat.beat_time) / TPQN)
        else:
            prev_beat = self.BDB.get_beat_from_idx(beat_idx-1)
            time = beat.beat_time + (metrical_pos * (beat.beat_time - prev_beat.beat_time) / TPQN)
        return time

    def rel_tick_to_time(self, beat_id: int, metrical_pos: int):
        beat = self.BDB.get_beat(beat_id)
        next_beat = self.BDB.get_next_beat(beat_id)
        time = beat.beat_time + (metrical_pos * (next_beat.beat_time - beat.beat_time) / TPQN)
        return time
    
    def abs_tick_to_rel_tick(self, abs_tick: int):
        beat_idx = abs_tick // TPQN
        metrical_pos = abs_tick % TPQN
        beat = self.BDB.get_beat_from_idx(beat_idx)
        return beat.id, metrical_pos

    def rel_tick_to_abs_tick(self, beat_id: int, metrical_pos: int):
        beat_idx = self.BDB.get_beat_idx(beat_id)
        abs_tick = beat_idx * TPQN + metrical_pos
        return abs_tick

    def get_beat_id_from_time(self, time: float):
        beat_id = self.BDB.get_beat_id_from_time(time)
        return beat_id

    def beat_insert(self, beat_time: float, is_downbeat: bool, delta: int):
        new_beat, new_idx = self.BDB.insert(beat_time, is_downbeat)
        if new_beat is None:
            return # errorも検討
        # 再量子化以降のビート時刻を取得
        prev_beat = self.BDB.get_beat_from_idx(new_idx - 1)
        prev_beat_time = prev_beat.beat_time
        next_beat = self.BDB.get_beat_from_idx(new_idx + 1)
        next_beat_time = next_beat.beat_time
        # 時刻(ideal time)ベースで修正するnote_idを分類する
        # note idをすべて取得してからそのidの音符に対して処理を行う
        # tick増加ではindex取得処理は使わない
        # prev beat time <= ideal time < next beat time: 再量子化
        # next beat time <= ideal time: tick増加
        requantize_ids = []
        increase_tick_ids = []
        for i in range(len(self.MTPR.index)):
            mtpr_note = self.MTPR.get_note_from_idx(i)
            ptpr_note = self.PTPR.get_note(mtpr_note.id)
            if (prev_beat_time <= ptpr_note.ideal_ontime and ptpr_note.ideal_ontime < next_beat_time) or (prev_beat_time <= ptpr_note.ideal_offtime and ptpr_note.ideal_offtime < next_beat_time):
                requantize_ids.append(mtpr_note.id)
            elif next_beat_time <= ptpr_note.ideal_ontime:
                increase_tick_ids.append(mtpr_note.id)
        for i in range(len(increase_tick_ids)):
            # recalcメソッドは要修正: id->recalc
            self.MTPR.recalc_abs_tick_beat_insert(increase_tick_ids[i])
        for i in range(len(requantize_ids)):
            self.quantize_note(requantize_ids[i], delta)

    def beat_delete(self, id: int, delta: int):
        beat, past_beat_idx = self.BDB.delete(id)
        if beat is None:
            return # errorも検討
        # 再量子化以降のビート時刻を取得
        prev_beat = self.BDB.get_beat_from_idx(past_beat_idx - 1)
        prev_beat_time = prev_beat.beat_time
        next_beat = self.BDB.get_beat_from_idx(past_beat_idx)
        next_beat_time = next_beat.beat_time
        # 時刻(ideal time)ベースで修正するnote_idを分類する
        # prev beat time <= ideal time < next beat time: 再量子化
        # next beat time <= ideal time: tick減少
        requantize_ids = []
        decrease_tick_ids = []
        for i in range(len(self.MTPR.index)):
            mtpr_note = self.MTPR.get_note_from_idx(i)
            ptpr_note = self.PTPR.get_note(mtpr_note.id)
            if (prev_beat_time <= ptpr_note.ideal_ontime and ptpr_note.ideal_ontime < next_beat_time) or (prev_beat_time <= ptpr_note.ideal_offtime and ptpr_note.ideal_offtime < next_beat_time):
                requantize_ids.append(mtpr_note.id)
            elif next_beat_time <= ptpr_note.ideal_ontime:
                decrease_tick_ids.append(mtpr_note.id)
        for i in range(len(decrease_tick_ids)):
            # recalcメソッドは要修正: id->recalc
            self.MTPR.recalc_abs_tick_beat_delete(decrease_tick_ids[i])
        for i in range(len(requantize_ids)):
            self.quantize_note(requantize_ids[i], delta)

    def beat_local_move(self, id: int, beat_time: float, delta: int):
        beat, idx = self.BDB.local_move(id, beat_time)
        if beat is None:
            return # errorも検討
        # 再量子化以降のビート時刻を取得
        prev_beat = self.BDB.get_beat_from_idx(idx - 1)
        prev_beat_time = prev_beat.beat_time
        next_beat = self.BDB.get_beat_from_idx(idx + 1)
        next_beat_time = next_beat.beat_time
        requantize_ids = []
        for i in range(len(self.MTPR.index)):
            mtpr_note = self.MTPR.get_note_from_idx(i)
            ptpr_note = self.PTPR.get_note(mtpr_note.id)
            if (prev_beat_time <= ptpr_note.ideal_ontime and ptpr_note.ideal_ontime < next_beat_time) or (prev_beat_time <= ptpr_note.ideal_offtime and ptpr_note.ideal_offtime < next_beat_time):
                requantize_ids.append(mtpr_note.id)
        for i in range(len(requantize_ids)):
            self.quantize_note(requantize_ids[i], delta)

    # def beat_global_shift(self,):
        
    # def beat_up_sampling(self,):
        
    # def beat_down_sampling(self,):
        
    # def beat_harf_shift(self,): # harf以外に対応するために修正する可能性あり

    #####

    def mtpr_note_insert(self, pitch: int, abs_ontick: int, abs_offtick: int):
        while True:
            new_id = next(self._note_id_counter)
            if new_id not in self.PTPR.notes:
                break
        on_beat_id, on_metrical_pos = self.abs_tick_to_rel_tick(abs_ontick)
        off_beat_id, off_metrical_pos = self.abs_tick_to_rel_tick(abs_offtick)
        ideal_ontime = self.rel_tick_to_time(on_beat_id, on_metrical_pos)
        ideal_offtime = self.rel_tick_to_time(off_beat_id, off_metrical_pos)
        self.MTPR.insert(new_id, pitch, abs_ontick, on_beat_id, on_metrical_pos, abs_offtick, off_beat_id, off_metrical_pos)
        self.PTPR.insert(new_id, pitch, ideal_ontime, ideal_ontime, ideal_offtime, ideal_offtime)

    # mtpr, ptpr共通
    def note_delete(self, id: int):
        self.MTPR.delete(id)
        self.PTPR.delete(id)

    def mtpr_note_move(self, id: int, pitch: int, abs_ontick: int, abs_offtick: int, keep_real: bool = False):
        on_beat_id, on_metrical_pos = self.abs_tick_to_rel_tick(abs_ontick)
        off_beat_id, off_metrical_pos = self.abs_tick_to_rel_tick(abs_offtick)
        ideal_ontime = self.rel_tick_to_time(on_beat_id, on_metrical_pos)
        ideal_offtime = self.rel_tick_to_time(off_beat_id, off_metrical_pos)
        self.MTPR.move(id, pitch, abs_ontick, on_beat_id, on_metrical_pos, abs_offtick, off_beat_id, off_metrical_pos)
        # デフォルトはptprのrealを修正, オプションでreal保持
        if keep_real:
            self.PTPR.move(id, pitch, ideal_ontime, None, ideal_offtime, None)
        else:
            self.PTPR.move(id, pitch, ideal_ontime, ideal_ontime, ideal_offtime, ideal_offtime)

    def mtpr_note_ontime_shift(self, id: int, abs_ontick: int, keep_real: bool = False):
        on_beat_id, on_metrical_pos = self.abs_tick_to_rel_tick(abs_ontick)
        ideal_ontime = self.rel_tick_to_time(on_beat_id, on_metrical_pos)
        self.MTPR.ontime_shift(id, abs_ontick, on_beat_id, on_metrical_pos)
        if keep_real:
            self.PTPR.ontime_shift(id, ideal_ontime, None)
        else:
            self.PTPR.ontime_shift(id, ideal_ontime, ideal_ontime)

    def mtpr_note_offtime_shift(self, id: int, abs_offtick: int, keep_real: bool = False):
        off_beat_id, off_metrical_pos = self.abs_tick_to_rel_tick(abs_offtick)
        ideal_offtime = self.rel_tick_to_time(off_beat_id, off_metrical_pos)
        self.MTPR.offtime_shift(id, abs_offtick, off_beat_id, off_metrical_pos)
        if keep_real:
            self.PTPR.offtime_shift(id, ideal_offtime, None)
        else:
            self.PTPR.offtime_shift(id, ideal_offtime, ideal_offtime)

    def mtpr_note_split(self, id: int, abs_split_tick: int):
        while True:
            new_id = next(self._note_id_counter)
            if new_id not in self.PTPR.notes:
                break
        split_beat_id, split_metrical_pos = self.abs_tick_to_rel_tick(abs_split_tick)
        ideal_split_time = self.rel_tick_to_time(split_beat_id, split_metrical_pos)
        self.MTPR.split(id, new_id, abs_split_tick, split_beat_id, split_metrical_pos)
        self.PTPR.split(id, new_id, ideal_split_time, ideal_split_time)

    # mtpr, ptpr共通
    def note_join(self, id1: int, id2: int):
        self.MTPR.join(id1, id2)
        self.PTPR.join(id1, id2)

    def ptpr_note_insert(self, pitch: int, real_ontime: float, real_offtime: float):
        while True:
            new_id = next(self._note_id_counter)
            if new_id not in self.PTPR.notes:
                break
        on_beat_id = self.BDB.get_beat_id_from_time(real_ontime)
        off_beat_id = self.BDB.get_beat_id_from_time(real_offtime)
        ideal_ontime, abs_ontick, on_beat_id, on_metrical_pos = self.real_time_to_ideal_time(real_ontime, on_beat_id)
        ideal_offtime, abs_offtick, off_beat_id, off_metrical_pos = self.real_time_to_ideal_time(real_offtime, off_beat_id)
        self.PTPR.insert(new_id, pitch, ideal_ontime, real_ontime, ideal_offtime, real_offtime)
        self.MTPR.insert(new_id, pitch, abs_ontick, on_beat_id, on_metrical_pos, abs_offtick, off_beat_id, off_metrical_pos)

    def ptpr_note_move(self, id: int, pitch: int, real_ontime: float, real_offtime: float, update_ideal: bool = False):
        if update_ideal: #ideal再計算
            on_beat_id = self.BDB.get_beat_id_from_time(real_ontime)
            off_beat_id = self.BDB.get_beat_id_from_time(real_offtime)
            ideal_ontime, abs_ontick, on_beat_id, on_metrical_pos = self.real_time_to_ideal_time(real_ontime, on_beat_id)
            ideal_offtime, abs_offtick, off_beat_id, off_metrical_pos = self.real_time_to_ideal_time(real_offtime, off_beat_id)
            self.PTPR.move(id, pitch, ideal_ontime, real_ontime, ideal_offtime, real_offtime)
            self.MTPR.move(id, pitch, abs_ontick, on_beat_id, on_metrical_pos, abs_offtick, off_beat_id, off_metrical_pos)
        else: #デフォルト
            self.PTPR.move(id, pitch, None, real_ontime, None, real_offtime)
            self.MTPR.move(id, pitch, None, None, None, None, None, None)

    def ptpr_note_ontime_shift(self, id: int, real_ontime: float, update_ideal: bool = False):
        if update_ideal: #ideal再計算
            on_beat_id = self.BDB.get_beat_id_from_time(real_ontime)
            ideal_ontime, abs_ontick, on_beat_id, on_metrical_pos = self.real_time_to_ideal_time(real_ontime, on_beat_id)
            self.PTPR.ontime_shift(id, ideal_ontime, real_ontime)
            self.MTPR.ontime_shift(id, abs_ontick, on_beat_id, on_metrical_pos)
        else: #デフォルト
            self.PTPR.ontime_shift(id, None, real_ontime)

    def ptpr_note_offtime_shift(self, id: int, real_offtime: float, update_ideal: bool = False):
        if update_ideal: #ideal再計算
            off_beat_id = self.BDB.get_beat_id_from_time(real_offtime)
            ideal_offtime, abs_offtick, off_beat_id, off_metrical_pos = self.real_time_to_ideal_time(real_offtime, off_beat_id)
            self.PTPR.offtime_shift(id, ideal_offtime, real_offtime)
            self.MTPR.offtime_shift(id, abs_offtick, off_beat_id, off_metrical_pos)
        else: #デフォルト
            self.PTPR.offtime_shift(id, None, real_offtime)

    def ptpr_note_split(self, id: int, real_split_time: float):
        while True:
            new_id = next(self._note_id_counter)
            if new_id not in self.PTPR.notes:
                break
        split_beat_id = self.BDB.get_beat_id_from_time(real_split_time)
        ideal_split_time, abs_split_tick, split_beat_id, split_metrical_pos = self.real_time_to_ideal_time(
            real_split_time,
            split_beat_id,
            self.quantize_delta,
        )
        self.PTPR.split(id, new_id, ideal_split_time, real_split_time)
        self.MTPR.split(id, new_id, abs_split_tick, split_beat_id, split_metrical_pos)

    def load_onset_uni_prob(self, path: str = None, tpqn: int = 24):
        if path is None:
            path = os.path.join(
                os.path.dirname(__file__),
                "MetUniProb_ScoreBased_munus_JBM602_TPQN24.txt"
            )

        onset_uni_prob = np.zeros(tpqn, dtype=np.float64)

        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                cols = line.split()
                if len(cols) != 2:
                    raise ValueError(
                        f"invalid format in {path} at line {line_no}: expected 2 columns, got {len(cols)}"
                    )

                met_pos = int(cols[0])
                prob = float(cols[1])

                if not (0 <= met_pos < tpqn):
                    raise ValueError(
                        f"invalid metrical position in {path} at line {line_no}: met_pos={met_pos}"
                    )
                if prob <= 0.0:
                    raise ValueError(
                        f"invalid probability in {path} at line {line_no}: prob={prob}"
                    )

                onset_uni_prob[met_pos] = prob

        prob_sum = onset_uni_prob.sum()
        if prob_sum <= 0.0:
            raise ValueError(f"invalid unigram probability in {path}: sum must be positive")

        # 念のため正規化
        onset_uni_prob = onset_uni_prob / prob_sum

        self.onset_uni_prob = np.log(onset_uni_prob)
        return self.onset_uni_prob

    def load_onset_metrical_markov(self, path: str = None, tpqn: int = 24):
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "MetTrProb_ScoreBased_munus_JBM602_TPQN24.txt")

        trans_prob = np.zeros((tpqn, tpqn), dtype=np.float64)

        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                cols = line.split()
                if len(cols) != 3:
                    raise ValueError(
                        f"invalid format in {path} at line {line_no}: expected 3 columns, got {len(cols)}"
                    )

                prev_onset = int(cols[0])
                next_onset = int(cols[1])
                prob = float(cols[2])

                if not (0 <= prev_onset < tpqn and 0 <= next_onset < tpqn):
                    raise ValueError(
                        f"invalid metrical position in {path} at line {line_no}: "
                        f"prev={prev_onset}, next={next_onset}"
                    )
                if prob <= 0.0:
                    raise ValueError(
                        f"invalid probability in {path} at line {line_no}: prob={prob}"
                    )

                trans_prob[prev_onset, next_onset] = prob

        row_sum = trans_prob.sum(axis=1)
        if np.any(row_sum <= 0.0):
            bad_rows = np.where(row_sum <= 0.0)[0]
            raise ValueError(
                f"invalid transition matrix in {path}: row sums must be positive, bad rows={bad_rows}"
            )

        # 念のため各行を正規化
        trans_prob = trans_prob / row_sum[:, None]
        self.onset_metrical_markov = np.log(trans_prob)
        return self.onset_metrical_markov

    def load_offset_metrical_markov(self, path: str = None, tpqn: int = 24):
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "OffMetProb_new_ScoreBased_munus_JBM602_TPQN24.txt")

        offset_prob = np.zeros((tpqn, tpqn, tpqn), dtype=np.float64)

        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                cols = line.split()
                if len(cols) != 4:
                    raise ValueError(
                        f"invalid format in {path} at line {line_no}: expected 4 columns, got {len(cols)}"
                    )

                cur_onset = int(cols[0])
                next_onset = int(cols[1])
                cur_offset = int(cols[2])
                prob = float(cols[3])

                if not (
                    0 <= cur_onset < tpqn and
                    0 <= next_onset < tpqn and
                    0 <= cur_offset < tpqn
                ):
                    raise ValueError(
                        f"invalid metrical position in {path} at line {line_no}: "
                        f"cur_onset={cur_onset}, next_onset={next_onset}, cur_offset={cur_offset}"
                    )
                if prob <= 0.0:
                    raise ValueError(
                        f"invalid probability in {path} at line {line_no}: prob={prob}"
                    )

                offset_prob[cur_onset, next_onset, cur_offset] = prob

        pair_sum = offset_prob.sum(axis=2)
        if np.any(pair_sum <= 0.0):
            bad_pairs = np.argwhere(pair_sum <= 0.0)
            raise ValueError(
                f"invalid offset transition table in {path}: "
                f"(cur_onset, next_onset) with non-positive sum exist: {bad_pairs[:10].tolist()}"
            )

        # 念のため (cur_onset, next_onset) ごとに正規化
        offset_prob = offset_prob / pair_sum[:, :, None]

        self.offset_metrical_markov = np.log(offset_prob)
        return self.offset_metrical_markov

    def load_offset_uni_prob(self, path: str = None, tpqn: int = 24):
        if path is None:
            path = os.path.join(
                os.path.dirname(__file__),
                "OffUniProb_new_ScoreBased_munus_JBM602_TPQN24.txt"
            )

        offset_uni_prob = np.zeros(tpqn, dtype=np.float64)

        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                cols = line.split()
                if len(cols) != 2:
                    raise ValueError(
                        f"invalid format in {path} at line {line_no}: expected 2 columns, got {len(cols)}"
                    )

                met_pos = int(cols[0])
                prob = float(cols[1])

                if not (0 <= met_pos < tpqn):
                    raise ValueError(
                        f"invalid metrical position in {path} at line {line_no}: met_pos={met_pos}"
                    )
                if prob <= 0.0:
                    raise ValueError(
                        f"invalid probability in {path} at line {line_no}: prob={prob}"
                    )

                offset_uni_prob[met_pos] = prob

        prob_sum = offset_uni_prob.sum()
        if prob_sum <= 0.0:
            raise ValueError(f"invalid unigram probability in {path}: sum must be positive")

        # 念のため正規化
        offset_uni_prob = offset_uni_prob / prob_sum

        self.offset_uni_prob = np.log(offset_uni_prob)
        return self.offset_uni_prob

    def interpolation_onset_markov(self, interp_lambda: float):
        # lambda1: マルコフモデルの補完係数、lambda2: ユニグラム確率の補完係数
        interp_prob = np.zeros(self.onset_metrical_markov.shape, dtype=np.float64)
        onset_markov = np.exp(self.onset_metrical_markov)
        onset_uni = np.exp(self.onset_uni_prob)
        for i in range(interp_prob.shape[0]):
            for j in range(interp_prob.shape[1]):
                interp_prob[i, j] = interp_lambda * onset_markov[i, j] + (1 - interp_lambda) * onset_uni[j]
        row_sum = interp_prob.sum(axis=1)
        if np.any(row_sum <= 0.0):
            bad_rows = np.where(row_sum <= 0.0)[0]
            raise ValueError(
                f"invalid interpolated onset transition matrix: "
                f"row sums must be positive, bad rows={bad_rows}"
            )
        interpolation_onset_markov = interp_prob / row_sum[:, None]
        self.onset_metrical_markov = np.log(interpolation_onset_markov)
        return self.onset_metrical_markov
    
    def interpolation_offset_markov(self, interp_lambda: float):
        interp_prob = np.zeros(self.offset_metrical_markov.shape, dtype=np.float64)
        offset_markov = np.exp(self.offset_metrical_markov)
        offset_uni = np.exp(self.offset_uni_prob)
        for i in range(interp_prob.shape[0]):
            for j in range(interp_prob.shape[1]):
                for k in range(interp_prob.shape[2]):
                    interp_prob[i, j, k] = interp_lambda * offset_markov[i, j, k] + (1 - interp_lambda) * offset_uni[k]
        row_sum = interp_prob.sum(axis=2)
        if np.any(row_sum <= 0.0):
            bad_rows = np.where(row_sum <= 0.0)[0]
            raise ValueError(
                f"invalid interpolated onset transition matrix: "
                f"row sums must be positive, bad rows={bad_rows}"
            )
        interpolation_offset_markov = interp_prob / row_sum[:, None]
        self.offset_metrical_markov = np.log(interpolation_offset_markov)
        return self.offset_metrical_markov
