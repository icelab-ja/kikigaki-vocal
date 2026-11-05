from dataclasses import dataclass
from typing import Callable, Optional

from candidate import Candidate
from candidate_items import (
    MTPRCandidateTickLineItem,
    OnsetCandidateLineItem,
    _apply_candidate_rank_alpha,
)
from piano_roll_items import GRID_SEC, NoteItem
from piano_roll_scene import MIN_NOTE_SEC
from physical_time_candidates import (
    PeakCandidate,
    PitchCandidateHLineItem,
    topk_peaks_around,
    topk_peaks_in_interval,
    topk_pitch_candidates_for_note,
)


@dataclass(frozen=True)
class CandidateControllerCallbacks:
    get_ptpr_scene: Callable[[], object | None]
    get_mtpr_scene: Callable[[], object | None]
    get_dtpr_adapter: Callable[[], object | None]
    get_confidence_data: Callable[[], object | None]
    get_pitch_logprobs: Callable[[], object | None]
    refresh_after_mtpr_apply: Callable[[Optional[list[int]]], None]


class CandidateController:
    def __init__(self, callbacks: CandidateControllerCallbacks):
        self.callbacks = callbacks
        self.ptpr_mode = "off"
        self.mtpr_mode = "onset"
        self.ptpr_pitch_enabled = False
        self.mtpr_pitch_enabled = True
        self.mtpr_engine: Optional[Candidate] = None
        self._grid_divisions = 4
        self._model_path: str | None = None
        self._onset_items = []
        self._pitch_items = []
        self._mtpr_tick_items = []
        self._mtpr_pitch_items = []
        self._onset_context = None
        self._mtpr_context = None
        self._onset_proxy = None

    @property
    def scene(self):
        return self.callbacks.get_ptpr_scene()

    @property
    def mtpr_scene(self):
        return self.callbacks.get_mtpr_scene()

    @property
    def dtpr_adapter(self):
        return self.callbacks.get_dtpr_adapter()

    def reset(self) -> None:
        self.clear_onset_candidates()
        self.clear_pitch_candidates()
        self.clear_mtpr_candidates()
        self.mtpr_engine = None

    def ensure_mtpr_engine(
        self,
        *,
        grid_divisions: int,
        model_path: str | None = None,
        reload_output: bool = False,
    ) -> Optional[Candidate]:
        self._grid_divisions = int(grid_divisions)
        self._model_path = model_path
        adapter = self.dtpr_adapter
        if adapter is None:
            self.mtpr_engine = None
            return None

        dtpr = getattr(adapter, "dtpr", None)
        if self.mtpr_engine is None or getattr(self.mtpr_engine, "dtpr", None) is not dtpr:
            self.mtpr_engine = Candidate(dtpr)
            reload_output = True

        try:
            self.mtpr_engine.set_separate_num(int(grid_divisions))
        except Exception:
            pass

        if reload_output and model_path:
            self.mtpr_engine.load_out_prob(model_path)
        return self.mtpr_engine

    def set_ptpr_mode(self, mode: str) -> None:
        self.ptpr_mode = str(mode)
        self.clear_onset_candidates()
        self.cancel_onset_candidate_ui()

    def set_mtpr_mode(self, mode: str) -> None:
        self.mtpr_mode = str(mode)
        self.clear_mtpr_tick_candidates()

    def set_pitch_modes(self, *, ptpr_enabled: bool, mtpr_enabled: bool) -> None:
        self.ptpr_pitch_enabled = bool(ptpr_enabled)
        self.mtpr_pitch_enabled = bool(mtpr_enabled)

    def has_active_candidates(self) -> bool:
        return bool(
            self.ptpr_mode != "off"
            or self.ptpr_pitch_enabled
            or self.mtpr_mode != "off"
            or self.mtpr_pitch_enabled
        )

    def refresh_candidates_from_selection(
        self,
        *,
        grid_divisions: int,
        model_path: str | None,
    ) -> None:
        self._grid_divisions = int(grid_divisions)
        self._model_path = model_path
        if self.ptpr_mode == "onset":
            self.refresh_onset_candidates_from_selection()
        elif self.ptpr_mode == "split":
            self.refresh_split_candidates_from_selection()
        else:
            self.clear_onset_candidates()
            self.cancel_onset_candidate_ui()

        self.refresh_pitch_candidates_from_selection()
        self.refresh_mtpr_candidates_from_selection(
            grid_divisions=grid_divisions,
            model_path=model_path,
        )

    def clear_mtpr_tick_candidates(self) -> None:
        scene = self.mtpr_scene
        if scene is not None:
            for item in self._mtpr_tick_items:
                try:
                    scene.removeItem(item)
                except Exception:
                    pass
        self._mtpr_tick_items = []

    def clear_mtpr_pitch_candidates(self) -> None:
        scene = self.mtpr_scene
        if scene is not None:
            for item in self._mtpr_pitch_items:
                try:
                    scene.removeItem(item)
                except Exception:
                    pass
        self._mtpr_pitch_items = []

    def clear_mtpr_candidates(self) -> None:
        self.clear_mtpr_tick_candidates()
        self.clear_mtpr_pitch_candidates()
        self._mtpr_context = None

    def refresh_mtpr_candidates_from_selection(
        self,
        *,
        grid_divisions: int,
        model_path: str | None,
    ) -> None:
        scene = self.mtpr_scene
        if self.mtpr_mode == "off" and not self.mtpr_pitch_enabled:
            self.clear_mtpr_candidates()
            return
        if scene is None:
            self.clear_mtpr_candidates()
            return

        engine = self.ensure_mtpr_engine(
            grid_divisions=grid_divisions,
            model_path=model_path,
            reload_output=False,
        )
        if engine is None:
            self.clear_mtpr_candidates()
            return

        selected = [item for item in scene.selectedItems() if isinstance(item, NoteItem)]
        if len(selected) != 1:
            self.clear_mtpr_candidates()
            return

        note_id = int(selected[0].note_id)
        adapter = self.dtpr_adapter
        note = adapter.get_mtpr_note_by_id(note_id) if adapter is not None else None
        if note is None:
            self.clear_mtpr_candidates()
            return

        self.clear_mtpr_tick_candidates()
        self.clear_mtpr_pitch_candidates()

        try:
            if self.mtpr_mode == "onset":
                ticks, probabilities = engine.calc_onset_cand_prob(note_id)
                self.show_mtpr_tick_candidate_lines(
                    note_id,
                    engine.top_k(ticks, probabilities, k=4),
                    kind="onset",
                )
            elif self.mtpr_mode == "offset":
                ticks, probabilities = engine.calc_offset_cand_prob(note_id)
                self.show_mtpr_tick_candidate_lines(
                    note_id,
                    engine.top_k(ticks, probabilities, k=4),
                    kind="offset",
                )
            elif self.mtpr_mode == "split":
                ticks, probabilities = engine.calc_split_cand_prob(note_id)
                self.show_mtpr_tick_candidate_lines(
                    note_id,
                    engine.top_k(ticks, probabilities, k=3),
                    kind="split",
                )

            if self.mtpr_pitch_enabled:
                # pitches, probabilities = engine.calc_pitch_cand_prob(note_id)
                pitches, probabilities = engine.calc_pitch_cand_prob_with_lang(note_id)
                self.show_mtpr_pitch_candidate_lines(
                    note_id,
                    engine.top_k_pitch(pitches, probabilities, k=3),
                )
        except Exception:
            self.clear_mtpr_candidates()

    def show_mtpr_tick_candidate_lines(self, note_id: int, ranked_candidates, *, kind: str) -> None:
        scene = self.mtpr_scene
        if scene is None:
            return
        items = []
        for rank, abs_tick, _log_probability in ranked_candidates:
            line = MTPRCandidateTickLineItem(
                x=float(scene.tick_to_x(int(abs_tick))),
                y0=0.0,
                y1=float(scene.scene_height),
                abs_tick=int(abs_tick),
                owner_note_id=int(note_id),
                on_click=self.on_mtpr_tick_candidate_line_clicked,
                kind=str(kind),
            )
            _apply_candidate_rank_alpha(line, int(rank), len(ranked_candidates))
            scene.addItem(line)
            items.append(line)
        self._mtpr_tick_items = items

    def show_mtpr_pitch_candidate_lines(self, note_id: int, ranked_candidates) -> None:
        scene = self.mtpr_scene
        adapter = self.dtpr_adapter
        if scene is None or adapter is None:
            return
        note = adapter.get_mtpr_note_by_id(int(note_id))
        if note is None:
            return

        x0 = float(scene.tick_to_x(int(note.abs_ontick)))
        x1 = float(scene.tick_to_x(int(note.abs_offtick)))
        items = []
        for rank, pitch, _log_probability in ranked_candidates:
            y = float(scene.pitch_to_y(int(pitch))) + float(scene.px_per_pitch) * 0.5
            item = PitchCandidateHLineItem(
                x0,
                x1,
                y,
                pitch=int(pitch),
                owner_note_id=int(note_id),
                on_click=self.on_mtpr_pitch_candidate_line_clicked,
            )
            _apply_candidate_rank_alpha(item, int(rank), len(ranked_candidates))
            scene.addItem(item)
            items.append(item)
        self._mtpr_pitch_items = items

    def on_mtpr_tick_candidate_line_clicked(self, line_item, scene_pos=None) -> None:
        self._mtpr_context = {
            "type": str(line_item.kind),
            "note_id": int(line_item.owner_note_id),
            "abs_tick": int(line_item.abs_tick),
        }
        if line_item.kind == "split":
            self.apply_mtpr_split_candidate()
        elif line_item.kind == "offset":
            self.apply_mtpr_offset_candidate()
        else:
            self.apply_mtpr_onset_candidate()

    def on_mtpr_pitch_candidate_line_clicked(self, line_item, scene_pos=None) -> None:
        self._mtpr_context = {
            "type": "pitch",
            "note_id": int(line_item.owner_note_id),
            "pitch": int(line_item.pitch),
        }
        self.apply_mtpr_pitch_candidate()

    def apply_mtpr_onset_candidate(self) -> None:
        scene = self.mtpr_scene
        adapter = self.dtpr_adapter
        context = self._mtpr_context
        if scene is None or adapter is None or not context or context.get("type") != "onset":
            return

        note_id = int(context["note_id"])
        new_tick = int(context["abs_tick"])
        note = adapter.get_mtpr_note_by_id(note_id)
        if note is None:
            return

        min_length = int(max(1.0, scene._domain_min_note_len()))
        if int(note.abs_offtick) - new_tick < min_length:
            new_tick = int(note.abs_offtick) - min_length
        target_item = scene.note_items.get(note_id)
        if target_item is None:
            return

        scene.start_edit_transaction(
            operation="mtpr_onset_candidate_apply",
            meta={"noteIds": [note_id]},
            candidate_meta={"type": "mtpr_onset", "from": int(note.abs_ontick), "to": new_tick},
        )
        x = float(scene.tick_to_x(new_tick))
        old_right_x = float(target_item.pos().x() + target_item.rect().width())
        width = max(float(scene.min_note_width_px()), old_right_x - x)
        old_resize_mode = getattr(target_item, "resize_mode", None)
        target_item.resize_mode = "left"
        try:
            target_item.setPos(x, float(target_item.pos().y()))
            target_item.setRect(0, 0, width, float(target_item.rect().height()))
        finally:
            target_item.resize_mode = old_resize_mode

        scene.commit_edit_transaction()
        self.callbacks.refresh_after_mtpr_apply([note_id])
        self._refresh_after_apply()

    def apply_mtpr_offset_candidate(self) -> None:
        scene = self.mtpr_scene
        adapter = self.dtpr_adapter
        context = self._mtpr_context
        if scene is None or adapter is None or not context or context.get("type") != "offset":
            return

        note_id = int(context["note_id"])
        new_tick = int(context["abs_tick"])
        note = adapter.get_mtpr_note_by_id(note_id)
        if note is None:
            return

        min_length = int(max(1.0, scene._domain_min_note_len()))
        if new_tick - int(note.abs_ontick) < min_length:
            new_tick = int(note.abs_ontick) + min_length
        target_item = scene.note_items.get(note_id)
        if target_item is None:
            return

        scene.start_edit_transaction(
            operation="mtpr_offset_candidate_apply",
            meta={"noteIds": [note_id]},
            candidate_meta={"type": "mtpr_offset", "from": int(note.abs_offtick), "to": new_tick},
        )
        x0 = float(target_item.pos().x())
        x1 = float(scene.tick_to_x(new_tick))
        target_item.setRect(
            0,
            0,
            max(float(scene.min_note_width_px()), x1 - x0),
            float(target_item.rect().height()),
        )

        scene.commit_edit_transaction()
        self.callbacks.refresh_after_mtpr_apply([note_id])
        self._refresh_after_apply()

    def apply_mtpr_split_candidate(self) -> None:
        scene = self.mtpr_scene
        adapter = self.dtpr_adapter
        context = self._mtpr_context
        if scene is None or adapter is None or not context or context.get("type") != "split":
            return

        note_id = int(context["note_id"])
        split_tick = int(context["abs_tick"])
        note = adapter.get_mtpr_note_by_id(note_id)
        if note is None:
            return

        if not scene.can_split_note_at(note, split_tick):
            return

        scene.start_edit_transaction(
            operation="mtpr_split_candidate_apply",
            candidate_meta={"type": "mtpr_split", "note_id": note_id, "at": split_tick},
        )
        split_ids = adapter.mtpr_note_split(id=note_id, abs_split_tick=split_tick)
        if not split_ids:
            scene.commit_edit_transaction()
            return
        selected_ids = [int(split_id) for split_id in split_ids]
        self.callbacks.refresh_after_mtpr_apply(selected_ids)
        scene.commit_edit_transaction()
        self._refresh_after_apply()

    def apply_mtpr_pitch_candidate(self) -> None:
        scene = self.mtpr_scene
        adapter = self.dtpr_adapter
        context = self._mtpr_context
        if scene is None or adapter is None or not context or context.get("type") != "pitch":
            return

        note_id = int(context["note_id"])
        new_pitch = int(context["pitch"])
        note = adapter.get_mtpr_note_by_id(note_id)
        if note is None or new_pitch == int(note.pitch):
            return

        old_pitch = int(note.pitch)
        scene.start_edit_transaction(
            operation="mtpr_pitch_candidate_apply",
            candidate_meta={"type": "mtpr_pitch", "from": old_pitch, "to": new_pitch},
        )
        adapter.mtpr_note_move(
            id=note_id,
            pitch=new_pitch,
            abs_ontick=int(note.abs_ontick),
            abs_offtick=int(note.abs_offtick),
            keep_real=False,
        )
        self.callbacks.refresh_after_mtpr_apply([note_id])
        scene.commit_edit_transaction()
        self._refresh_after_apply()

    def _refresh_after_apply(self) -> None:
        self.refresh_candidates_from_selection(
            grid_divisions=self._grid_divisions,
            model_path=self._model_path,
        )

    def clear_onset_candidates(self) -> None:
        scene = self.scene
        if scene is not None:
            for item in self._onset_items:
                try:
                    scene.removeItem(item)
                except Exception:
                    pass
        self._onset_items = []
        if self._onset_proxy is not None and scene is not None:
            try:
                scene.removeItem(self._onset_proxy)
            except Exception:
                pass
        self._onset_proxy = None
        self._onset_context = None

    def refresh_onset_candidates_from_selection(self) -> None:
        scene = self.scene
        if scene is None or self.ptpr_mode != "onset":
            return
        confidence = self.callbacks.get_confidence_data()
        if confidence is None or getattr(confidence, "size", 0) == 0:
            self.clear_onset_candidates()
            return
        selected = [item for item in scene.selectedItems() if isinstance(item, NoteItem)]
        if len(selected) != 1:
            self.clear_onset_candidates()
            return
        note_id = int(selected[0].note_id)
        note = scene._get_display_note_by_id(note_id)
        if note is None:
            self.clear_onset_candidates()
            return

        candidates = topk_peaks_around(
            confidence,
            center_time_sec=float(scene._note_onset(note)),
            hop_sec=GRID_SEC,
            window_sec=0.15,
            topk=3,
            min_separation_sec=0.02,
            exclude_center_sec=0.02,
        )
        self.show_onset_candidate_lines(note_id, candidates)

    def show_onset_candidate_lines(self, note_id: int, candidates: list[PeakCandidate]) -> None:
        self.clear_onset_candidates()
        scene = self.scene
        if scene is None:
            return
        for candidate in candidates:
            line = OnsetCandidateLineItem(
                x=float(scene.time_to_x(candidate.time_sec)),
                y0=0.0,
                y1=float(scene.scene_height),
                time_sec=candidate.time_sec,
                owner_note_id=note_id,
                on_click=self.on_onset_candidate_line_clicked,
                kind="onset",
            )
            scene.addItem(line)
            self._onset_items.append(line)

    def on_onset_candidate_line_clicked(self, line_item, scene_pos=None) -> None:
        if self.scene is None:
            return
        self.cancel_onset_candidate_ui()
        self._onset_context = {
            "type": str(line_item.kind),
            "note_id": int(line_item.owner_note_id),
            "time_sec": float(line_item.time_sec),
        }
        if line_item.kind == "split":
            self.apply_split_candidate()
        else:
            self.apply_onset_candidate()

    def cancel_onset_candidate_ui(self) -> None:
        self._dispose_onset_candidate_proxy()
        self._onset_context = None

    def _dispose_onset_candidate_proxy(self) -> None:
        proxy = self._onset_proxy
        if proxy is None:
            return
        self._onset_proxy = None
        scene = self.scene
        if scene is not None:
            try:
                scene.removeItem(proxy)
            except Exception:
                pass
        try:
            widget = proxy.widget()
            if widget is not None:
                widget.deleteLater()
        except Exception:
            pass
        try:
            proxy.deleteLater()
        except Exception:
            pass

    def apply_onset_candidate(self) -> None:
        scene = self.scene
        context = self._onset_context
        if scene is None or not context:
            return
        note_id = int(context["note_id"])
        new_onset = float(context["time_sec"])
        note = scene.dtpr_adapter.get_ptpr_note_by_id(note_id)
        if note is None:
            self.cancel_onset_candidate_ui()
            return

        old_offset = float(note.real_offtime)
        if old_offset - new_onset < MIN_NOTE_SEC:
            new_onset = old_offset - MIN_NOTE_SEC
        target_item = next(
            (
                item
                for item in scene.items()
                if isinstance(item, NoteItem) and item.note_id == note_id
            ),
            None,
        )
        if target_item is None:
            self.cancel_onset_candidate_ui()
            return

        scene.start_edit_transaction(
            operation="ptpr_onset_candidate_apply",
            meta={"noteIds": [note_id]},
            candidate_meta={"type": "onset", "from": float(note.real_ontime), "to": new_onset},
        )
        x = float(scene.time_to_x(new_onset))
        width = float(scene.dur_to_w(old_offset - new_onset))
        old_resize_mode = getattr(target_item, "resize_mode", None)
        target_item.resize_mode = "left"
        try:
            target_item.setPos(x, float(target_item.pos().y()))
            target_item.setRect(0, 0, width, float(target_item.rect().height()))
        finally:
            target_item.resize_mode = old_resize_mode
        scene.commit_edit_transaction()
        scene.update()
        self.cancel_onset_candidate_ui()
        self.refresh_onset_candidates_from_selection()

    def apply_split_candidate(self) -> None:
        scene = self.scene
        context = self._onset_context
        if scene is None or not context or context.get("type") != "split":
            return
        note_id = int(context["note_id"])
        split_sec = round(float(context["time_sec"]) / GRID_SEC) * GRID_SEC
        note = scene.dtpr_adapter.get_ptpr_note_by_id(note_id)
        if note is None:
            self.cancel_onset_candidate_ui()
            return
        if not scene.can_split_note_at(note, split_sec):
            self.cancel_onset_candidate_ui()
            return

        scene.start_edit_transaction(
            operation="split_candidate_apply",
            candidate_meta={"type": "split", "note_id": note_id, "at": split_sec},
        )
        scene.split_note_at(note, split_sec)
        scene.commit_edit_transaction()
        scene.update()
        self.cancel_onset_candidate_ui()
        self.clear_onset_candidates()
        scene.clearSelection()

    def refresh_split_candidates_from_selection(self) -> None:
        scene = self.scene
        if scene is None or self.ptpr_mode != "split":
            return
        confidence = self.callbacks.get_confidence_data()
        if confidence is None or getattr(confidence, "size", 0) == 0:
            self.clear_onset_candidates()
            return
        selected = [item for item in scene.selectedItems() if isinstance(item, NoteItem)]
        if len(selected) != 1:
            self.clear_onset_candidates()
            return
        note_id = int(selected[0].note_id)
        note = scene.dtpr_adapter.get_ptpr_note_by_id(note_id)
        if note is None:
            self.clear_onset_candidates()
            return

        self.clear_onset_candidates()
        candidates = topk_peaks_in_interval(
            confidence,
            onset_sec=float(note.real_ontime),
            offset_sec=float(note.real_offtime),
            hop_sec=GRID_SEC,
            min_margin_sec=0.0,
            topk=3,
            min_separation_sec=0.02,
        )
        for candidate in candidates:
            line = OnsetCandidateLineItem(
                x=float(scene.time_to_x(candidate.time_sec)),
                y0=0.0,
                y1=float(scene.sceneRect().height()),
                time_sec=float(candidate.time_sec),
                owner_note_id=note_id,
                on_click=self.on_onset_candidate_line_clicked,
                kind="split",
            )
            scene.addItem(line)
            self._onset_items.append(line)

    def apply_candidate_by_number_key(self, number: int) -> bool:
        scene = self.scene
        if scene is None:
            return False
        items = [item for item in self._onset_items if isinstance(item, OnsetCandidateLineItem)]
        items.sort(key=lambda item: float(getattr(item, "time_sec", 0.0)))
        if number < 1 or number > len(items):
            return False
        selected = [item for item in scene.selectedItems() if isinstance(item, NoteItem)]
        if len(selected) != 1:
            return False

        line = items[number - 1]
        selected_note_id = int(selected[0].note_id)
        owner_note_id = int(getattr(line, "owner_note_id", selected_note_id))
        if owner_note_id != selected_note_id:
            return False
        self._onset_context = {
            "note_id": owner_note_id,
            "time_sec": float(getattr(line, "time_sec", 0.0)),
            "type": str(getattr(line, "kind", "")),
        }
        if self._onset_context["type"] == "split":
            self.apply_split_candidate()
            return True
        if self._onset_context["type"] == "onset":
            self.apply_onset_candidate()
            return True
        return False

    def clear_pitch_candidates(self) -> None:
        scene = self.scene
        if scene is not None:
            for item in self._pitch_items:
                try:
                    scene.removeItem(item)
                except Exception:
                    pass
        self._pitch_items = []

    def refresh_pitch_candidates_from_selection(self) -> None:
        scene = self.scene
        if scene is None:
            return
        if not self.ptpr_pitch_enabled:
            self.clear_pitch_candidates()
            return
        log_probabilities = self.callbacks.get_pitch_logprobs()
        if log_probabilities is None or getattr(log_probabilities, "size", 0) == 0:
            self.clear_pitch_candidates()
            return
        selected = [item for item in scene.selectedItems() if isinstance(item, NoteItem)]
        if len(selected) != 1:
            self.clear_pitch_candidates()
            return
        note_id = int(selected[0].note_id)
        note = scene.dtpr_adapter.get_ptpr_note_by_id(note_id)
        if note is None:
            self.clear_pitch_candidates()
            return

        self.clear_pitch_candidates()
        candidates = topk_pitch_candidates_for_note(
            log_probabilities,
            onset_sec=float(note.real_ontime),
            offset_sec=float(note.real_offtime),
            hop_sec=GRID_SEC,
            exclude_pitch=int(note.pitch),
            topk=3,
            pitch_min=scene.min_pitch,
            pitch_max=scene.max_pitch,
        )
        x0 = float(scene.time_to_x(float(note.real_ontime)))
        x1 = float(scene.time_to_x(float(note.real_offtime)))
        items = []
        for candidate in candidates:
            y = float(scene.pitch_to_y(int(candidate.pitch))) + float(scene.px_per_pitch) * 0.5
            item = PitchCandidateHLineItem(
                x0,
                x1,
                y,
                pitch=int(candidate.pitch),
                owner_note_id=note_id,
                on_click=self.on_pitch_candidate_line_clicked,
            )
            scene.addItem(item)
            items.append(item)
        self._pitch_items = items

    def on_pitch_candidate_line_clicked(self, line_item, scene_pos=None) -> None:
        scene = self.scene
        if scene is None:
            return
        note_id = int(line_item.owner_note_id)
        new_pitch = int(line_item.pitch)
        note = scene.dtpr_adapter.get_ptpr_note_by_id(note_id)
        if note is None or new_pitch == int(note.pitch):
            return
        target_item = next(
            (
                item
                for item in scene.items()
                if isinstance(item, NoteItem) and int(item.note_id) == note_id
            ),
            None,
        )
        if target_item is None:
            return

        old_pitch = int(note.pitch)
        scene.start_edit_transaction(
            operation="pitch_candidate_apply",
            candidate_meta={"type": "pitch", "from": old_pitch, "to": new_pitch},
        )
        target_item.setPos(float(target_item.pos().x()), float(scene.pitch_to_y(new_pitch)))
        scene.commit_edit_transaction()
        scene.update()
        self.refresh_pitch_candidates_from_selection()
