from dataclasses import dataclass
from typing import Callable

import numpy as np
from PySide6.QtWidgets import QGraphicsScene

from audio_features import FeatureCache, extract_vocal_features
from piano_roll_items import GRID_SEC


@dataclass(frozen=True)
class FeatureBackgroundData:
    hop_sec: float
    pitch_min: int
    pitch_max: int
    spectrogram: np.ndarray
    chroma: np.ndarray


def extract_feature_cache(audio_data: np.ndarray, sample_rate: int) -> FeatureCache:
    params = {
        "target_sr": 16000,
        "hop_length": 160,
        "midi_min": 21,
        "midi_max": 108,
        "top_db": 80.0,
        "n_fft": 1024,
        "win_length": 1024,
    }
    try:
        return extract_vocal_features(
            audio_data,
            int(sample_rate),
            enable_cqt=True,
            **params,
        )
    except ImportError:
        return extract_vocal_features(
            audio_data,
            int(sample_rate),
            enable_cqt=False,
            **params,
        )


def prepare_feature_background(cache: FeatureCache) -> FeatureBackgroundData:
    hop_sec = float(cache.hop_length) / float(cache.sr)
    pitch_min = int(cache.params["midi_min"])
    pitch_max = int(cache.params["midi_max"])

    spec_db = cache.spec_cqt_db if cache.spec_cqt_db is not None else cache.spec_stft_log_db
    spec_db = np.ascontiguousarray(spec_db[::-1, :])
    db_min = -60.0
    spectrogram = np.clip(spec_db, db_min, 0.0)
    spectrogram = (spectrogram - db_min) / (-db_min)
    spectrogram = np.power(spectrogram, 0.6)

    chroma_12 = cache.chroma_cqt if cache.chroma_cqt is not None else cache.chroma_stft
    pitches = np.arange(pitch_min, pitch_max + 1, dtype=np.int32)
    pitch_classes = pitches % 12
    chroma = np.ascontiguousarray(chroma_12[pitch_classes, :][::-1, :])

    return FeatureBackgroundData(
        hop_sec=hop_sec,
        pitch_min=pitch_min,
        pitch_max=pitch_max,
        spectrogram=spectrogram,
        chroma=chroma,
    )


def prepare_pitch_probability_background(
    log_probabilities: np.ndarray | None,
    pitch_min: int,
    pitch_max: int,
) -> np.ndarray | None:
    if log_probabilities is None or getattr(log_probabilities, "size", 0) == 0:
        return None

    logp = np.asarray(log_probabilities, dtype=np.float32)
    if logp.ndim != 2 or logp.shape[1] < 128:
        return None

    pitch_min = max(0, int(pitch_min))
    pitch_max = min(127, int(pitch_max))
    if pitch_max < pitch_min:
        return None

    logp_pitch = np.nan_to_num(
        logp[:, :128],
        nan=-np.inf,
        posinf=-np.inf,
        neginf=-np.inf,
    )
    max_values = np.max(logp_pitch, axis=1, keepdims=True)
    probabilities = np.exp(logp_pitch - max_values)
    denominator = np.sum(probabilities, axis=1, keepdims=True)
    denominator = np.where(denominator > 0, denominator, 1.0)
    probabilities = (probabilities / denominator).T

    pitch_slice = probabilities[pitch_min:pitch_max + 1, :]
    return np.ascontiguousarray(pitch_slice[::-1, :])


@dataclass
class FeatureControllerCallbacks:
    get_ptpr_scene: Callable
    get_mtpr_scene: Callable
    time_sec_to_abs_tick: Callable[[float], float]


class FeatureBackgroundController:
    def __init__(self, callbacks: FeatureControllerCallbacks):
        self.callbacks = callbacks

    @staticmethod
    def _invalidate(scene) -> None:
        scene.invalidate(scene.sceneRect(), QGraphicsScene.BackgroundLayer)

    def _build_mtpr_frame_x_edges(
        self,
        num_frames: int,
        hop_sec: float,
    ) -> np.ndarray | None:
        mtpr_scene = self.callbacks.get_mtpr_scene()
        if mtpr_scene is None or num_frames <= 0 or hop_sec <= 0:
            return None

        frame_times = np.arange(num_frames + 1, dtype=np.float64) * float(hop_sec)
        abs_ticks = np.asarray(
            [self.callbacks.time_sec_to_abs_tick(float(t)) for t in frame_times],
            dtype=np.float64,
        )
        return np.asarray(
            [float(mtpr_scene.tick_to_x(float(tick))) for tick in abs_ticks],
            dtype=np.float32,
        )

    def _apply_mtpr_feature(self, background: FeatureBackgroundData) -> None:
        mtpr_scene = self.callbacks.get_mtpr_scene()
        if mtpr_scene is None:
            return

        x_edges = self._build_mtpr_frame_x_edges(
            background.spectrogram.shape[1],
            background.hop_sec,
        )
        if x_edges is None:
            mtpr_scene.clear_background_spec()
        else:
            mtpr_scene.set_background_spec_mtpr(
                background.spectrogram,
                pitch_min=background.pitch_min,
                pitch_max=background.pitch_max,
                x_edges=x_edges,
            )

        x_edges = self._build_mtpr_frame_x_edges(
            background.chroma.shape[1],
            background.hop_sec,
        )
        if x_edges is None:
            mtpr_scene.clear_background_chroma()
        else:
            mtpr_scene.set_background_chroma_mtpr(
                background.chroma,
                pitch_min=background.pitch_min,
                pitch_max=background.pitch_max,
                x_edges=x_edges,
            )

    def apply_audio_features(self, feature_cache: FeatureCache | None) -> None:
        ptpr_scene = self.callbacks.get_ptpr_scene()
        if feature_cache is None or ptpr_scene is None:
            return

        background = prepare_feature_background(feature_cache)
        ptpr_scene.set_background_spec(
            background.spectrogram,
            background.hop_sec,
            background.pitch_min,
            background.pitch_max,
        )
        ptpr_scene.set_background_chroma(
            background.chroma,
            background.hop_sec,
            background.pitch_min,
            background.pitch_max,
        )
        self._apply_mtpr_feature(background)

        self._invalidate(ptpr_scene)
        mtpr_scene = self.callbacks.get_mtpr_scene()
        if mtpr_scene is not None:
            self._invalidate(mtpr_scene)

    def apply_pitch_probability(self, log_probabilities: np.ndarray | None) -> None:
        ptpr_scene = self.callbacks.get_ptpr_scene()
        mtpr_scene = self.callbacks.get_mtpr_scene()
        if ptpr_scene is None and mtpr_scene is None:
            return

        scene_ref = ptpr_scene if ptpr_scene is not None else mtpr_scene
        pitch_min = max(0, int(scene_ref.min_pitch))
        pitch_max = min(127, int(scene_ref.max_pitch))
        probability = prepare_pitch_probability_background(
            log_probabilities,
            pitch_min,
            pitch_max,
        )
        if probability is None:
            if ptpr_scene is not None:
                ptpr_scene.clear_background_pitch_prob()
                self._invalidate(ptpr_scene)
            if mtpr_scene is not None:
                mtpr_scene.clear_background_pitch_prob()
                self._invalidate(mtpr_scene)
            return

        if ptpr_scene is not None:
            ptpr_scene.set_background_pitch_prob(
                probability,
                hop_sec=GRID_SEC,
                pitch_min=pitch_min,
                pitch_max=pitch_max,
            )
        if mtpr_scene is not None:
            x_edges = self._build_mtpr_frame_x_edges(probability.shape[1], GRID_SEC)
            if x_edges is None:
                mtpr_scene.clear_background_pitch_prob()
            else:
                mtpr_scene.set_background_pitch_prob_mtpr(
                    probability,
                    pitch_min=pitch_min,
                    pitch_max=pitch_max,
                    x_edges=x_edges,
                )

        if ptpr_scene is not None:
            self._invalidate(ptpr_scene)
        if mtpr_scene is not None:
            self._invalidate(mtpr_scene)

    def refresh_mtpr_backgrounds(
        self,
        feature_cache: FeatureCache | None,
        log_probabilities: np.ndarray | None,
    ) -> None:
        mtpr_scene = self.callbacks.get_mtpr_scene()
        if mtpr_scene is None:
            return

        if feature_cache is not None:
            self._apply_mtpr_feature(prepare_feature_background(feature_cache))

        if log_probabilities is not None:
            self.apply_pitch_probability(log_probabilities)
        else:
            self._invalidate(mtpr_scene)
