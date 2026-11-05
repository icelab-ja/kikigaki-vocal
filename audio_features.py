# audio_features.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from scipy.signal import resample_poly, stft
except Exception as e:
    raise ImportError("scipy is required: pip install scipy") from e

# CQT は librosa を使う（STFT系は scipy/numpy のみで完結）
try:
    import librosa
    _HAS_LIBROSA = True
except Exception:
    _HAS_LIBROSA = False


@dataclass
class FeatureCache:
    sr: int
    hop_length: int
    times: np.ndarray                 # (T,)
    bin_midi: np.ndarray              # (F,) 例: 21..108
    spec_cqt_db: Optional[np.ndarray] # (F,T) or None
    spec_stft_log_db: np.ndarray      # (F,T)
    chroma_cqt: Optional[np.ndarray]  # (12,T) or None
    chroma_stft: np.ndarray           # (12,T)
    params: Dict


def _to_mono_float32(wav_2d: np.ndarray) -> np.ndarray:
    """
    wav_2d: (N, ch) を想定 (soundfile always_2d=True)
    """
    if wav_2d.ndim == 1:
        y = wav_2d
    else:
        # L/R 平均でmono化
        y = np.mean(wav_2d, axis=1)
    return np.asarray(y, dtype=np.float32)


def _resample_to_16k(y: np.ndarray, sr_in: int, sr_out: int = 16000) -> Tuple[np.ndarray, int]:
    if sr_in == sr_out:
        return y, sr_in

    # resample_poly は高品質かつ依存が軽い
    # up/down は整数が必要なので最大公約数で簡約
    from math import gcd
    g = gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    y_rs = resample_poly(y, up=up, down=down).astype(np.float32)
    return y_rs, sr_out


def _midi_bins(midi_min: int = 21, midi_max: int = 108) -> np.ndarray:
    # inclusive
    return np.arange(midi_min, midi_max + 1, dtype=np.float32)


def _midi_to_hz(midi: np.ndarray) -> np.ndarray:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def _power_to_db(power: np.ndarray, top_db: float = 80.0, eps: float = 1e-10) -> np.ndarray:
    # ref=max を 0dB とし、[-top_db, 0] にクリップ
    power = np.maximum(power, eps)
    ref = np.max(power)
    ref = max(ref, eps)
    db = 10.0 * np.log10(power / ref)
    db = np.maximum(db, -float(top_db))
    return db.astype(np.float32)


def _compute_stft_power(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    n_fft: int = 1024,
    win_length: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    戻り値:
      power: (F_stft, T)
      freqs: (F_stft,)
    """
    f, _, Zxx = stft(
        y,
        fs=sr,
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        boundary=None,  # center=False 相当（0番フレームが先頭）
        padded=False,
    )
    power = (np.abs(Zxx) ** 2).astype(np.float32)  # (F, T)
    freqs = f.astype(np.float32)
    return power, freqs


def _triangular_fb_midi_from_freqs(freqs: np.ndarray, midi_bins: np.ndarray) -> np.ndarray:
    """
    STFT周波数ビン → MIDI(半音)ビン(21..108) への三角フィルタバンク
    戻り値 FB: (F_midi, F_stft)
    """
    # 中心周波数（Hz）
    center_hz = _midi_to_hz(midi_bins)              # (F_midi,)
    left_hz = _midi_to_hz(midi_bins - 1.0)
    right_hz = _midi_to_hz(midi_bins + 1.0)

    Fm = midi_bins.shape[0]
    Fs = freqs.shape[0]
    FB = np.zeros((Fm, Fs), dtype=np.float32)

    # 各MIDI bin について三角形
    for i in range(Fm):
        l = left_hz[i]
        c = center_hz[i]
        r = right_hz[i]

        # 上り
        up = (freqs - l) / (c - l + 1e-12)
        # 下り
        down = (r - freqs) / (r - c + 1e-12)

        w = np.minimum(up, down)
        w = np.maximum(w, 0.0)

        # freqs が [l, r] の範囲にあるところだけ
        FB[i, :] = w.astype(np.float32)

    # 正規化（各フィルタの総和が1になるように）
    denom = np.sum(FB, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    FB /= denom
    return FB


def compute_spec_stft_log_db(
    y: np.ndarray,
    sr: int = 16000,
    hop_length: int = 160,
    midi_min: int = 21,
    midi_max: int = 108,
    n_fft: int = 1024,
    win_length: int = 1024,
    top_db: float = 80.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    power, freqs = _compute_stft_power(y, sr, hop_length, n_fft=n_fft, win_length=win_length)
    midi_bins = _midi_bins(midi_min, midi_max)  # (88,)
    FB = _triangular_fb_midi_from_freqs(freqs, midi_bins)  # (88, F_stft)

    # 射影: (88, F_stft) @ (F_stft, T) -> (88, T)
    proj = FB @ power
    spec_db = _power_to_db(proj, top_db=top_db)

    T = spec_db.shape[1]
    times = (np.arange(T, dtype=np.float32) * (hop_length / sr)).astype(np.float32)
    return spec_db, midi_bins, times


def _fold_to_chroma_from_midi_bins(midi_bins: np.ndarray, midi_power_or_mag: np.ndarray) -> np.ndarray:
    """
    midi_bins: (F_midi,) 例: 21..108
    midi_power_or_mag: (F_midi, T)
    """
    pcs = (midi_bins.astype(np.int32) % 12)  # (F_midi,)
    chroma = np.zeros((12, midi_power_or_mag.shape[1]), dtype=np.float32)
    for i in range(midi_bins.shape[0]):
        chroma[pcs[i], :] += midi_power_or_mag[i, :]

    # L1 正規化（フレームごと合計1）
    denom = np.sum(chroma, axis=0, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    chroma /= denom
    return chroma.astype(np.float32)


def compute_spec_cqt_db(
    y: np.ndarray,
    sr: int = 16000,
    hop_length: int = 160,
    midi_min: int = 21,
    midi_max: int = 108,
    bins_per_octave: int = 12,
    top_db: float = 80.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    戻り値:
      cqt_db: (F, T)  dB
      cqt_mag: (F, T) magnitude (クロマ計算などで使用)
      midi_bins: (F,)
      times: (T,)
    """
    if not _HAS_LIBROSA:
        raise ImportError("librosa is required to use CQT: pip install librosa")

    midi_bins = _midi_bins(midi_min, midi_max)  # (88,)
    fmin = float(_midi_to_hz(np.array([midi_min], dtype=np.float32))[0])
    n_bins = int(midi_max - midi_min + 1)  # 88

    # librosa.cqt は複素数。|.| を取り magnitude にする
    C = librosa.cqt(
        y,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
    )
    cqt_mag = np.abs(C).astype(np.float32)
    cqt_power = (cqt_mag ** 2).astype(np.float32)
    cqt_db = _power_to_db(cqt_power, top_db=top_db)

    T = cqt_db.shape[1]
    times = (np.arange(T, dtype=np.float32) * (hop_length / sr)).astype(np.float32)
    return cqt_db, cqt_mag, midi_bins, times


def extract_vocal_features(
    wav2_data_2d: np.ndarray,
    wav2_sr: int,
    *,
    target_sr: int = 16000,
    hop_length: int = 160,
    midi_min: int = 21,
    midi_max: int = 108,
    top_db: float = 80.0,
    n_fft: int = 1024,
    win_length: int = 1024,
    enable_cqt: bool = True,
) -> FeatureCache:
    """
    歌声WAVから、背景表示用の特徴量を一括抽出して返す。
    - Spec(CQT) / Chroma(CQT) は librosa が必要
    - Spec(STFT-log) / Chroma(STFT) は scipy/numpy だけで動く
    """
    y = _to_mono_float32(wav2_data_2d)
    y, sr = _resample_to_16k(y, wav2_sr, sr_out=target_sr)

    # STFT → MIDIログ軸スペクトログラム（必須）
    spec_stft_log_db, midi_bins, times = compute_spec_stft_log_db(
        y,
        sr=sr,
        hop_length=hop_length,
        midi_min=midi_min,
        midi_max=midi_max,
        n_fft=n_fft,
        win_length=win_length,
        top_db=top_db,
    )

    # STFT系クロマ：STFT→MIDI 88bin をfoldするため、dB前の "proj" が本来欲しい。
    # ここでは簡潔のため、dBの逆変換ではなく、再計算を避ける目的で
    # spec_stft_log_db を power 近似として扱わず、別途 power で計算する実装にする。
    # （正確にやる場合は compute_spec_stft_log_db を power返しに拡張するのが推奨）
    # ここでは安定動作優先で、内部を少し拡張して power も取る:
    power, freqs = _compute_stft_power(y, sr, hop_length, n_fft=n_fft, win_length=win_length)
    FB = _triangular_fb_midi_from_freqs(freqs, midi_bins)
    proj_power = FB @ power  # (88, T)
    chroma_stft = _fold_to_chroma_from_midi_bins(midi_bins, proj_power)

    spec_cqt_db = None
    chroma_cqt = None

    if enable_cqt:
        cqt_db, cqt_mag, midi_bins_cqt, times_cqt = compute_spec_cqt_db(
            y,
            sr=sr,
            hop_length=hop_length,
            midi_min=midi_min,
            midi_max=midi_max,
            bins_per_octave=12,
            top_db=top_db,
        )
        # 時間軸が一致している前提（hop_length固定）。万一ズレたら短い方に合わせる。
        T = min(times.shape[0], times_cqt.shape[0], cqt_db.shape[1], spec_stft_log_db.shape[1])
        times = times[:T]
        spec_stft_log_db = spec_stft_log_db[:, :T]
        chroma_stft = chroma_stft[:, :T]

        spec_cqt_db = cqt_db[:, :T]
        # CQT mag を fold して chroma
        chroma_cqt = _fold_to_chroma_from_midi_bins(midi_bins_cqt, cqt_mag[:, :T] ** 2)

    params = {
        "target_sr": target_sr,
        "hop_length": hop_length,
        "midi_min": midi_min,
        "midi_max": midi_max,
        "top_db": top_db,
        "n_fft": n_fft,
        "win_length": win_length,
        "enable_cqt": enable_cqt,
    }

    return FeatureCache(
        sr=sr,
        hop_length=hop_length,
        times=times,
        bin_midi=midi_bins,
        spec_cqt_db=spec_cqt_db,
        spec_stft_log_db=spec_stft_log_db,
        chroma_cqt=chroma_cqt,
        chroma_stft=chroma_stft,
        params=params,
    )
