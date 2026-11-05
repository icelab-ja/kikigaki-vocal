import os
import numpy as np
from scipy.special import logsumexp
from DataRepresentation import  BDB, MTPR, PTPR, DTPR, TPQN

separate = [4, 8, 12, 24]

class Candidate: # すべて対数確率
    def __init__(self, dtpr: DTPR):
        self.dtpr = dtpr
        self.tpqn = 24
        self.onset_uni_prob = None # 24
        self.onset_metrical_markov = None # 24 * 24
        self.offset_uni_prob = None # 24
        self.offset_metrical_markov = None # 24 * 24 * 24
        self.rel_pitch_class_uni_prob = None # 12
        self.rel_pitch_class_markov = None # 12 * 12
        self.pitch_output_prob = None # T * 129
        self.onset_output_prob = None # T
        self.separate_num = 4
        self.cand_gauss_sigma = 50 # ms 標準偏差が50ms
        self.frame_hop_ms = 10
        self.off_cand_gauss_sigma = 100 # ms 標準偏差が100ms
        self.n_cand = 3
        # self.onset_markov_weight = 1.6 # 実験
        self.onset_markov_weight = 2.0
        self.offset_markov_weight = 2.0
        self.rel_pitch_class_markov_weight = 1.0
        self.product_pitch_range = 10 # 定式化のM
        self.beta = 1.0
        self.interp_onset = 0.9
        self.interp_offset = 0.8
        self.load_onset_uni_prob()
        self.load_onset_metrical_markov()
        self.load_offset_uni_prob()
        self.load_offset_metrical_markov()
        self.interpolation_onset_markov(interp_lambda=self.interp_onset)
        self.interpolation_offset_markov(interp_lambda=self.interp_offset)
        self.load_rel_pitch_class_uni_prob()
        self.load_rel_pitch_class_markov()
    
    def set_separate_num(self, separate_num: int):
        if separate_num not in separate:
            raise ValueError(f"separate number must be one of {separate}, got {separate_num}")
        self.separate_num = separate_num
    
    def set_lm_weight(self, lm_weight: float):
        self.onset_markov_weight = lm_weight
        self.offset_markov_weight = lm_weight

    def set_lm_pitch_param(self, lm_weight: float = 1.0, range: int = 10):
        self.rel_pitch_class_markov_weight = lm_weight
        self.product_pitch_range = range

    def load_onset_uni_prob(self, path: str = None):
        if path is None:
            path = os.path.join(
                os.path.dirname(__file__),
                "MetUniProb_ScoreBased_munus_JBM602_TPQN24.txt"
            )

        onset_uni_prob = np.zeros(self.tpqn, dtype=np.float64)

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

                if not (0 <= met_pos < self.tpqn):
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

    def load_offset_uni_prob(self, path: str = None):
        if path is None:
            path = os.path.join(
                os.path.dirname(__file__),
                "OffUniProb_new_ScoreBased_munus_JBM602_TPQN24.txt"
            )

        offset_uni_prob = np.zeros(self.tpqn, dtype=np.float64)

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

                if not (0 <= met_pos < self.tpqn):
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

    def load_onset_metrical_markov(self, path: str = None):
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "MetTrProb_ScoreBased_munus_JBM602_TPQN24.txt")

        trans_prob = np.zeros((self.tpqn, self.tpqn), dtype=np.float64)

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

                if not (0 <= prev_onset < self.tpqn and 0 <= next_onset < self.tpqn):
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

    def interpolation_onset_markov(self, interp_lambda: float):
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

    def load_offset_metrical_markov(self, path: str = None):
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "OffMetProb_new_ScoreBased_munus_JBM602_TPQN24.txt")

        offset_prob = np.zeros((self.tpqn, self.tpqn, self.tpqn), dtype=np.float64)

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
                    0 <= cur_onset < self.tpqn and
                    0 <= next_onset < self.tpqn and
                    0 <= cur_offset < self.tpqn
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

    def load_rel_pitch_class_uni_prob(self, path: str = None):
        if path is None:
            path = os.path.join(
                os.path.dirname(__file__),
                "RelPitchClassUniProb_ScoreBased_munus_JBM602.txt"
            )

        rel_pitch_class_uni_prob = np.zeros(12, dtype=np.float64)

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

                pitch_class = int(cols[0])
                prob = float(cols[1])

                if not (0 <= pitch_class < 12):
                    raise ValueError(
                        f"invalid relative pitch class in {path} at line {line_no}: pitch_class={pitch_class}"
                    )
                if prob <= 0.0:
                    raise ValueError(
                        f"invalid probability in {path} at line {line_no}: prob={prob}"
                    )

                rel_pitch_class_uni_prob[pitch_class] = prob

        prob_sum = rel_pitch_class_uni_prob.sum()
        if prob_sum <= 0.0:
            raise ValueError(f"invalid relative pitch class unigram probability in {path}: sum must be positive")

        rel_pitch_class_uni_prob = rel_pitch_class_uni_prob / prob_sum
        self.rel_pitch_class_uni_prob = np.log(rel_pitch_class_uni_prob)
        return self.rel_pitch_class_uni_prob

    def load_rel_pitch_class_markov(self, path: str = None):
        if path is None:
            path = os.path.join(
                os.path.dirname(__file__),
                "RelPitchClassTrProb_ScoreBased_munus_JBM602.txt"
            )

        trans_prob = np.zeros((12, 12), dtype=np.float64)

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

                prev_pitch_class = int(cols[0])
                next_pitch_class = int(cols[1])
                prob = float(cols[2])

                if not (0 <= prev_pitch_class < 12 and 0 <= next_pitch_class < 12):
                    raise ValueError(
                        f"invalid relative pitch class in {path} at line {line_no}: "
                        f"prev={prev_pitch_class}, next={next_pitch_class}"
                    )
                if prob <= 0.0:
                    raise ValueError(
                        f"invalid probability in {path} at line {line_no}: prob={prob}"
                    )

                trans_prob[prev_pitch_class, next_pitch_class] = prob

        row_sum = trans_prob.sum(axis=1)
        if np.any(row_sum <= 0.0):
            bad_rows = np.where(row_sum <= 0.0)[0]
            raise ValueError(
                f"invalid relative pitch class transition matrix in {path}: "
                f"row sums must be positive, bad rows={bad_rows}"
            )

        trans_prob = trans_prob / row_sum[:, None]
        self.rel_pitch_class_markov = np.log(trans_prob)
        return self.rel_pitch_class_markov

    def load_out_prob(self, path: str):
        ###
        # 入力:
        #   path: npyファイルのパス
        #
        # npyファイルは shape [T, 130] を想定
        #   0:128   -> pitch (0-127)
        #   128     -> rest
        #   129     -> onset
        #
        # すでに正規化済みの対数確率が入っている前提なので、
        # 再正規化や再計算は行わず、そのまま self に格納する
        ###
        out_prob = np.load(path)

        if out_prob.ndim != 2:
            raise ValueError(
                f"output probability array must be 2D, got ndim={out_prob.ndim}"
            )

        if out_prob.shape[1] != 130:
            raise ValueError(
                f"output probability array must have shape [T, 130], got {out_prob.shape}"
            )

        self.pitch_output_prob = out_prob[:, :129]
        self.onset_output_prob = out_prob[:, 129]

        return self.pitch_output_prob, self.onset_output_prob

    def _truncated_log_gauss_score(self, log_prob_vec: np.ndarray, center_frame: float, sigma_frame: float) -> float:
        """
        log_prob_vec: 各フレームの対数確率
        center_frame: ガウス中心
        sigma_frame: フレーム単位の標準偏差

        return:
            log sum_t exp(log_prob_vec[t] + log_gauss[t])
            ただし t は center_frame ± 3*sigma_frame の範囲に制限
        """
        if sigma_frame <= 0:
            raise ValueError(f"sigma_frame must be positive, got {sigma_frame}")

        T = len(log_prob_vec)
        radius = int(np.ceil(3.0 * sigma_frame))
        center_int = int(round(center_frame))

        left = max(0, center_int - radius)
        right = min(T - 1, center_int + radius)

        local_idx = np.arange(left, right + 1, dtype=np.float64)
        log_gauss = -0.5 * ((local_idx - center_frame) / sigma_frame) ** 2

        return logsumexp(log_prob_vec[left:right + 1] + log_gauss)

    def calc_onset_cand_prob(self, note_id: int):
        ###
        # n番目の音符の発音時刻候補確率を出力
        # 入力: n番目の音符のid
        # 出力: 絶対tick, 修正候補の確率
        ###

        # n-1番目の音符の属するビートからn+1番目の音符の属するビートの範囲の配列を作成
        note = self.dtpr.MTPR.get_note(note_id)
        note_idx = self.dtpr.MTPR.get_note_idx(note_id)
        max_prob_beat_index = len(self.dtpr.BDB.index) - 2
        if note_idx > 0:
            prev_note = self.dtpr.MTPR.get_note_from_idx(note_idx-1)
            prev_note_met_pos = (prev_note.on_metrical_pos%TPQN) // (TPQN // self.tpqn) # TPQN=1440から24に対応付けるため60で割る
            prev_note_ontick = prev_note.abs_ontick
            prev_note_beat = self.dtpr.BDB.get_beat(prev_note.on_beat_id)
            prev_note_beat_index = min(self.dtpr.BDB.get_beat_idx(prev_note_beat.id), max_prob_beat_index)
        else:
            prev_note = None
            prev_note_met_pos = 0
            note_on_beat = self.dtpr.BDB.get_beat(note.on_beat_id)
            # prev_note_beat_index = max(0, self.dtpr.BDB.get_beat_idx(note_on_beat.id)-1)
            prev_note_beat_index = min(max(0, self.dtpr.BDB.get_beat_idx(note_on_beat.id) - 1), max_prob_beat_index)
            prev_note_ontick = prev_note_beat_index*TPQN
        if note_idx < len(self.dtpr.MTPR.index) - 1:
            next_note = self.dtpr.MTPR.get_note_from_idx(note_idx+1)
            next_note_met_pos = (next_note.on_metrical_pos%TPQN) // (TPQN // self.tpqn)
            next_note_beat = self.dtpr.BDB.get_beat(next_note.off_beat_id)
            # next_note_beat_index = self.dtpr.BDB.get_beat_idx(next_note_beat.id)
            next_note_beat_index = min(self.dtpr.BDB.get_beat_idx(next_note_beat.id), max_prob_beat_index)
        else:
            next_note = None
            next_note_met_pos = 0
            note_off_beat = self.dtpr.BDB.get_beat(note.off_beat_id)
            # next_note_beat_index = min(len(self.dtpr.BDB.index)-1 ,self.dtpr.BDB.get_beat_idx(note_off_beat.id)+1)
            next_note_beat_index = min(self.dtpr.BDB.get_beat_idx(note_off_beat.id) + 1, max_prob_beat_index)

        prob = np.zeros(self.tpqn*(next_note_beat_index - prev_note_beat_index + 1), dtype=np.float64) - 1000.0

        # beat timeを利用し、time配列を設定する
        time = np.zeros(prob.shape[0], dtype=np.float64)
        time_abs_tick = np.zeros(prob.shape[0], dtype=int)
        beat_time_array = []
        for i in range(prev_note_beat_index, next_note_beat_index + 2):
            beat = self.dtpr.BDB.get_beat_from_idx(i)
            beat_time = self.dtpr.BDB.get_beat_time_from_id(beat.id)
            beat_time_array.append(beat_time)

        for i in range(len(time)):
            beat_time = beat_time_array[i//self.tpqn]
            next_beat_time = beat_time_array[i//self.tpqn + 1]
            time[i] = beat_time + (i%self.tpqn)*(next_beat_time - beat_time)/self.tpqn
            time_abs_tick[i] = prev_note_beat_index * TPQN + i * (TPQN // self.tpqn)

        sigma_frame = self.cand_gauss_sigma / self.frame_hop_ms   # 50ms -> 5 frame

        prev_note_on = -1
        next_note_on = len(prob)

        if prev_note is not None:
            prev_note_on = (prev_note.abs_ontick - prev_note_beat_index * TPQN) // (TPQN // self.tpqn)
        
        if next_note is not None:
            next_note_on = (next_note.abs_ontick - prev_note_beat_index * TPQN) // (TPQN // self.tpqn)

        # 確率を計算
        step = int(self.tpqn/self.separate_num) #分割数から計算不要な時刻はスキップ
        for i in range(0, len(prob), step):
            # if i <= prev_note_on or i >= next_note_on:# 時刻がprev_noteのオンセット時刻以前またはnext_noteのオンセット時刻以降はスキップ
            #     continue
            if time_abs_tick[i] <= prev_note_ontick or time_abs_tick[i] >= note.abs_offtick:# 時刻がprev_noteのオンセット時刻以前またはnext_noteのオンセット時刻以降はスキップ
                continue
            if time_abs_tick[i] == note.abs_ontick:
                continue

            # メトリカルマルコフモデルの計算
            if prev_note is None:
                prob[i] = self.onset_markov_weight * self.onset_uni_prob[i%self.tpqn]
            elif next_note is None:
                prob[i] = self.onset_markov_weight * self.onset_metrical_markov[prev_note_met_pos, i%self.tpqn]
            else:
                prob[i] = self.onset_markov_weight * (self.onset_metrical_markov[prev_note_met_pos, i%self.tpqn] + self.onset_metrical_markov[i%self.tpqn, next_note_met_pos])
            
            # 出力確率の計算
            cand_frame = time[i] * 1000.0 / self.frame_hop_ms
            prob[i] += self._truncated_log_gauss_score(self.onset_output_prob, cand_frame, sigma_frame)

        return time_abs_tick, prob

    def calc_offset_cand_prob(self, note_id: int):
        ###
        #入力: 音符nのid
        #
        ###
        note = self.dtpr.MTPR.get_note(note_id)
        note_onset_met_pos = (note.on_metrical_pos%TPQN) // (TPQN // self.tpqn)
        note_idx = self.dtpr.MTPR.get_note_idx(note_id)
        pitch = note.pitch

        if note_idx < len(self.dtpr.MTPR.index) - 1:
            next_note = self.dtpr.MTPR.get_note_from_idx(note_idx+1)
            next_note_met_pos = (next_note.on_metrical_pos%TPQN) // (TPQN // self.tpqn)
            next_note_ontick = next_note.abs_ontick
        else: # 要修正
            next_note = None
            next_note_met_pos = 0
            next_note_ontick = note.abs_offtick - note.abs_offtick%TPQN + TPQN
        
        # n番目の音符の属するビートからn+1番目の音符の属するビートの範囲の配列を作成
        note_beat = self.dtpr.BDB.get_beat(note.on_beat_id)
        note_beat_idx = self.dtpr.BDB.get_beat_idx(note_beat.id)

        max_prob_beat_idx = len(self.dtpr.BDB.index) - 2
        if next_note is not None:
            next_note_beat = self.dtpr.BDB.get_beat(next_note.off_beat_id)
            next_note_beat_idx = min(
                self.dtpr.BDB.get_beat_idx(next_note_beat.id),
                max_prob_beat_idx
            )
        else:
            note_off_beat = self.dtpr.BDB.get_beat(note.off_beat_id)
            next_note_beat_idx = min(
                self.dtpr.BDB.get_beat_idx(note_off_beat.id) + 1,
                max_prob_beat_idx
            )

        prob = np.zeros(self.tpqn*(next_note_beat_idx - note_beat_idx + 1), dtype=np.float64) - 1000.0
        # time配列の設定
        time = np.zeros(prob.shape[0], dtype=np.float64)
        time_abs_tick = np.zeros(prob.shape[0], dtype=int)
        beat_time_array = []
        for i in range(note_beat_idx, next_note_beat_idx+2):
            beat = self.dtpr.BDB.get_beat_from_idx(i)
            beat_time = self.dtpr.BDB.get_beat_time_from_id(beat.id)
            beat_time_array.append(beat_time)

        for i in range(len(time)):
            beat_time = beat_time_array[i//self.tpqn]
            next_beat_time = beat_time_array[(i//self.tpqn) + 1]
            time[i] = beat_time + (i%self.tpqn)*(next_beat_time - beat_time)/self.tpqn
            time_abs_tick[i] = note_beat_idx * TPQN + i * (TPQN // self.tpqn)

        sigma_frame = self.off_cand_gauss_sigma / self.frame_hop_ms   # 100ms -> 10frame
        radius = int(np.ceil(3.0 * sigma_frame))
        T = len(self.pitch_output_prob)

        # 発音時刻のフレームを求める
        note_ptpr = self.dtpr.PTPR.get_note(note_id)
        note_onframe = int(round(note_ptpr.ideal_ontime * 1000.0 / self.frame_hop_ms))
        note_onframe = max(0, min(note_onframe, T - 1))
        if next_note is not None:
            next_note_ptpr = self.dtpr.PTPR.get_note(next_note.id)
            next_note_onframe = int(round(next_note_ptpr.ideal_ontime * 1000.0 / self.frame_hop_ms))
            next_note_onframe = max(0, min(next_note_onframe, T - 1))
        else:
            next_note_onframe = T

        if next_note_onframe <= note_onframe + 1:
            return time_abs_tick, prob

        pitch_cumsum = np.zeros(T + 1, dtype=np.float64)
        rest_cumsum = np.zeros(T + 1, dtype=np.float64)
        pitch_cumsum[1:] = np.cumsum(self.pitch_output_prob[:, pitch])
        rest_cumsum[1:] = np.cumsum(self.pitch_output_prob[:, 128])

        # 確率を計算
        step = int(self.tpqn/self.separate_num)
        for i in range(0, len(prob), step):
            cand_off_frame = time[i] * 1000.0 / self.frame_hop_ms
            # if cand_off_frame <= note_onframe or cand_off_frame > next_note_onframe:
            #     continue
            if time_abs_tick[i] <= note.abs_ontick or time_abs_tick[i] > next_note_ontick:
                continue
            if time_abs_tick[i] == note.abs_offtick:
                continue

            cand_tick = time_abs_tick[i]
            # メトリカルマルコフモデルの計算
            # if next_note is not None:
            #     prob[i] = self.offset_markov_weight * self.offset_metrical_markov[note_onset_met_pos, 0, i%self.tpqn]
            # else:
            #     prob[i] = self.offset_markov_weight * self.offset_metrical_markov[note_onset_met_pos, next_note_met_pos, i%self.tpqn]
            if cand_tick == next_note_ontick:
                prob[i] = self.offset_markov_weight * -0.102629
            else:
                prob[i] = self.offset_markov_weight * (-2.32751 + self.offset_metrical_markov[note_onset_met_pos, next_note_met_pos, i%self.tpqn])


            # real offset の探索範囲を ±3σ に制限
            center_int = int(round(cand_off_frame))
            left = max(note_onframe + 1, center_int - radius)
            right = min(next_note_onframe - 1, center_int + radius)

            if left > right:
                continue

            local_scores = []
            for t in range(left, right + 1):
                pitch_sum = pitch_cumsum[t] - pitch_cumsum[note_onframe]
                rest_sum = rest_cumsum[next_note_onframe] - rest_cumsum[t]
                scale_sum_out_prob = (self.beta * (pitch_sum + rest_sum))/(next_note_onframe-1-note_onframe)
                log_gauss = -0.5 * ((t - cand_off_frame) / sigma_frame) ** 2
                local_scores.append(scale_sum_out_prob + log_gauss)

            prob[i] += logsumexp(np.asarray(local_scores, dtype=np.float64))

        return time_abs_tick, prob

    def calc_split_cand_prob(self, note_id: int):
        ###
        # 入力: 音符nのid
        # 出力: 絶対tick, 分割候補の対数確率
        #
        # 現在の音符の内部に新たな onset を挿入する候補を計算する。
        # 基本形は calc_onset_cand_prob と同様。
        ###
        note = self.dtpr.MTPR.get_note(note_id)
        note_idx = self.dtpr.MTPR.get_note_idx(note_id)

        # split では現在音符の onset を「直前 onset」とみなす
        prev_note_met_pos = (note.on_metrical_pos%TPQN) // (TPQN // self.tpqn)
        note_on_beat = self.dtpr.BDB.get_beat(note.on_beat_id)
        note_on_beat_index = self.dtpr.BDB.get_beat_idx(note_on_beat.id)

        # 次音符があればその onset を右側条件に使う
        if note_idx < len(self.dtpr.MTPR.index) - 1:
            next_note = self.dtpr.MTPR.get_note_from_idx(note_idx + 1)
            next_note_met_pos = (next_note.on_metrical_pos%TPQN) // (TPQN // self.tpqn)
            next_note_beat = self.dtpr.BDB.get_beat(next_note.on_beat_id)
            next_note_beat_index = self.dtpr.BDB.get_beat_idx(next_note_beat.id)
        else:
            next_note = None
            next_note_met_pos = 0
            note_off_beat = self.dtpr.BDB.get_beat(note.off_beat_id)
            next_note_beat_index = min(
                len(self.dtpr.BDB.index) - 1,
                self.dtpr.BDB.get_beat_idx(note_off_beat.id) + 1
            )

        prob = np.zeros(
            self.tpqn * (next_note_beat_index - note_on_beat_index + 1),
            dtype=np.float64
        ) - 1000.0

        # beat time を利用して time 配列を設定
        time = np.zeros(prob.shape[0], dtype=np.float64)
        time_abs_tick = np.zeros(prob.shape[0], dtype=int)
        beat_time_array = []

        for i in range(note_on_beat_index, next_note_beat_index + 2):
            beat = self.dtpr.BDB.get_beat_from_idx(i)
            beat_time = self.dtpr.BDB.get_beat_time_from_id(beat.id)
            beat_time_array.append(beat_time)

        for i in range(len(time)):
            beat_time = beat_time_array[i // self.tpqn]
            next_beat_time = beat_time_array[i // self.tpqn + 1]
            time[i] = beat_time + (i % self.tpqn) * (next_beat_time - beat_time) / self.tpqn
            time_abs_tick[i] = note_on_beat_index * TPQN + i * (TPQN // self.tpqn)

        sigma_frame = self.cand_gauss_sigma / self.frame_hop_ms

        # split 候補は現在音符の内部に限定
        note_on_tick = note.abs_ontick
        note_off_tick = note.abs_offtick

        # 必要なら右側に次音符 onset も課す
        next_note_on = len(prob)
        if next_note is not None:
            next_note_on = (
                (next_note.abs_ontick - note_on_beat_index * TPQN)
                // (TPQN // self.tpqn)
            )

        step = int(self.tpqn / self.separate_num)

        for i in range(0, len(prob), step):
            cand_tick = time_abs_tick[i]

            # 現在音符の内部だけを候補にする
            if cand_tick <= note_on_tick or cand_tick >= note_off_tick:
                continue

            # 念のため次音符 onset より前にも制限
            if i >= next_note_on:
                continue

            # メトリカルマルコフモデル
            if next_note is None:
                prob[i] = self.onset_markov_weight * self.onset_metrical_markov[prev_note_met_pos, i % self.tpqn]
            else:
                prob[i] = self.onset_markov_weight * (
                    self.onset_metrical_markov[prev_note_met_pos, i % self.tpqn]
                    + self.onset_metrical_markov[i % self.tpqn, next_note_met_pos]
                )

            # 出力確率
            cand_frame = time[i] * 1000.0 / self.frame_hop_ms
            prob[i] += self._truncated_log_gauss_score(
                self.onset_output_prob,
                cand_frame,
                sigma_frame
            )

        return time_abs_tick, prob

    def calc_pitch_cand_prob(self, note_id: int):
        ###
        # 入力: 音符nのid
        # 出力:
        #   cand_pitch: 候補ピッチ番号の配列
        #   cand_prob: 各候補ピッチの対数確率
        #
        # 現在の音符区間 [on, off) における各 pitch の平均確率を
        # log-domain で計算し、現在の pitch は候補から除外する
        ###
        if self.pitch_output_prob is None:
            raise ValueError("pitch_output_prob is not set")

        note = self.dtpr.MTPR.get_note(note_id)
        ptpr_note = self.dtpr.PTPR.get_note(note_id)

        T = len(self.pitch_output_prob)
        current_pitch = note.pitch

        onframe = int(round(ptpr_note.ideal_ontime * 1000.0 / self.frame_hop_ms))
        offframe = int(round(ptpr_note.ideal_offtime * 1000.0 / self.frame_hop_ms))

        onframe = max(0, min(onframe, T - 1))
        offframe = max(0, min(offframe, T))

        # 区間が空にならないようにする
        if offframe <= onframe:
            offframe = min(T, onframe + 1)

        seg_len = offframe - onframe
        if seg_len <= 0:
            raise ValueError(f"invalid note segment: onframe={onframe}, offframe={offframe}")

        # 0-127 を音高候補とする（rest=128は除外）
        all_pitch_prob = self.pitch_output_prob[onframe:offframe, :128]  # shape: (seg_len, 128)

        cand_prob = np.sum(all_pitch_prob, axis=0)/(offframe-onframe)

        # 現在の pitch を除外
        valid_mask = np.ones(128, dtype=bool)
        if 0 <= current_pitch < 128:
            valid_mask[current_pitch] = False

        cand_pitch = np.arange(128, dtype=int)[valid_mask]
        cand_prob = cand_prob[valid_mask]

        return cand_pitch, cand_prob

    def top_k(self, time_abs_tick: np.ndarray, prob: np.ndarray, k: int = None):
        ###
        # 入力:
        #   time_abs_tick: 各候補の絶対tick
        #   prob: 各候補の対数確率
        #   k: 返す候補数（省略時は self.n_cand）
        #
        # 出力:
        #   [(rank, abs_tick, log_prob), ...]
        #   確率の高い順に最大k件
        ###
        if k is None:
            k = self.n_cand

        if len(time_abs_tick) != len(prob):
            raise ValueError(
                f"time_abs_tick and prob must have same length, got "
                f"{len(time_abs_tick)} and {len(prob)}"
            )

        if k <= 0:
            return []

        # 無効候補（初期値のままなど）を除外
        # valid_idx = np.where(np.isfinite(prob) & (prob > -100.0))[0]
        valid_idx = np.where(np.isfinite(prob))[0]
        if len(valid_idx) == 0:
            return []

        # # 確率の高い順にソート
        # sorted_idx = valid_idx[np.argsort(prob[valid_idx])[::-1]]

        # # 上位k件
        # top_idx = sorted_idx[:k]

        # results = []
        # for rank, idx in enumerate(top_idx, start=1):
        #     results.append((rank, int(time_abs_tick[idx]), float(prob[idx])))

        # return results
        step = int(self.tpqn / self.separate_num)
        if step <= 0:
            raise ValueError(
                f"invalid step: tpqn={self.tpqn}, separate_num={self.separate_num}"
            )

        grid_mask = np.zeros(len(prob), dtype=bool)
        grid_mask[::step] = True

        # valid_idx = np.where(grid_mask & np.isfinite(prob) & (prob != -100.0))[0]
        valid_idx = np.where(grid_mask & np.isfinite(prob) & (prob > -1000.0))[0]
        if len(valid_idx) == 0:
            return []

        sorted_idx = valid_idx[np.argsort(prob[valid_idx])[::-1]]
        top_idx = sorted_idx[:k]

        results = []
        for rank, idx in enumerate(top_idx, start=1):
            results.append((rank, int(time_abs_tick[idx]), float(prob[idx])))

        return results

    def calc_pitch_cand_prob_with_lang(self, note_id):
        # return
        if self.pitch_output_prob is None:
            raise ValueError("pitch_output_prob is not set")

        if self.rel_pitch_class_uni_prob is None:
            raise ValueError("rel_pitch_class_uni_prob is not set")

        if self.rel_pitch_class_markov is None:
            raise ValueError("rel_pitch_class_markov is not set")
        
        note = self.dtpr.MTPR.get_note(note_id)
        ptpr_note = self.dtpr.PTPR.get_note(note_id)
        note_idx = self.dtpr.MTPR.get_note_idx(note_id)

        T = len(self.pitch_output_prob)
        current_pitch = note.pitch

        onframe = int(round(ptpr_note.ideal_ontime * 1000.0 / self.frame_hop_ms))
        offframe = int(round(ptpr_note.ideal_offtime * 1000.0 / self.frame_hop_ms))

        onframe = max(0, min(onframe, T - 1))
        offframe = max(0, min(offframe, T))

        # 区間が空にならないようにする
        if offframe <= onframe:
            offframe = min(T, onframe + 1)

        seg_len = offframe - onframe
        if seg_len <= 0:
            raise ValueError(f"invalid note segment: onframe={onframe}, offframe={offframe}")

        # 0-127 を音高候補とする（rest=128は除外）
        all_pitch_prob = self.pitch_output_prob[onframe:offframe, :128]  # shape: (seg_len, 128)

        cand_acoustic_prob = np.sum(all_pitch_prob, axis=0)/(offframe-onframe)

        # 言語モデル確率の計算
        cand_lang_prob = np.zeros(128, dtype=np.float64)
        # 範囲指定
        product_range_start = max(0, note_idx - self.product_pitch_range)
        product_range_end = min(note_idx + self.product_pitch_range, len(self.dtpr.MTPR.index) - 1)

        cand_lang_prob_pitch_class = np.zeros(12, dtype=np.float64)
        for pc in range(12):
            lang_prob_tonic = np.zeros(12, dtype=np.float64)
            for tonic in range(12):
        # product部分の計算（対数領域）
                pitch = self.dtpr.MTPR.get_note_from_idx(product_range_start).pitch
                if product_range_start == 0:
                    prev_pitch = -1
                else:
                    prev_pitch = self.dtpr.MTPR.get_note_from_idx(product_range_start - 1).pitch
                for l in range(product_range_start, product_range_end + 1): # indexの確認
                    if l == note_idx:
                        if l == 0:
                            lang_prob_tonic[tonic] += self.rel_pitch_class_uni_prob[(pc - tonic + 1200)%12]
                        else:
                            lang_prob_tonic[tonic] += self.rel_pitch_class_markov[(prev_pitch - tonic + 1200)%12, (pc - tonic + 1200)%12]
                    elif l == note_idx + 1:
                        if l == 0:
                            lang_prob_tonic[tonic] += self.rel_pitch_class_uni_prob[(pitch - tonic + 1200)%12]
                        else:
                            lang_prob_tonic[tonic] += self.rel_pitch_class_markov[(pc - tonic + 1200)%12, (pitch - tonic + 1200)%12]
                    else:
                        if l == 0:
                            lang_prob_tonic[tonic] += self.rel_pitch_class_uni_prob[(pitch - tonic + 1200)%12]
                        else:
                            lang_prob_tonic[tonic] += self.rel_pitch_class_markov[(prev_pitch - tonic + 1200)%12, (pitch - tonic + 1200)%12]

                    if l < product_range_end:
                        prev_pitch = pitch
                        pitch = self.dtpr.MTPR.get_note_from_idx(l + 1).pitch

            # log_sum_exp + 1/12
            # log_prob_temp = 0
            # for i in range(len(lang_prob_tonic)):
            #     log_prob_temp += np.exp(lang_prob_tonic[i] - np.log(12))
            log_prob = logsumexp(lang_prob_tonic) - np.log(12.0)

            cand_lang_prob_pitch_class[pc] = log_prob

        # cand_lang_prob[:] = np.tile(cand_lang_prob_pitch_class, 11)[:128]
        cand_lang_prob = cand_lang_prob_pitch_class[np.arange(128) % 12]

        # cand_lang_prob /= (product_range_end - product_range_start + 1)

        # 音響確率と言語モデル確率を足し合わせる
        cand_prob = cand_acoustic_prob + self.rel_pitch_class_markov_weight * cand_lang_prob

        # cand_prob = self.rel_pitch_class_markov_weight * cand_lang_prob

        # 現在の pitch を除外
        valid_mask = np.ones(128, dtype=bool)
        if 0 <= current_pitch < 128:
            valid_mask[current_pitch] = False

        cand_pitch = np.arange(128, dtype=int)[valid_mask]
        cand_prob = cand_prob[valid_mask]

        return cand_pitch, cand_prob

    def top_k_pitch(self, cand_pitch: np.ndarray, cand_prob: np.ndarray, k: int = None):
        ###
        # 入力:
        #   cand_pitch: 候補ピッチ
        #   cand_prob: 各候補の対数確率
        # 出力:
        #   [(rank, pitch, log_prob), ...]
        ###
        if k is None:
            k = self.n_cand

        if len(cand_pitch) != len(cand_prob):
            raise ValueError(
                f"cand_pitch and cand_prob must have same length, got "
                f"{len(cand_pitch)} and {len(cand_prob)}"
            )

        if k <= 0:
            return []

        valid_idx = np.where(np.isfinite(cand_prob))[0]
        if len(valid_idx) == 0:
            return []

        sorted_idx = valid_idx[np.argsort(cand_prob[valid_idx])[::-1]]
        top_idx = sorted_idx[:k]

        return [
            (rank, int(cand_pitch[idx]), float(cand_prob[idx]))
            for rank, idx in enumerate(top_idx, start=1)
        ]

    def calc_onset_cand_prob_analysis(self, note_id: int):
        ###
        # 分析用:
        #   次音符との重なり制約を無視し、
        #   現在の推定 onset から前後半ビートの範囲「のみ」で onset 候補を計算する
        #
        # 出力:
        #   time_abs_tick, prob
        ###
        note = self.dtpr.MTPR.get_note(note_id)
        note_idx = self.dtpr.MTPR.get_note_idx(note_id)

        if note_idx > 0:
            prev_note = self.dtpr.MTPR.get_note_from_idx(note_idx - 1)
            prev_note_met_pos = (prev_note.on_metrical_pos % TPQN) // (TPQN // self.tpqn)
        else:
            prev_note = None
            prev_note_met_pos = 0

        if note_idx < len(self.dtpr.MTPR.index) - 1:
            next_note = self.dtpr.MTPR.get_note_from_idx(note_idx + 1)
            next_note_met_pos = (next_note.on_metrical_pos % TPQN) // (TPQN // self.tpqn)
        else:
            next_note = None
            next_note_met_pos = 0

        center_tick = note.abs_ontick
        half_beat_tick = TPQN // 2
        tick_step = TPQN // self.tpqn

        start_tick = center_tick - half_beat_tick
        end_tick = center_tick + half_beat_tick

        # 前後半ビート区間「のみ」の格子を直接生成
        time_abs_tick = np.arange(start_tick, end_tick + 1, tick_step, dtype=int)
        prob = np.zeros(len(time_abs_tick), dtype=np.float64) - 1000.0
        time = np.zeros(len(time_abs_tick), dtype=np.float64)

        sigma_frame = self.cand_gauss_sigma / self.frame_hop_ms
        step = int(self.tpqn / self.separate_num)

        candidate_met_pos = np.full(len(time_abs_tick), -1, dtype=int)

        for i, cand_tick in enumerate(time_abs_tick):
            if cand_tick == note.abs_ontick:
                continue

            # separate_num に応じた格子だけ評価
            rel_idx_24 = ((cand_tick % TPQN) // tick_step)
            if rel_idx_24 % step != 0:
                continue

            cand_time = self.dtpr.abs_tick_to_time(cand_tick) if hasattr(self.dtpr, "abs_tick_to_time") else None
            if cand_time is None:
                # beat 時刻から線形補間
                beat_idx = cand_tick // TPQN
                if beat_idx < 0 or beat_idx + 1 >= len(self.dtpr.BDB.index):
                    continue
                beat = self.dtpr.BDB.get_beat_from_idx(beat_idx)
                next_beat = self.dtpr.BDB.get_beat_from_idx(beat_idx + 1)
                beat_time = self.dtpr.BDB.get_beat_time_from_id(beat.id)
                next_beat_time = self.dtpr.BDB.get_beat_time_from_id(next_beat.id)
                local_tick = cand_tick - beat_idx * TPQN
                cand_time = beat_time + local_tick * (next_beat_time - beat_time) / TPQN
                rel_idx_24 = ((cand_tick % TPQN) // tick_step)

            time[i] = cand_time

            candidate_met_pos[i] = rel_idx_24
            met_pos = rel_idx_24

            if prev_note is None:
                prob[i] = self.onset_markov_weight * self.onset_uni_prob[met_pos]
            elif next_note is None:
                prob[i] = self.onset_markov_weight * self.onset_metrical_markov[prev_note_met_pos, met_pos]
            else:
                prob[i] = self.onset_markov_weight * (self.onset_metrical_markov[prev_note_met_pos, met_pos] + self.onset_metrical_markov[met_pos, next_note_met_pos])

            cand_frame = cand_time * 1000.0 / self.frame_hop_ms
            prob[i] += self._truncated_log_gauss_score(self.onset_output_prob, cand_frame, sigma_frame)

        context_info = {
            "prev_note_met_pos": prev_note_met_pos,
            "next_note_met_pos": next_note_met_pos,
            "candidate_met_pos": candidate_met_pos,
        }

        return time_abs_tick, prob, context_info
    
    # def calc_onset_cand_prob_analysis_uni(self, note_id: int):
        ###
        # 分析用:
        #   次音符との重なり制約を無視し、
        #   現在の推定 onset から前後半ビートの範囲「のみ」で onset 候補を計算する
        #
        # 出力:
        #   time_abs_tick, prob
        ###
        # note = self.dtpr.MTPR.get_note(note_id)
        # note_idx = self.dtpr.MTPR.get_note_idx(note_id)

        # if note_idx > 0:
        #     prev_note = self.dtpr.MTPR.get_note_from_idx(note_idx - 1)
        #     prev_note_met_pos = (prev_note.on_metrical_pos % TPQN) // (TPQN // self.tpqn)
        # else:
        #     prev_note = None
        #     prev_note_met_pos = 0

        # if note_idx < len(self.dtpr.MTPR.index) - 1:
        #     next_note = self.dtpr.MTPR.get_note_from_idx(note_idx + 1)
        #     next_note_met_pos = (next_note.on_metrical_pos % TPQN) // (TPQN // self.tpqn)
        # else:
        #     next_note = None
        #     next_note_met_pos = 0

        # center_tick = note.abs_ontick
        # half_beat_tick = TPQN // 2
        # tick_step = TPQN // self.tpqn

        # start_tick = center_tick - half_beat_tick
        # end_tick = center_tick + half_beat_tick

        # # 前後半ビート区間「のみ」の格子を直接生成
        # time_abs_tick = np.arange(start_tick, end_tick + 1, tick_step, dtype=int)
        # prob = np.zeros(len(time_abs_tick), dtype=np.float64) - 1000.0
        # time = np.zeros(len(time_abs_tick), dtype=np.float64)

        # sigma_frame = self.cand_gauss_sigma / self.frame_hop_ms
        # step = int(self.tpqn / self.separate_num)

        # candidate_met_pos = np.full(len(time_abs_tick), -1, dtype=int)

        # for i, cand_tick in enumerate(time_abs_tick):
        #     if cand_tick == note.abs_ontick:
        #         continue

        #     # separate_num に応じた格子だけ評価
        #     rel_idx_24 = ((cand_tick % TPQN) // tick_step)
        #     if rel_idx_24 % step != 0:
        #         continue

        #     cand_time = self.dtpr.abs_tick_to_time(cand_tick) if hasattr(self.dtpr, "abs_tick_to_time") else None
        #     if cand_time is None:
        #         # beat 時刻から線形補間
        #         beat_idx = cand_tick // TPQN
        #         if beat_idx < 0 or beat_idx + 1 >= len(self.dtpr.BDB.index):
        #             continue
        #         beat = self.dtpr.BDB.get_beat_from_idx(beat_idx)
        #         next_beat = self.dtpr.BDB.get_beat_from_idx(beat_idx + 1)
        #         beat_time = self.dtpr.BDB.get_beat_time_from_id(beat.id)
        #         next_beat_time = self.dtpr.BDB.get_beat_time_from_id(next_beat.id)
        #         local_tick = cand_tick - beat_idx * TPQN
        #         cand_time = beat_time + local_tick * (next_beat_time - beat_time) / TPQN
        #         rel_idx_24 = ((cand_tick % TPQN) // tick_step)

        #     time[i] = cand_time

        #     candidate_met_pos[i] = rel_idx_24
        #     met_pos = rel_idx_24

        #     # ユニグラムでの検証
        #     prob[i] = self.onset_markov_weight * self.onset_uni_prob[met_pos]

        #     # 言語モデル確率確認のためコメントアウト
        #     cand_frame = cand_time * 1000.0 / self.frame_hop_ms
        #     prob[i] += self._truncated_log_gauss_score(self.onset_output_prob, cand_frame, sigma_frame)

        # context_info = {
        #     "prev_note_met_pos": prev_note_met_pos,
        #     "next_note_met_pos": next_note_met_pos,
        #     "candidate_met_pos": candidate_met_pos,
        # }

        # return time_abs_tick, prob, context_info

    def calc_offset_cand_prob_analysis(self, note_id: int):
        ###
        # 分析用:
        #   次音符との重なり制約を無視し、
        #   現在の推定 offset から前後1ビートの範囲「のみ」で offset 候補を計算する
        #
        # 出力:
        #   time_abs_tick, prob
        ###
        note = self.dtpr.MTPR.get_note(note_id)
        note_idx = self.dtpr.MTPR.get_note_idx(note_id)
        pitch = note.pitch
        note_onset_met_pos = (note.on_metrical_pos % TPQN) // (TPQN // self.tpqn)

        if note_idx < len(self.dtpr.MTPR.index) - 1:
            next_note = self.dtpr.MTPR.get_note_from_idx(note_idx + 1)
            next_note_met_pos = (next_note.on_metrical_pos % TPQN) // (TPQN // self.tpqn)
            next_abs_ontick = next_note.abs_ontick
        else:
            next_note = None
            next_note_met_pos = 0
            next_abs_ontick = note.abs_offtick - note.abs_offtick%TPQN + TPQN

        center_tick = note.abs_offtick
        one_beat_tick = TPQN
        tick_step = TPQN // self.tpqn

        start_tick = center_tick - one_beat_tick
        end_tick = center_tick + one_beat_tick

        # 前後1ビート区間「のみ」の格子を直接生成
        time_abs_tick = np.arange(start_tick, end_tick + 1, tick_step, dtype=int)
        prob = np.zeros(len(time_abs_tick), dtype=np.float64) - 1000.0
        time = np.zeros(len(time_abs_tick), dtype=np.float64)

        sigma_frame = self.off_cand_gauss_sigma / self.frame_hop_ms
        radius = int(np.ceil(3.0 * sigma_frame))
        T = len(self.pitch_output_prob)

        note_ptpr = self.dtpr.PTPR.get_note(note_id)
        note_onframe = int(round(note_ptpr.ideal_ontime * 1000.0 / self.frame_hop_ms))
        note_onframe = max(0, min(note_onframe, T - 1))

        analysis_end_frame = T

        if next_note is not None:
            next_note_ptpr = self.dtpr.PTPR.get_note(next_note.id)
            next_note_onframe = int(round(next_note_ptpr.ideal_ontime * 1000.0 / self.frame_hop_ms))
            next_note_onframe = max(0, min(next_note_onframe, T - 1))
        else:
            next_note_onframe = T

        pitch_cumsum = np.zeros(T + 1, dtype=np.float64)
        rest_cumsum = np.zeros(T + 1, dtype=np.float64)
        pitch_cumsum[1:] = np.cumsum(self.pitch_output_prob[:, pitch])
        rest_cumsum[1:] = np.cumsum(self.pitch_output_prob[:, 128])

        step = int(self.tpqn / self.separate_num)

        candidate_met_pos = np.full(len(time_abs_tick), -1, dtype=int)

        for i, cand_tick in enumerate(time_abs_tick):
            if cand_tick == note.abs_offtick:
                continue

            rel_idx_24 = ((cand_tick % TPQN) // tick_step)
            if rel_idx_24 % step != 0:
                continue

            cand_time = self.dtpr.abs_tick_to_time(cand_tick) if hasattr(self.dtpr, "abs_tick_to_time") else None
            if cand_time is None:
                beat_idx = cand_tick // TPQN
                if beat_idx < 0 or beat_idx + 1 >= len(self.dtpr.BDB.index):
                    continue
                beat = self.dtpr.BDB.get_beat_from_idx(beat_idx)
                next_beat = self.dtpr.BDB.get_beat_from_idx(beat_idx + 1)
                beat_time = self.dtpr.BDB.get_beat_time_from_id(beat.id)
                next_beat_time = self.dtpr.BDB.get_beat_time_from_id(next_beat.id)
                local_tick = cand_tick - beat_idx * TPQN
                cand_time = beat_time + local_tick * (next_beat_time - beat_time) / TPQN

            time[i] = cand_time
            cand_off_frame = cand_time * 1000.0 / self.frame_hop_ms

            candidate_met_pos[i] = rel_idx_24

            # if next_note is not None:
            #     prob[i] = self.offset_markov_weight * self.offset_metrical_markov[note_onset_met_pos, 0, rel_idx_24]
            # else:
            #     prob[i] = self.offset_markov_weight * self.offset_metrical_markov[note_onset_met_pos, next_note_met_pos, rel_idx_24]

            if cand_tick == next_abs_ontick:
                prob[i] = self.offset_markov_weight * -0.102629
            else:
                prob[i] = self.offset_markov_weight * (-2.32751 + self.offset_metrical_markov[note_onset_met_pos, next_note_met_pos, rel_idx_24])

            center_int = int(round(cand_off_frame))
            left = max(0, center_int - radius)
            right = min(analysis_end_frame - 1, center_int + radius)

            if left > right:
                continue

            local_scores = []
            for t in range(left, right + 1):
                pitch_sum = pitch_cumsum[t] - pitch_cumsum[note_onframe]
                rest_sum = rest_cumsum[next_note_onframe] - rest_cumsum[t]
                scale_sum_out_prob = (self.beta * (pitch_sum + rest_sum))/(next_note_onframe-1-note_onframe)
                log_gauss = -0.5 * ((t - cand_off_frame) / sigma_frame) ** 2
                local_scores.append(scale_sum_out_prob + log_gauss)

            prob[i] += logsumexp(np.asarray(local_scores, dtype=np.float64))
        context_info = {
            "note_onset_met_pos": note_onset_met_pos,
            "next_note_met_pos": next_note_met_pos,
            "candidate_met_pos": candidate_met_pos,
        }

        return time_abs_tick, prob, context_info
    
    # def calc_offset_cand_prob_analysis_uni(self, note_id: int):
        ###
        # 分析用:
        #   次音符との重なり制約を無視し、
        #   現在の推定 offset から前後1ビートの範囲「のみ」で offset 候補を計算する
        #
        # 出力:
        #   time_abs_tick, prob
        ###
        # note = self.dtpr.MTPR.get_note(note_id)
        # note_idx = self.dtpr.MTPR.get_note_idx(note_id)
        # pitch = note.pitch
        # note_onset_met_pos = (note.on_metrical_pos % TPQN) // (TPQN // self.tpqn)

        # if note_idx < len(self.dtpr.MTPR.index) - 1:
        #     next_note = self.dtpr.MTPR.get_note_from_idx(note_idx + 1)
        #     next_note_met_pos = (next_note.on_metrical_pos % TPQN) // (TPQN // self.tpqn)
        # else:
        #     next_note = None
        #     next_note_met_pos = 0

        # center_tick = note.abs_offtick
        # one_beat_tick = TPQN
        # tick_step = TPQN // self.tpqn

        # start_tick = center_tick - one_beat_tick
        # end_tick = center_tick + one_beat_tick

        # # 前後1ビート区間「のみ」の格子を直接生成
        # time_abs_tick = np.arange(start_tick, end_tick + 1, tick_step, dtype=int)
        # prob = np.zeros(len(time_abs_tick), dtype=np.float64) - 1000.0
        # time = np.zeros(len(time_abs_tick), dtype=np.float64)

        # sigma_frame = self.off_cand_gauss_sigma / self.frame_hop_ms
        # radius = int(np.ceil(3.0 * sigma_frame))
        # T = len(self.pitch_output_prob)

        # note_ptpr = self.dtpr.PTPR.get_note(note_id)
        # note_onframe = int(round(note_ptpr.ideal_ontime * 1000.0 / self.frame_hop_ms))
        # note_onframe = max(0, min(note_onframe, T - 1))

        # analysis_end_frame = T

        # if next_note is not None:
        #     next_note_ptpr = self.dtpr.PTPR.get_note(next_note.id)
        #     next_note_onframe = int(round(next_note_ptpr.ideal_ontime * 1000.0 / self.frame_hop_ms))
        #     next_note_onframe = max(0, min(next_note_onframe, T - 1))
        # else:
        #     next_note_onframe = T

        # pitch_cumsum = np.zeros(T + 1, dtype=np.float64)
        # rest_cumsum = np.zeros(T + 1, dtype=np.float64)
        # pitch_cumsum[1:] = np.cumsum(self.pitch_output_prob[:, pitch])
        # rest_cumsum[1:] = np.cumsum(self.pitch_output_prob[:, 128])

        # step = int(self.tpqn / self.separate_num)

        # candidate_met_pos = np.full(len(time_abs_tick), -1, dtype=int)

        # for i, cand_tick in enumerate(time_abs_tick):
        #     if cand_tick == note.abs_offtick:
        #         continue

        #     rel_idx_24 = ((cand_tick % TPQN) // tick_step)
        #     if rel_idx_24 % step != 0:
        #         continue

        #     cand_time = self.dtpr.abs_tick_to_time(cand_tick) if hasattr(self.dtpr, "abs_tick_to_time") else None
        #     if cand_time is None:
        #         beat_idx = cand_tick // TPQN
        #         if beat_idx < 0 or beat_idx + 1 >= len(self.dtpr.BDB.index):
        #             continue
        #         beat = self.dtpr.BDB.get_beat_from_idx(beat_idx)
        #         next_beat = self.dtpr.BDB.get_beat_from_idx(beat_idx + 1)
        #         beat_time = self.dtpr.BDB.get_beat_time_from_id(beat.id)
        #         next_beat_time = self.dtpr.BDB.get_beat_time_from_id(next_beat.id)
        #         local_tick = cand_tick - beat_idx * TPQN
        #         cand_time = beat_time + local_tick * (next_beat_time - beat_time) / TPQN

        #     time[i] = cand_time
        #     cand_off_frame = cand_time * 1000.0 / self.frame_hop_ms

        #     candidate_met_pos[i] = rel_idx_24

        #     prob[i] = self.offset_markov_weight * self.offset_uni_prob[rel_idx_24]

        #     center_int = int(round(cand_off_frame))
        #     left = max(0, center_int - radius)
        #     right = min(analysis_end_frame - 1, center_int + radius)

        #     if left > right:
        #         continue

        #     local_scores = []
        #     for t in range(left, right + 1):
        #         pitch_sum = pitch_cumsum[t] - pitch_cumsum[note_onframe]
        #         rest_sum = rest_cumsum[next_note_onframe] - rest_cumsum[t]
        #         scale_sum_out_prob = (self.beta * (pitch_sum + rest_sum))/(next_note_onframe-1-note_onframe)
        #         log_gauss = -0.5 * ((t - cand_off_frame) / sigma_frame) ** 2
        #         local_scores.append(scale_sum_out_prob + log_gauss)

        #     prob[i] += logsumexp(np.asarray(local_scores, dtype=np.float64))
        # context_info = {
        #     "note_onset_met_pos": note_onset_met_pos,
        #     "next_note_met_pos": next_note_met_pos,
        #     "candidate_met_pos": candidate_met_pos,
        # }

        # return time_abs_tick, prob, context_info

    def calc_join_cand_prob_analysis(self, note_id: int, ref_on_tick: int, delta: int):
        ###
        # 分析用 join 候補:
        #   正解音符 A の onset 時刻 ref_on_tick と、
        #   その時刻で active な推定音符 A' (= note_id) に対して、
        #   候補範囲を
        #
        #   left  = max(ref_on_tick - TPQN//2, A'_on + delta)
        #   right = min(ref_on_tick + TPQN//2, A'_off - delta)
        #
        #   の閉区間 [left, right] に限定して split 候補を計算する
        #
        # 入力:
        #   note_id     : active な推定音符 A' の id
        #   ref_on_tick : 正解音符 A の onset tick (absolute tick)
        #   delta       : 境界から除外する tick 幅
        #
        # 出力:
        #   time_abs_tick, prob
        ###
        note = self.dtpr.MTPR.get_note(note_id)
        note_idx = self.dtpr.MTPR.get_note_idx(note_id)

        prev_note_met_pos = (note.on_metrical_pos % TPQN) // (TPQN // self.tpqn)

        if note_idx < len(self.dtpr.MTPR.index) - 1:
            next_note = self.dtpr.MTPR.get_note_from_idx(note_idx + 1)
            next_note_met_pos = (next_note.on_metrical_pos % TPQN) // (TPQN // self.tpqn)
        else:
            next_note = None
            next_note_met_pos = 0

        note_on_tick = note.abs_ontick
        note_off_tick = note.abs_offtick

        half_beat_tick = TPQN // 2
        tick_step = TPQN // self.tpqn
        step = int(self.tpqn / self.separate_num)

        left = max(ref_on_tick - half_beat_tick, note_on_tick + delta)
        right = min(ref_on_tick + half_beat_tick, note_off_tick - delta)

        # 候補が存在しない場合
        if left > right:
            return np.array([], dtype=int), np.array([], dtype=np.float64)

        # 閉区間 [left, right] の格子を直接生成
        time_abs_tick = np.arange(left, right + 1, tick_step, dtype=int)
        prob = np.zeros(len(time_abs_tick), dtype=np.float64) - 1000.0

        sigma_frame = self.cand_gauss_sigma / self.frame_hop_ms

        candidate_met_pos = np.full(len(time_abs_tick), -1, dtype=int)

        for i, cand_tick in enumerate(time_abs_tick):
            rel_idx_24 = ((cand_tick % TPQN) // tick_step)

            # separate_num に応じた格子だけ評価
            if rel_idx_24 % step != 0:
                continue

            cand_time = self.dtpr.abs_tick_to_time(cand_tick)
            prob_i = None
            candidate_met_pos[i] = rel_idx_24

            if next_note is None:
                prob_i = self.onset_markov_weight * self.onset_metrical_markov[prev_note_met_pos, rel_idx_24]
            else:
                prob_i = self.onset_markov_weight * (
                    self.onset_metrical_markov[prev_note_met_pos, rel_idx_24]
                    + self.onset_metrical_markov[rel_idx_24, next_note_met_pos]
                )

            # 言語モデル確率確認のためコメントアウト
            cand_frame = cand_time * 1000.0 / self.frame_hop_ms
            prob_i += self._truncated_log_gauss_score(
                self.onset_output_prob,
                cand_frame,
                sigma_frame
            )

            prob[i] = prob_i

        context_info = {
            "source_note_onset_met_pos": prev_note_met_pos,
            "next_note_met_pos": next_note_met_pos,
            "candidate_met_pos": candidate_met_pos,
        }

        return time_abs_tick, prob, context_info
    
