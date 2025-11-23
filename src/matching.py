from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2

from .models import TrackSummary, CameraConfig, SceneConfig
from .utils_geom import gaussian_similarity


# ========= ユーティリティ =========


def _intervals_disjoint(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """閉区間 [a_start, a_end] と [b_start, b_end] が重ならないなら True。"""
    return (a_end < b_start) or (b_end < a_start)


def _get_time_window_for_entry(src_cam: CameraConfig, neighbor_cam_name: str) -> Optional[Tuple[int, int]]:
    """
    src_cam.candidate_time_window から neighbor_cam_name の [min,max] (frames) を取得。
    例: camB の 'camA': [30,450] は「camA の終了フレームは camB の start_frame の 30〜450 フレ前」。
    """
    if not src_cam.candidate_time_window:
        return None
    win = src_cam.candidate_time_window.get(neighbor_cam_name)
    if not win:
        return None
    lo, hi = int(win[0]), int(win[1])
    if lo < 0 or hi < 0 or lo > hi:
        return None
    return lo, hi


def _tgt_end_in_entry_window(src: TrackSummary, tgt: TrackSummary, src_cfg: CameraConfig, neighbor_cam_name: str) -> bool:
    """
    時間窓チェック:
      tgt.end_frame ∈ [ src.start_frame - hi , src.start_frame - lo ] であること。
    - neighbor_cam_name: 「どのカメラから来た」と仮定しているか（entry_dir_cam の 1 要素）
    """
    win = _get_time_window_for_entry(src_cfg, neighbor_cam_name)
    if win is None:
        return False
    lo, hi = win
    min_end = src.start_frame - hi
    max_end = src.start_frame - lo
    return (min_end <= tgt.end_frame <= max_end)


# ========= 候補列挙の中核 =========


def find_candidates_for_track(src: TrackSummary, scene: SceneConfig, summaries_by_cam: Dict[str, List[TrackSummary]]) -> List[TrackSummary]:
    """
    「候補の最終絞り込み」までを行う。

    1) src.entry_dir_cam に入っている **全てのカメラ** を「来たカメラ候補」として使う
       - 例: entry_dir_cam = ["camA", "camB"] なら camA, camB のトラックを両方見る
    2) その中から、tgt.exit_dir_cam に src.cam が含まれるものだけ残す
       （= tgt は実際に src.cam 方向へ消えている）
    3) 時間窓: tgt.end_frame が src.cam の candidate_time_window[neighbor] 内に入る
       （neighbor は「そこから来た」と仮定しているカメラ）
    4) フレーム重複なし（= [tgt.start,tgt.end] と [src.start,src.end] が非重複）

    -> 条件を満たした TrackSummary の配列を返す
    """
    src_cfg = scene.cameras.get(src.cam)
    if src_cfg is None:
        return []

    # 1) entry_dir_cam が空リストなら候補なし
    neighbor_names = src.entry_dir_cams or []
    if not neighbor_names:
        return []

    out: List[TrackSummary] = []
    seen: set[Tuple[str, int]] = set()  # (cam, track_id) 重複防止

    for neighbor_name in neighbor_names:
        tgts = summaries_by_cam.get(neighbor_name, [])
        if not tgts:
            continue

        for tgt in tgts:
            key = (tgt.cam, tgt.track_id)
            if key in seen:
                continue

            # 2) tgt は実際に src.cam 側へ消えている必要（exit_dir_cam に src.cam が含まれる）
            if src.cam not in (tgt.exit_dir_cams or []):
                continue

            # 3) 時間窓チェック（src.cam の設定＋「neighbor_name から来た」という仮定）
            if not _tgt_end_in_entry_window(src, tgt, src_cfg, neighbor_name):
                continue

            # 4) フレーム重複なし（同一人物が同時刻に2カメで見えるのを除外）
            if not _intervals_disjoint(src.start_frame, src.end_frame, tgt.start_frame, tgt.end_frame):
                continue

            out.append(tgt)
            seen.add(key)

    return out


# ========= 全カメラ・全トラックを一括処理 =========


@dataclass(slots=True)
class CandidateList:
    """可視化・後続処理のための軽いラッパー。"""
    cam: str
    track_id: int
    neighbors: List[str]                   # entry_dir_cam (複数候補)
    candidates: List[Tuple[str, int]]      # (neighbor_cam, target_track_id)


def build_all_candidates(scene: SceneConfig, summaries_by_cam: Dict[str, List[TrackSummary]]) -> List[CandidateList]:
    """
    全カメラ・全トラックについて、上の find_candidates_for_track を適用して列挙。
    返り値は (cam, track_id, entry_dir_cam(リスト), [(neighbor, target_track_id)...]) の配列。
    """
    results: List[CandidateList] = []
    for cam_name, summaries in summaries_by_cam.items():
        for src in summaries:
            cands = find_candidates_for_track(src, scene, summaries_by_cam)
            tup_list = [(tgt.cam, tgt.track_id) for tgt in cands]
            results.append(CandidateList(
                cam=src.cam,
                track_id=src.track_id,
                neighbors=list(src.entry_dir_cams or []),
                candidates=tup_list,
            ))
    return results


# ========= マッチスコア =========

@dataclass(slots=True)
class MatchScore:
    src_cam: str
    src_id: int
    src_start_frame: Optional[int]             # src の開始フレーム
    tgt_cam: str
    tgt_id: int
    tgt_end_frame: Optional[int]               # tgt の終了フレーム
    tgt_speed_mps: Optional[float]             # tgt の速度[m/s]
    predicted_appearance_frame: Optional[int]  # 計算不可時は None
    delta_frames: Optional[int]                # (pred - src.start_frame)
    position_score: Optional[float]            # 位置(時刻)のガウス類似度
    appearance_score: Optional[float]          # HSヒスト類似度
    fusion_score: Optional[float]              # 位置×外観の融合スコア


# HS 類似度用の重み（相関 / Bhattacharyya）
W_CORR = 0.5
W_BHAT = 0.5


def _compute_hs_similarity(src_hist: Optional[np.ndarray], tgt_hist: Optional[np.ndarray]) -> Optional[float]:
    """
    HS ヒスト同士の類似度を計算。
    - 相関（-1〜1）→ [0,1]
    - Bhattacharyya 距離（0〜1）→ 類似度（1-距離）
    - 上記を W_CORR / W_BHAT で加重平均
    どちらかが None / 次元不一致なら None。
    """
    if src_hist is None or tgt_hist is None:
        return None
    if not isinstance(src_hist, np.ndarray) or not isinstance(tgt_hist, np.ndarray):
        return None
    if src_hist.shape != tgt_hist.shape:
        return None

    # float32 に揃える
    a = src_hist.astype(np.float32, copy=False)
    b = tgt_hist.astype(np.float32, copy=False)

    # 相関
    corr = cv2.compareHist(a, b, cv2.HISTCMP_CORREL)  # -1〜1
    corr01 = (corr + 1.0) * 0.5                      # [-1,1] → [0,1]

    # Bhattacharyya
    bhat_dist = cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA)
    bhat_dist = max(0.0, min(1.0, float(bhat_dist)))  # 0〜1 にクランプ
    bhat_sim = 1.0 - bhat_dist                        # 0(不一致)〜1(一致)

    score = W_CORR * corr01 + W_BHAT * bhat_sim
    return float(score)


def predict_arrival_and_score_for_src(src: TrackSummary, scene: SceneConfig, summaries_by_cam: Dict[str, List[TrackSummary]]) -> List[MatchScore]:
    """
    1) src に対する最終候補（find_candidates_for_track）を取得
    2) tgt.end_frame, tgt.speed_mps, src_cam.distance_to[tgt.cam], src_cam.fps から
       predicted_appearance_frame を計算
         predicted = tgt.end_frame + round((distance / speed_mps) * fps)
    3) ΔF = predicted - src.start_frame
    4) 位置スコア position_score = exp(-(ΔF^2)/(2*sigma^2))
    5) HSヒストから appearance_score を計算
    6) scene.fusion_w_pos / fusion_w_app で融合スコア fusion_score を計算
    """
    src_cfg = scene.cameras.get(src.cam)
    if src_cfg is None:
        return []

    # 候補列挙（入口で絞り込み済み）
    tgts = find_candidates_for_track(src, scene, summaries_by_cam)
    if not tgts:
        return []

    fps = float(src_cfg.fps)
    sigma = float(src_cfg.sigma_frames if src_cfg.sigma_frames is not None else scene.default_sigma_frames)

    # --- ベクトル化準備（位置スコア用） ---
    end_frames = np.array([t.end_frame for t in tgts], dtype=np.float64)

    distances = np.array(
        [float(src_cfg.distance_to.get(t.cam, np.nan)) for t in tgts],
        dtype=np.float64,
    )
    speeds = np.array(
        [float(t.speed_mps) if (t.speed_mps is not None and t.speed_mps > 0.0) else np.nan for t in tgts],
        dtype=np.float64,
    )

    move_time_sec = distances / speeds           # [s]
    move_frames = np.rint(move_time_sec * fps)   # 四捨五入（int化前段階）
    predicted = end_frames + move_frames         # まだ float

    delta = predicted - float(src.start_frame)
    valid = np.isfinite(delta)

    pos_scores = np.full_like(delta, np.nan, dtype=np.float64)
    pos_scores[valid] = gaussian_similarity(delta[valid], sigma)

    # --- HS 類似度（Appearance） ---
    app_scores: List[Optional[float]] = []
    for tgt in tgts:
        app_scores.append(_compute_hs_similarity(src.hs_hist_avg, tgt.hs_hist_avg))

    # --- 融合スコア ---
    fw_pos = float(scene.fusion_w_pos)
    fw_app = float(scene.fusion_w_app)

    out: List[MatchScore] = []
    for t, pred, d, ps, app, v in zip(tgts, predicted, delta, pos_scores, app_scores, valid):
        # 融合ロジック
        pos_val = float(ps) if (v and np.isfinite(ps)) else None
        app_val = app if (app is not None and np.isfinite(app)) else None

        if (pos_val is not None) and (app_val is not None):
            fusion = fw_pos * pos_val + fw_app * app_val
        elif pos_val is not None:
            fusion = pos_val
        elif app_val is not None:
            fusion = app_val
        else:
            fusion = None

        out.append(MatchScore(
            src_cam=src.cam,
            src_id=src.track_id,
            src_start_frame=src.start_frame,
            tgt_cam=t.cam,
            tgt_id=t.track_id,
            tgt_end_frame=t.end_frame,
            tgt_speed_mps=t.speed_mps,
            predicted_appearance_frame=int(pred) if v else None,
            delta_frames=int(d) if v else None,
            position_score=pos_val,
            appearance_score=app_val,
            fusion_score=fusion,
        ))
    return out


def compute_all_match_scores(scene: SceneConfig, summaries_by_cam: Dict[str, List[TrackSummary]], sort_desc: bool = True) -> Dict[Tuple[str, int], List[MatchScore]]:
    """
    各 (src_cam, src_id) ごとに候補スコアのリストを返す。
    - sort_desc=True の場合、fusion_score 降順（None は最後）で並べ替え。
    """
    results: Dict[Tuple[str, int], List[MatchScore]] = {}
    for cam_name, summaries in summaries_by_cam.items():
        for src in summaries:
            scores = predict_arrival_and_score_for_src(src, scene, summaries_by_cam)
            if sort_desc:
                scores = sorted(
                    scores,
                    key=lambda r: (
                        -1.0 if (r.fusion_score is None) else -r.fusion_score,  # None は末尾
                        abs(r.delta_frames) if r.delta_frames is not None else 9e9,
                    )
                )
            results[(src.cam, src.track_id)] = scores
    return results
