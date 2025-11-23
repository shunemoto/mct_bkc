from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import math
import numpy as np

from .utils_geom import apply_homography_xy, normalize_signed_deg, compute_direction
from .models import TrackPoint, TrackSummary, CameraConfig
from .filtering import filter_tracks_by_rois


def _estimate_tail_speed_mps(seg: List[TrackPoint], cam: CameraConfig, seconds: float = 2.0,) -> Optional[float]:
    """
    追跡終了直前（末尾）の速度 [m/s] を推定。
    - 対象区間: 最後のフレーム i2 と、その「seconds 秒前」に最も近いフレーム i1 の間
    - H（ホモグラフィ）が無ければ None
    - 区間の実時間 dt が 0 以下なら None
    """
    H = cam.homography
    if H is None or cam.fps <= 0 or seconds <= 0.0 or len(seg) < 2:
        return None

    i2 = len(seg) - 1
    frames_back = max(60, int(round(seconds * cam.fps)))
    i1 = max(0, i2 - frames_back)

    start_xy_img = seg[i1].center_bottom
    end_xy_img   = seg[i2].center_bottom

    xy = np.array([start_xy_img, end_xy_img], dtype=np.float32)
    bev = apply_homography_xy(xy, H)  # shape (2,2)

    dist_m = float(np.linalg.norm(bev[1] - bev[0]))

    dt_sec = (seg[i2].frame - seg[i1].frame) / float(cam.fps)
    if dt_sec <= 0.0:
        return None

    return dist_m / dt_sec


def _in_rect(x: float, y: float, xmin: float, xmax: float, ymin: float, ymax: float) -> bool:
    # 境界含む
    return (xmin <= x <= xmax) and (ymin <= y <= ymax)

def _angle_in_range_signed(angle_deg: float, ang_min: float, ang_max: float) -> bool:
    """
    角度レンジ判定（すべて [0,360] 前提）。
    """
    a = normalize_signed_deg(angle_deg)
    amin = normalize_signed_deg(ang_min)
    amax = normalize_signed_deg(ang_max)
    if amin < amax:
        return amin <= a <= amax
    else:
        return 0 <= a <= amax or amin <= a <= 360

def _rule_bounds(rule) -> Tuple[float, float, float, float, float, float]:
    """
    rule は dict 想定。欠損はデフォルト補完。
    """
    xmin = float(rule.get("xmin", -math.inf))
    xmax = float(rule.get("xmax",  math.inf))
    ymin = float(rule.get("ymin", -math.inf))
    ymax = float(rule.get("ymax",  math.inf))
    amin = float(rule.get("ang_min", 0))
    amax = float(rule.get("ang_max", 360))
    return (xmin, xmax, ymin, ymax, amin, amax)

def _classify_by_rules(xy: Tuple[float, float], ang_deg: float, rules: Dict[str, object]) -> List[str]:
    """
    位置(xy)と向き(ang_deg)が一致する **全てのルール** のキー（= 相手カメラ名）を配列で返す。
    - マッチ無しなら空リスト
    - ルールの順序（YAML記述順）をそのまま保持
    """
    if not (math.isfinite(ang_deg)) or not rules:
        return []

    x, y = float(xy[0]), float(xy[1])
    hits: List[str] = []
    for neighbor_cam, rule in rules.items():
        xmin, xmax, ymin, ymax, amin, amax = _rule_bounds(rule)
        if _in_rect(x, y, xmin, xmax, ymin, ymax) and _angle_in_range_signed(ang_deg, amin, amax):
            hits.append(neighbor_cam)
    return hits


def _compute_hs_mean(seg: List[TrackPoint], cam: CameraConfig) -> Optional[np.ndarray]:
    """
    conf >= cam.hs_conf_min のフレームだけ使って HS ヒストの等重み平均を取り、L1 正規化して返す。
    期待サイズ: cam.hs_hbins * cam.hs_sbins（サイズ不一致は除外）。
    """
    want_dim = int(cam.hs_hbins) * int(cam.hs_sbins)
    if want_dim <= 0:
        return None

    vecs = []
    for tp in seg:
        if tp.hs_hist is None:
            continue
        if not (tp.conf >= cam.hs_conf_min):
            continue
        arr = np.asarray(tp.hs_hist, dtype=np.float32).reshape(-1)
        if arr.size != want_dim:
            continue
        # フレーム側でL1済みだが、数値誤差に備えて再正規化
        s = float(arr.sum())
        if s > 0:
            arr = arr / s
        vecs.append(arr)

    if not vecs:
        return None

    mean = np.mean(np.stack(vecs, axis=0), axis=0)
    s = float(mean.sum())
    if s > 0:
        mean = mean / s
    return mean.astype(np.float32)


def build_summaries_after_roi_filter(raw_tracks: Dict[int, List[TrackPoint]], cam: CameraConfig) -> List[TrackSummary]:
    """
    1) ROI 除外 → 2) 連続セグメント化 → 3) 最長セグメント採用（min_track_len以上）
    → 4) TrackSummary を構築（方向・速度・HS平均）
    """
    # 1) ROI 除外（点ベース）
    filtered = filter_tracks_by_rois(raw_tracks, cam)

    summaries: List[TrackSummary] = []
    for tid, pts in filtered.items():
        if not pts:
            continue
        
        # 2) トラック全体を 1 セグメントとして扱う（フレーム順にソート）
        seg = sorted(pts, key=lambda p: p.frame)

        # 3) 長さチェック：min_track_len 未満のトラックは捨てる
        if len(seg) < cam.min_track_len:
            continue

        # 4) Summary 構築
        start_tp, end_tp = seg[0], seg[-1]
        start_xy = tuple(map(float, start_tp.center_bottom))
        end_xy = tuple(map(float, end_tp.center_bottom))

        # entry/exit 方向を “dir_smooth_frames” だけ先/前からとる（はみ出しに注意）
        w = max(1, int(cam.dir_smooth_frames))
        entry_src = seg[0].center_bottom
        entry_ref = seg[min(len(seg) - 1, w)].center_bottom  # 範囲内にクリップ
        exit_src = seg[max(0, len(seg) - 1 - w)].center_bottom
        exit_ref = seg[-1].center_bottom

        entry_dir = compute_direction(entry_src, entry_ref)
        exit_dir = compute_direction(exit_src, exit_ref)
        
        entry_dir_cams = _classify_by_rules(start_xy, entry_dir, cam.entry_rules)
        exit_dir_cams  = _classify_by_rules(end_xy,   exit_dir,  cam.exit_rules)

        speed = _estimate_tail_speed_mps(seg, cam, seconds=2.0)

        # HS 平均
        hs_mean = _compute_hs_mean(seg, cam)

        summaries.append(
            TrackSummary(
                cam=cam.name,
                track_id=tid,
                start_frame=start_tp.frame,
                end_frame=end_tp.frame,
                start_xy=(float(start_xy[0]), float(start_xy[1])),
                end_xy=(float(end_xy[0]), float(end_xy[1])),
                entry_dir_deg=float(entry_dir),
                exit_dir_deg=float(exit_dir),
                entry_dir_cams=entry_dir_cams,
                exit_dir_cams=exit_dir_cams,
                speed_mps=speed,
                hs_hist_avg=hs_mean,
                features_cache=({"hs_mean": hs_mean} if hs_mean is not None else {}),
            )
        )

    return summaries