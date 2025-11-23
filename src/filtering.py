from __future__ import annotations
from typing import Dict, List, Tuple, Iterable, Union, Any
import math
from .models import TrackPoint, CameraConfig

# 型エイリアス
Rect = Tuple[float, float, float, float]          # (xmin, xmax, ymin, ymax)
Point = Tuple[float, float]                       # (x, y)


# ----------------- 基本チェック -----------------

def _rect_valid(r: Rect) -> bool:
    xmin, xmax, ymin, ymax = r
    return (xmax >= xmin) and (ymax >= ymin)


def _point_in_rect(x: float, y: float, rect: Rect, inclusive: bool = True) -> bool:
    xmin, xmax, ymin, ymax = rect
    if inclusive:
        return (xmin <= x <= xmax) and (ymin <= y <= ymax)
    else:
        return (xmin < x < xmax) and (ymin < y < ymax)


# ----------------- 点が三角形内にあるか判定 -----------------

def _is_point_in_triangle(
    x: float,
    y: float,
    A: Point,
    B: Point,
    C: Point,
    eps: float = 1e-6,
) -> bool:
    """
    点 (x,y) が三角形 ABC の内部（または境界上）かどうかを判定する。
    バarycentric 座標を使った方法。
    """
    Ax, Ay = A
    Bx, By = B
    Cx, Cy = C

    v0x, v0y = Cx - Ax, Cy - Ay
    v1x, v1y = Bx - Ax, By - Ay
    v2x, v2y = x - Ax, y - Ay

    dot00 = v0x * v0x + v0y * v0y
    dot01 = v0x * v1x + v0y * v1y
    dot02 = v0x * v2x + v0y * v2y
    dot11 = v1x * v1x + v1y * v1y
    dot12 = v1x * v2x + v1y * v2y

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < eps:
        # 面積ほぼゼロ（三点がほぼ一直線）の場合は除外
        return False

    inv_denom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    # 0 <= u, v, u+v <= 1 なら内部（境界含む）
    return (u >= -eps) and (v >= -eps) and (u + v <= 1.0 + eps)


# ----------------- ROI 判定 -----------------

def _point_in_any_roi(x: float, y: float, rois: Iterable[Any], inclusive: bool = True) -> bool:
    """
    いずれかの ROI（矩形 or 三角形）に入っていれば True。NaNは常にFalse。

    ROI の表現は以下の2種類を想定
      - 矩形: (xmin, xmax, ymin, ymax) もしくは [xmin, xmax, ymin, ymax]
      - 三角（多角）形: [(x1,y1), (x2,y2), (x3,y3)]  それ以上の頂点数の場合も先頭3点で三角形として判定
    """
    if not (math.isfinite(x) and math.isfinite(y)):
        return False

    for shape in rois:
        # --- 矩形形式: 4つのスカラー ---
        if isinstance(shape, (list, tuple)) and len(shape) == 4 and all(
            not isinstance(v, (list, tuple)) for v in shape
        ):
            try:
                xmin, xmax, ymin, ymax = map(float, shape)
            except Exception:
                continue
            if _rect_valid((xmin, xmax, ymin, ymax)) and _point_in_rect(
                x, y, (xmin, xmax, ymin, ymax), inclusive=inclusive
            ):
                return True

        # --- 三角形（or 多角形）形式: [[x1,y1], [x2,y2], [x3,y3], ...] ---
        elif isinstance(shape, (list, tuple)) and len(shape) >= 3 and all(
            isinstance(p, (list, tuple)) and len(p) == 2 for p in shape
        ):
            try:
                # ここでは先頭3点を三角形として判定（多角形拡張は必要ならあとで）
                A = (float(shape[0][0]), float(shape[0][1]))
                B = (float(shape[1][0]), float(shape[1][1]))
                C = (float(shape[2][0]), float(shape[2][1]))
            except Exception:
                continue

            if _is_point_in_triangle(x, y, A, B, C):
                return True

        # それ以外の形式は無視

    return False


# ----------------- トラックフィルタリング -----------------

def filter_tracks_by_rois(tracks: Dict[int, List[TrackPoint]], cam: CameraConfig, inclusive: bool = True) -> Dict[int, List[TrackPoint]]:
    """
    ROI（矩形 / 三角形）領域(center_bottom基準) に入っているフレームだけ“点で”除外。
    track 自体は残すが、全点が除外された track は削除。

    - cam.roi_boxes は config_loader.py でパースされた dict で、
      値は (xmin,xmax,ymin,ymax) または [(x1,y1), (x2,y2), (x3,y3)] など。
    """
    # ROI が何もなければ昇順コピーを返す
    if not cam.roi_boxes:
        return {tid: sorted(seq, key=lambda p: p.frame) for tid, seq in tracks.items()}

    # CameraConfig.roi_boxes の value をそのまま渡す（矩形 or 三角形）
    rois = list(cam.roi_boxes.values())

    out: Dict[int, List[TrackPoint]] = {}
    for tid, seq in tracks.items():
        if not seq:
            continue
        seq_sorted = sorted(seq, key=lambda p: p.frame)
        kept: List[TrackPoint] = []
        for tp in seq_sorted:
            x, y = tp.center_bottom
            # ROI 内なら「除外」、外なら「保持」
            if not _point_in_any_roi(float(x), float(y), rois, inclusive=inclusive):
                kept.append(tp)
        if kept:
            out[tid] = kept
    return out


def split_into_contiguous_segments(points: List[TrackPoint]) -> List[List[TrackPoint]]:
    """
    フレーム連番でセグメント分割（差が1なら連続、>1で分割）。
    入力が未整列でも安全に動作するよう、ここで昇順化する。
    """
    if not points:
        return []
    pts = sorted(points, key=lambda p: p.frame)

    segs: List[List[TrackPoint]] = []
    cur: List[TrackPoint] = [pts[0]]
    for prev, curp in zip(pts, pts[1:]):
        if curp.frame == prev.frame + 1:
            cur.append(curp)
        else:
            segs.append(cur)
            cur = [curp]
    segs.append(cur)
    return segs