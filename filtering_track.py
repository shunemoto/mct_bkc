import json
import os
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable, Union

import numpy as np


# =========================
# データクラス
# =========================

@dataclass(slots=True)
class TrackPoint:
    """
    1フレーム分のトラッキング情報。
    """
    track_id: int
    frame: int
    x: float
    y: float
    conf: float = 0.0
    left: float = float("nan")
    top: float = float("nan")
    right: float = float("nan")
    bottom: float = float("nan")
    hs_hist: Optional[np.ndarray] = None

    @property
    def center_bottom(self) -> Tuple[float, float]:
        """(x, y) をタプルで返すヘルパー。"""
        return (self.x, self.y)


@dataclass(slots=True)
class CameraConfig:
    """
    scene.yaml の各カメラ設定をまとめたもの。
    （ここではデバッグ用に必要最低限だけ使う）
    """
    name: str
    fps: float

    # ホモグラフィ関連
    homography: Optional[np.ndarray]
    homography_source: Optional[str]

    # 近接カメラ・距離情報
    neighbors: List[str]
    distance_to: Dict[str, float]

    # トラック要件・方向推定
    min_track_len: int
    dir_smooth_frames: int
    sigma_frames: float

    # ROI・入口/出口ルール
    # 実際には Tuple だけど、ここでは三角形も入れるので型は無視して使う
    roi_boxes: Dict[str, object]
    entry_rules: Dict[str, dict]
    exit_rules: Dict[str, dict]
    candidate_time_window: Dict[str, Tuple[int, int]]

    # HSヒストと信頼度しきい値
    hs_hbins: int = 16
    hs_sbins: int = 16
    hs_conf_min: float = 0.7


# =========================
# ROI 判定用の型・ヘルパー
# =========================

# Rect: (xmin, xmax, ymin, ymax)
Rect = Tuple[float, float, float, float]
# Tri : ((Ax,Ay), (Bx,By), (Cx,Cy))
Tri = Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
ROIShape = Union[Rect, Tri]


def _rect_valid(rect: Rect, eps: float = 1e-9) -> bool:
    xmin, xmax, ymin, ymax = rect
    if not all(math.isfinite(v) for v in rect):
        return False
    return (xmax - xmin) > eps and (ymax - ymin) > eps


def _tri_valid(tri: Tri, eps: float = 1e-9) -> bool:
    (ax, ay), (bx, by), (cx, cy) = tri
    if not all(
        math.isfinite(v) for v in (ax, ay, bx, by, cx, cy)
    ):
        return False
    # 面積の2倍（外積）でチェック
    area2 = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
    return abs(area2) > eps


def _point_in_triangle(
    x: float,
    y: float,
    tri: Tri,
    eps: float = 1e-6,
    inclusive: bool = True,
) -> bool:
    """
    点 (x, y) が三角形 tri 内にあるか判定。
    inclusive=True のときは辺上も「内側」とみなす。
    """
    (ax, ay), (bx, by), (cx, cy) = tri
    px, py = float(x), float(y)

    def cross_sign(x1, y1, x2, y2, x3, y3) -> float:
        # ベクトル (x1-x3, y1-y3) と (x2-x3, y2-y3) の外積 z 成分
        return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)

    s1 = cross_sign(px, py, ax, ay, bx, by)
    s2 = cross_sign(px, py, bx, by, cx, cy)
    s3 = cross_sign(px, py, cx, cy, ax, ay)

    has_pos = (s1 > eps) or (s2 > eps) or (s3 > eps)
    has_neg = (s1 < -eps) or (s2 < -eps) or (s3 < -eps)

    if inclusive:
        # 正と負が同時にある → 三角形外
        return not (has_pos and has_neg)
    else:
        # 正負混在なし かつ どれも 0 に近くない → 完全内部
        return (not (has_pos and has_neg)) and not (
            abs(s1) <= eps or abs(s2) <= eps or abs(s3) <= eps
        )


def _point_in_any_roi(
    x: float,
    y: float,
    rois: Iterable[ROIShape],
    inclusive: bool = True,
    eps: float = 1e-6,
) -> bool:
    """
    いずれかの ROI（矩形 or 三角形）に点 (x,y) が入っていれば True。
    """
    if not (math.isfinite(x) and math.isfinite(y)):
        return False

    for spec in rois:
        # list/tuple 以外はスキップ
        if not isinstance(spec, (list, tuple)):
            continue

        # 矩形: 長さ4 & 全要素が数値
        if (
            len(spec) == 4
            and all(isinstance(v, (int, float)) for v in spec)
        ):
            xmin, xmax, ymin, ymax = map(float, spec)  # type: ignore
            if not _rect_valid((xmin, xmax, ymin, ymax)):
                continue
            if inclusive:
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    return True
            else:
                if xmin < x < xmax and ymin < y < ymax:
                    return True
            continue

        # 三角形: 長さ3 & 各要素が (x,y)
        if (
            len(spec) == 3
            and all(
                isinstance(p, (list, tuple)) and len(p) == 2
                for p in spec
            )
        ):
            A = (float(spec[0][0]), float(spec[0][1]))
            B = (float(spec[1][0]), float(spec[1][1]))
            C = (float(spec[2][0]), float(spec[2][1]))
            tri: Tri = (A, B, C)
            if not _tri_valid(tri):
                continue
            if _point_in_triangle(x, y, tri, eps=eps, inclusive=inclusive):
                return True
            continue

        # それ以外の形式は無視
        continue

    return False


# =========================
# トラックの ROI フィルタ
# =========================

def filter_tracks_by_rois(
    tracks: Dict[int, List[TrackPoint]],
    cam: CameraConfig,
    inclusive: bool = True,
) -> Dict[int, List[TrackPoint]]:
    """
    ROI（矩形または三角形）の内側にあるフレームだけ“点で”除外。
    - cam.roi_boxes には複数の ROI を定義できる。
      例:
        矩形: [xmin, xmax, ymin, ymax]
        三角: [(x1, y1), (x2, y2), (x3, y3)]
    - track 自体は残すが、全点が除外された track は削除。
    """
    rois: List[ROIShape] = list(cam.roi_boxes.values()) if cam.roi_boxes else []
    if not rois:
        # 除外領域がなければ昇順コピーを返す
        return {tid: sorted(seq, key=lambda p: p.frame) for tid, seq in tracks.items()}

    out: Dict[int, List[TrackPoint]] = {}
    for tid, seq in tracks.items():
        if not seq:
            continue
        seq_sorted = sorted(seq, key=lambda p: p.frame)
        kept: List[TrackPoint] = []
        for tp in seq_sorted:
            x, y = tp.center_bottom
            # ROI 内なら除外、外なら保持
            if not _point_in_any_roi(float(x), float(y), rois, inclusive=inclusive):
                kept.append(tp)
        if kept:
            out[tid] = kept
    return out


# =========================
# JSON 読み込みと保存
# =========================

def load_tracks_from_json(json_path: str) -> Dict[int, List[TrackPoint]]:
    """
    JSONファイルから TrackPoint 群を読み込んで
    Dict[int, List[TrackPoint]] に変換する。
    想定フォーマット（例）:

    {
      "tracks": {
        "1": [
          {
            "frame_idx": 534,
            "center_bottom": [917.5, 432.0],
            "conf": 0.35,
            "left": 878.0,
            "top": 164.0,
            "right": 957.0,
            "bottom": 432.0,
            "hs_hist": [...]
          },
          ...
        ],
        "2": [...]
      }
    }

    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # "tracks" キーがあればそこを見る、なければトップレベルを tracks とみなす
    tracks_raw = data.get("tracks", data)

    out: Dict[int, List[TrackPoint]] = {}

    for tid_str, items in tracks_raw.items():
        try:
            tid = int(tid_str)
        except ValueError:
            # track_id が数字でない場合はスキップ
            continue

        seq: List[TrackPoint] = []
        for item in items:
            # frame 番号
            if "frame" in item:
                frame = int(item["frame"])
            else:
                frame = int(item.get("frame_idx", 0))

            # 位置
            if "center_bottom" in item and item["center_bottom"] is not None:
                x, y = item["center_bottom"]
            else:
                x = item.get("x", 0.0)
                y = item.get("y", 0.0)

            conf = float(item.get("conf", 0.0))
            left = float(item.get("left", float("nan")))
            top = float(item.get("top", float("nan")))
            right = float(item.get("right", float("nan")))
            bottom = float(item.get("bottom", float("nan")))

            hs = item.get("hs_hist", None)
            if hs is not None:
                hs_arr = np.asarray(hs, dtype=np.float32)
            else:
                hs_arr = None

            tp = TrackPoint(
                track_id=tid,
                frame=frame,
                x=float(x),
                y=float(y),
                conf=conf,
                left=left,
                top=top,
                right=right,
                bottom=bottom,
                hs_hist=hs_arr,
            )
            seq.append(tp)

        if seq:
            out[tid] = seq

    return out


def save_filtered_track_to_json(
    json_path: str,
    target_track_id: int,
    raw_tracks: Dict[int, List[TrackPoint]],
    cam: CameraConfig,
) -> str:
    """
    1) ROI フィルタを適用したのち、
    2) 指定した TrackID のトラックだけを
       別 JSON ファイルに保存するデバッグ用関数。
    """
    # 1) ROI フィルタ適用
    filtered = filter_tracks_by_rois(raw_tracks, cam)

    # 指定IDのトラックを取得（存在しなければ空リスト）
    track_pts = filtered.get(target_track_id, [])

    # JSON に書き出せる形に変換
    points_json = []
    for tp in sorted(track_pts, key=lambda p: p.frame):
        if tp.hs_hist is not None:
            hs_hist_list = tp.hs_hist.tolist()
        else:
            hs_hist_list = None

        points_json.append(
            {
                "track_id": int(tp.track_id),
                "frame": int(tp.frame),
                "x": float(tp.x),
                "y": float(tp.y),
            }
        )

    out_data = {
        "cam": cam.name,
        "track_id": int(target_track_id),
        "num_points_after_roi_filter": len(points_json),
        "roi_boxes": cam.roi_boxes,
        "points": points_json,
    }

    base, ext = os.path.splitext(json_path)
    if not ext:
        ext = ".json"
    out_path = f"{base}_tid{target_track_id}_filtered{ext}"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    return out_path


# =========================
# main
# =========================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="特定 TrackID の ROI フィルタ後トラックを別 JSON に保存して確認するスクリプト"
    )
    parser.add_argument(
        "--json",
        required=True,
        help="トラックが入っている JSON ファイルパス（tracks.{id} 配下）",
    )
    parser.add_argument(
        "--track-id",
        type=int,
        required=True,
        help="フィルタ後の軌跡を確認したい TrackID（整数）",
    )
    args = parser.parse_args()

    json_path = args.json
    target_tid = args.track_id

    print(f"[INFO] loading tracks from: {json_path}")
    raw_tracks = load_tracks_from_json(json_path)
    print(f"[INFO] loaded {len(raw_tracks)} tracks")

    # camE 用の CameraConfig（デバッグ用に必要な項目だけまともならOK）
    camE = CameraConfig(
        name="camE",
        fps=30.0,
        homography=None,
        homography_source=None,
        neighbors=[],
        distance_to={},
        min_track_len=1,
        dir_smooth_frames=1,
        sigma_frames=1.0,
        roi_boxes={
            # 問題の drop_zone_1（三角形）
            "drop_zone_1": [(500.0, 0.0), (1920.0, 0.0), (1920.0, 555.0)],
        },
        entry_rules={},
        exit_rules={},
        candidate_time_window={},
    )

    print(f"[INFO] applying ROI filter for cam: {camE.name}")
    out_path = save_filtered_track_to_json(
        json_path=json_path,
        target_track_id=target_tid,
        raw_tracks=raw_tracks,
        cam=camE,
    )

    print(f"[INFO] filtered track {target_tid} saved to: {out_path}")


if __name__ == "__main__":
    main()
