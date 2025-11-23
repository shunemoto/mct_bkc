from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional, Mapping
import json
import numpy as np
from .models import TrackPoint

def load_trackpoints_from_json(json_path: str,) -> Tuple[Dict[int, List[TrackPoint]], dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    meta: dict = payload.get("meta", {})
    tracks_raw: dict[str, Any] = payload.get("tracks", {})

    out: Dict[int, List[TrackPoint]] = {}

    for tid_str, frames in tracks_raw.items():
        try:
            tid = int(tid_str)
        except Exception:
            continue

        seq: List[TrackPoint] = []
        if not isinstance(frames, list):
            continue

        for rec in frames:
            # 必須: frame_idx, center_bottom
            try:
                frame = int(rec.get("frame_idx"))
                cb = rec.get("center_bottom")
                x = float(cb[0]); y = float(cb[1])
            except Exception:
                continue

            # 任意: conf, bbox, hs_hist
            def _f(v): 
                try: return float(v)
                except Exception: return float("nan")

            conf   = _f(rec.get("conf", 0.0))
            left   = _f(rec.get("left"))
            top    = _f(rec.get("top"))
            right  = _f(rec.get("right"))
            bottom = _f(rec.get("bottom"))

            h = rec.get("hs_hist", None)
            if h is None:
                hs_hist_arr: Optional[np.ndarray] = None
            else:
                try:
                    arr = np.asarray(h, dtype=np.float32).reshape(-1)
                    s = float(arr.sum())
                    if s > 0:
                        arr = arr / s  # フレーム毎にL1正規化（サイズチェックはしない）
                    hs_hist_arr = arr
                except Exception:
                    hs_hist_arr = None

            seq.append(TrackPoint(
                track_id=tid, frame=frame, x=x, y=y,
                conf=conf, left=left, top=top, right=right, bottom=bottom,
                hs_hist=hs_hist_arr
            ))

        if seq:
            seq.sort(key=lambda p: p.frame)
            out[tid] = seq

    return out, meta


def load_multi_cams_from_json(cam_to_json: Mapping[str, str]) -> Tuple[Dict[str, Dict[int, List[TrackPoint]]], Dict[str, dict]]:
    """
    cam_to_json: {"camA": "path/to/A.json", "camB": "...", ...}
    戻り値:
      tracks_by_cam: {"A": {tid: [TrackPoint,...]}, ...}
      meta_by_cam:   {"A": {...}, "B": {...}, ...}
    """
    tracks_by_cam: Dict[str, Dict[int, List[TrackPoint]]] = {}
    meta_by_cam: Dict[str, dict] = {}

    for cam, path in cam_to_json.items():
        tracks, meta = load_trackpoints_from_json(path)
        tracks_by_cam[cam] = tracks
        meta_by_cam[cam] = meta
    return tracks_by_cam, meta_by_cam