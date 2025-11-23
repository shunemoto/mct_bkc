from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List
import os
import yaml
import numpy as np
import cv2

from .models import CameraConfig, SceneConfig

# --------- 小ユーティリティ ---------
def _as_float(v: Any, default: float = float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default

def _ensure_np_xy2(points: Any, name: str) -> Optional[np.ndarray]:
    """
    [[x,y], ...] を (N,2) float32 の np.ndarray に。
    妥当でなければ None。
    """
    try:
        arr = np.asarray(points, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 1:
            return None
        return arr
    except Exception:
        return None

def _ensure_np33(mat_like: Any) -> Optional[np.ndarray]:
    """3x3 の行列 (list/np) を np.float64 に。"""
    try:
        M = np.asarray(mat_like, dtype=np.float64)
        if M.shape == (3, 3) and np.all(np.isfinite(M)):
            return M
    except Exception:
        pass
    return None

def _find_homography_from_pairs(src_pts: np.ndarray, dst_pts: np.ndarray) -> Optional[np.ndarray]:
    """
    src_pts, dst_pts: (N,2) float32, N>=4 推奨
    RANSAC で外れ値に頑健に推定。
    """
    if src_pts is None or dst_pts is None:
        return None
    if min(src_pts.shape[0], dst_pts.shape[0]) < 4:
        return None
    H, _mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=10.0)
    return H

def _load_cached_H(cache_dir: str, cam_name: str) -> Optional[np.ndarray]:
    path = os.path.join(cache_dir, f"{cam_name}_H.npy")
    if os.path.isfile(path):
        try:
            H = np.load(path)
            if isinstance(H, np.ndarray) and H.shape == (3, 3):
                return H.astype(np.float64)
        except Exception:
            pass
    return None

def _save_cached_H(cache_dir: str, cam_name: str, H: np.ndarray) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{cam_name}_H.npy")
    try:
        np.save(path, H)
    except Exception:
        # キャッシュ失敗は致命的ではないので握りつぶす
        pass

def _parse_rules_dict(rect_with_angle: Dict[str, Any]) -> Dict[str, dict]:
    """
    { label: {xmin:..., xmax:..., ymin:..., ymax:..., ang_min:..., ang_max:...}, ... }
    → そのまま dict を返す（dataclass は用いない）。
    欠けているキーは後段でデフォルト補完される想定。
    """
    out: Dict[str, dict] = {}
    for label, spec in (rect_with_angle or {}).items():
        if isinstance(spec, dict):
            out[label] = dict(spec)  # 浅いコピー
    return out

def _polar_to_xy(points: Any, x_axis_deg: float) -> Optional[np.ndarray]:
    """
    points: [{r: float, theta: float}, ...]  (theta: コンパス角, 北=0°, 時計回り+)
    x_axis_deg: この方位を +X 方向とみなす（例: 137）
    return: (N,2) float32（単位はメートルを推奨）
    """
    if points is None:
        return None
    try:
        rs, thetas = [], []
        for p in points:
            if not isinstance(p, dict):
                return None
            r = float(p.get("r", None))
            th = float(p.get("theta", None))
            if not np.isfinite(r) or not np.isfinite(th):
                return None
            rs.append(r); thetas.append(th)
        rs = np.asarray(rs, dtype=np.float64)
        thetas = np.asarray(thetas, dtype=np.float64)
        rel_deg = thetas - float(x_axis_deg)         # 定義した x 軸を 0° に
        rel_rad = np.deg2rad(rel_deg)
        xs = rs * np.cos(rel_rad)
        ys = rs * np.sin(rel_rad)
        return np.stack([xs, ys], axis=1).astype(np.float32)
    except Exception:
        return None


# --------- メイン：YAML→SceneConfig ---------
def load_scene_from_yaml(yaml_path: str, cache_dir: str = ".cache_h") -> SceneConfig:
    """
    scene.yaml を読み、各カメラの CameraConfig を組み立てて SceneConfig を返す。

    - 共通値(common)を各カメラにマージ
    - homography は以下の優先順位で決定し、source を記録
        1) キャッシュ (cache_dir/{cam}_H.npy) → homography_source="cache"
        2) YAML直書き (calib.homography)      → "yaml"
        3) calib.src_points / calib.dst_points  → "calib"
           または (polar_points + x_axis_defined_deg) から dst_points を生成して推定 → "calib"
      ※ いずれも無ければ None（source=None）
    - data_paths: {"camA": "/path/A.json", ...} は SceneConfig には保持しない（デモで直接読む）
    - HSヒスト/信頼度しきい値は common または camera 個別で指定可
    - matching.fusion_w_pos, matching.fusion_w_app を SceneConfig に保持
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)

    common: Dict[str, Any] = y.get("common", {}) or {}
    cams_y: Dict[str, Any] = y.get("cameras", {}) or {}

    # --- matching（位置/外観の融合重み） ---
    matching_cfg: Dict[str, Any] = y.get("matching", {}) or {}
    fusion_w_pos = float(matching_cfg.get("fusion_w_pos", 0.5))
    fusion_w_app = float(matching_cfg.get("fusion_w_app", 0.5))

    # --- 共通のHS設定（flatキーを推奨） ---
    common_hs_hbins = int(common.get("hs_hbins", 16))
    common_hs_sbins = int(common.get("hs_sbins", 16))
    common_hs_conf_min = float(common.get("hs_conf_min", 0.8))

    cameras: Dict[str, CameraConfig] = {}

    for cam_name, conf in cams_y.items():
        # 1) 共通をベースに、個別で上書き
        merged = dict(common)
        merged.update(conf or {})

        # 2) ホモグラフィの決定
        H: Optional[np.ndarray] = None
        H_src: Optional[str] = None

        # 2-1) キャッシュ優先
        H_cache = _load_cached_H(cache_dir, cam_name)
        if H_cache is not None:
            H, H_src = H_cache, "cache"
        else:
            calib = merged.get("calib", {}) or {}

            # 2-2) YAML直書きの H
            H_yaml = _ensure_np33(calib.get("homography"))
            if H_yaml is not None:
                H, H_src = H_yaml, "yaml"
            else:
                # 2-3) src/dst points or polar→dst から推定
                src_points = _ensure_np_xy2(calib.get("src_points"), "src_points")

                dst_points = None
                polar_pts = calib.get("polar_points", None)
                x_axis_deg = calib.get("x_axis_defined_deg", None)
                if polar_pts is not None and x_axis_deg is not None:
                    dst_points = _polar_to_xy(polar_pts, float(x_axis_deg))

                if src_points is not None and dst_points is not None:
                    H_est = _find_homography_from_pairs(src_points, dst_points)
                    if H_est is not None:
                        H, H_src = H_est, "calib"
                        _save_cached_H(cache_dir, cam_name, H_est)

        # 3) その他のフィールド
        fps = float(merged.get("fps", 30.0))
        min_track_len = int(merged.get("min_track_len", 31))
        dir_smooth_frames = int(merged.get("dir_smooth_frames", 30))
        sigma_frames = float(merged.get("sigma_frames", 45.0))

        neighbors = list(merged.get("neighbors", []) or [])
        distance_to = {str(k): float(v) for k, v in (merged.get("distance_to", {}) or {}).items()}

        # roi_boxes: {label: [xmin,xmax,ymin,ymax]} など
        roi_boxes_raw = merged.get("roi_boxes", {}) or {}
        roi_boxes: Dict[str, Any] = {}
        for lb, box in roi_boxes_raw.items():
            if not isinstance(box, (list, tuple)):
                continue

            # ケース1: 矩形 [xmin, xmax, ymin, ymax]
            if len(box) == 4 and all(not isinstance(v, (list, tuple)) for v in box):
                try:
                    xmin, xmax, ymin, ymax = map(float, box)
                    roi_boxes[lb] = (xmin, xmax, ymin, ymax)
                except Exception:
                    continue

            # ケース2: 三角形（or 多角形） [[x1,y1], [x2,y2], [x3,y3], ...]
            elif len(box) >= 3 and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in box):
                try:
                    pts: List[Tuple[float, float]] = [(float(p[0]), float(p[1])) for p in box]
                    roi_boxes[lb] = pts  # filtering 側で三角形として利用
                except Exception:
                    continue

        entry_rules = _parse_rules_dict(merged.get("entry_rules", {}) or {})
        exit_rules  = _parse_rules_dict(merged.get("exit_rules", {}) or {})

        # candidate_time_window: {neighbor: [pre_min, pre_max]}
        ctw_raw = merged.get("candidate_time_window", {}) or {}
        candidate_time_window: Dict[str, Tuple[int, int]] = {}
        for nb, arr in ctw_raw.items():
            if isinstance(arr, (list, tuple)) and len(arr) == 2:
                pre_min = int(arr[0]); pre_max = int(arr[1])
                candidate_time_window[str(nb)] = (pre_min, pre_max)

        # --- HS/信頼度しきい値（個別→無ければ共通） ---
        hs_hbins = int(merged.get("hs_hbins", common_hs_hbins))
        hs_sbins = int(merged.get("hs_sbins", common_hs_sbins))
        hs_conf_min = float(merged.get("hs_conf_min", common_hs_conf_min))

        cameras[cam_name] = CameraConfig(
            name=cam_name,
            fps=fps,
            homography=H,
            homography_source=H_src,
            neighbors=neighbors,
            distance_to=distance_to,
            min_track_len=min_track_len,
            dir_smooth_frames=dir_smooth_frames,
            sigma_frames=sigma_frames,
            roi_boxes=roi_boxes,
            entry_rules=entry_rules,
            exit_rules=exit_rules,
            candidate_time_window=candidate_time_window,
            hs_hbins=hs_hbins,
            hs_sbins=hs_sbins,
            hs_conf_min=hs_conf_min,
        )

    # SceneConfig
    default_sigma = float(common.get("sigma_frames", 45.0))
    scene = SceneConfig(
        cameras=cameras,
        default_sigma_frames=default_sigma,
        fusion_w_pos=fusion_w_pos,
        fusion_w_app=fusion_w_app,
    )
    return scene