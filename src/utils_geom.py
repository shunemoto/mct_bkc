import numpy as np
import math
import cv2

def normalize_signed_deg(a: float) -> float:
    """角度a[deg]を[0, 360)に正規化"""
    return a % 360.0

def _to_xy(p) -> tuple[float, float]:
    """
    入力 p から (x, y) を取り出すユーティリティ。
    - p が .x, .y 属性を持つオブジェクト（例: Point, TrackPoint）
    - p が [x,y] / (x,y) のようなシーケンス
    のどちらにも対応する。
    """
    # パターン1: .x / .y 属性を持つオブジェクト
    if hasattr(p, "x") and hasattr(p, "y"):
        return float(p.x), float(p.y)

    # パターン2: インデックスアクセス可能なシーケンス（tuple, list, np.ndarray など）
    try:
        return float(p[0]), float(p[1])
    except Exception as e:
        raise TypeError(f"_to_xy: サポートされていない型です: {type(p)}") from e

def compute_direction(p1, p2) -> float:
    """
    2点 p1, p2 の間の「p1 → p2」方向を、画像座標系での方位[deg]として返す。

    - p1, p2 は
        - .x, .y 属性を持つオブジェクト（例: Point, TrackPoint.center_bottom を持つクラス）
        - または (x, y) / [x, y] / np.array([x, y]) など
      のどちらでもよい。
    - 座標系は画像座標系（左上原点, x→右+, y→下+）を仮定。
      → 0° = 右, +90° = 下, -90° = 上。
    - ベクトルがゼロ、または NaN/inf が含まれる場合は NaN を返す。
    """
    x1, y1 = _to_xy(p1)
    x2, y2 = _to_xy(p2)

    if not (math.isfinite(x1) and math.isfinite(y1) and math.isfinite(x2) and math.isfinite(y2)):
        return float("nan")

    dx = x2 - x1
    dy = y2 - y1

    if dx == 0.0 and dy == 0.0:
        # 方向が定義できない
        return float("nan")

    ang_rad = math.atan2(dy, dx)      # ラジアン
    ang_deg = math.degrees(ang_rad)   # 度に変換
    return normalize_signed_deg(ang_deg)

def apply_homography_xy(xy_np: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    xy_np: (N,2) のfloat配列。返り: (N,2)
    OpenCVのperspectiveTransformで正規化。非有限が出た点はそのまま返るので、
    呼び出し側で必要なら除去してください。
    """
    pts = np.asarray(xy_np, dtype=np.float32).reshape(-1, 1, 2)
    H = np.asarray(H, dtype=np.float64).reshape(3, 3)
    dst = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return dst

def gaussian_similarity(delta_frames: np.ndarray, sigma: float) -> np.ndarray:
    """
    Δフレーム差に対するガウス類似度。sigma<=0は微小値に丸める。
    """
    df = np.asarray(delta_frames, dtype=np.float64)
    sig = float(sigma)
    if not np.isfinite(sig) or sig <= 0.0:
        sig = 1e-9
    return np.exp(-(df**2) / (2.0 * sig**2))
