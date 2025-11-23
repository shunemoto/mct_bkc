from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


# ========== 単フレーム情報（元トラッカー出力の1点） ==========

@dataclass(slots=True)
class TrackPoint:
    """
    1フレーム分のトラッキング情報。

    - track_id: トラッカー内でのID
    - frame: フレーム番号
    - x, y: center_bottom の座標（画像座標系）
    - conf: 検出信頼度
    - left, top, right, bottom: バウンディングボックス（必要なら使用）
    - hs_hist: HSヒストグラム (HBINS*SBINS,) もしくは None
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


# ========== カメラ設定 ==========

@dataclass(slots=True)
class CameraConfig:
    """
    scene.yaml の各カメラ設定をまとめたもの。
    config_loader.load_scene_from_yaml から生成される。
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
    roi_boxes: Dict[str, Tuple[float, float, float, float]]
    entry_rules: Dict[str, dict]
    exit_rules: Dict[str, dict]
    candidate_time_window: Dict[str, Tuple[int, int]]

    # HSヒストと信頼度しきい値
    hs_hbins: int = 16
    hs_sbins: int = 16
    hs_conf_min: float = 0.7


# ========== トラックの要約（一人の1カメラ内の1トラック） ==========

@dataclass(slots=True)
class TrackSummary:
    """
    1トラック（1人1カメラ）の要約情報。

    - cam: カメラ名
    - track_id: トラッカーID
    - start_frame, end_frame: 最初と最後のフレーム
    - start_xy, end_xy: 画像座標系での開始/終了位置 (x, y)
    - entry_dir_deg, exit_dir_deg: 入口/出口の向き（度。右=0°, 下=+90°）
    - entry_dir_cams, exit_dir_cams:
        入口/出口で「候補になりうるカメラ名」のリスト（複数可）
    - speed_mps: トラック末尾の速度 [m/s]（計算できなければ None）
    - hs_hist_avg: そのトラックの HS ヒストグラム平均（L1正規化済み）または None
    - features_cache: 追加特徴量用のキャッシュ（今後拡張用）
    """
    cam: str
    track_id: int
    start_frame: int
    end_frame: int
    start_xy: Tuple[float, float]
    end_xy: Tuple[float, float]
    entry_dir_deg: float
    exit_dir_deg: float

    # ★ ここを「単数」ではなく「複数候補のリスト」にした
    entry_dir_cams: List[str]
    exit_dir_cams: List[str]

    # 以下は「無くても動くように」デフォルト値を与えている
    speed_mps: Optional[float] = None
    hs_hist_avg: Optional[np.ndarray] = None
    features_cache: Dict[str, Any] = field(default_factory=dict)


# ========== シーン全体の設定（全カメラ＋グローバル設定） ==========

@dataclass(slots=True)
class SceneConfig:
    """
    シーン全体の設定。
    - cameras: カメラ名 → CameraConfig
    - default_sigma_frames: 時間ガウシアンのデフォルトσ（カメラ側で未指定時に使用）
    - fusion_w_pos / fusion_w_app: 位置スコアと外観スコアの融合重み
    """
    cameras: Dict[str, CameraConfig]
    default_sigma_frames: float = 45.0
    fusion_w_pos: float = 0.5
    fusion_w_app: float = 0.5
