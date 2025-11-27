import logging
from ultralytics import YOLO
from single_tracking_module import single_tracking  # single_tracking をインポート（後述）

# 設定ファイルから設定をインポート
from configs.settings import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# track_model と seg_model を YOLO モデルとして読み込む
track_model = YOLO(settings.track_model)
seg_model = YOLO(settings.seg_model)

# single_tracking 関数を実行
single_tracking(
    filename=settings.filename,
    track_model=track_model,
    seg_model=seg_model,
    output_video_file=settings.output_video_file,
    conf_limit=settings.conf_limit,
    output_json_file=settings.output_json_file,
    tracker_yaml=settings.tracker_yaml
)
