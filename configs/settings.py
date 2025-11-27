# 検出＋追跡（BoT-SORT + ReID）用モデル
track_model = 'yolo11n.pt'

# セグメンテーション用モデル
seg_model = 'yolo11n-seg.pt'

# 動画ファイルパス
filename = "/content/drive/MyDrive/video_3/CameraE_2.mp4"

# 出力動画ファイルパス
output_video_file = "/content/drive/MyDrive/video_5/tracking_result_E.mp4"

# 出力JSONファイルパス
output_json_file = "/content/drive/MyDrive/video_5/tracking_result_E.json"

# 信頼度閾値
conf_limit = 0.1

# BoT-SORTの設定ファイル
tracker_yaml = "/content/drive/MyDrive/ultralytics/botsort.yaml"