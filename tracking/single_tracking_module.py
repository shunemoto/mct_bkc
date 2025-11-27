import ultralytics
from ultralytics import YOLO
import csv
import numpy as np
from collections import defaultdict
import cv2
import math
import itertools
import logging
import json
import os

HBINS, SBINS = 16, 16
HIST_DIM = HBINS * SBINS

def compute_hs_hist_from_hsv(hsv, mask_u8):
    """既にHSVに変換済みの画像から、マスク領域のHS 2DヒストをL1正規化で返す（float32, shape=(256,)）。"""
    hist2d = cv2.calcHist([hsv], [0, 1], mask_u8, [HBINS, SBINS], [0, 180, 0, 256])
    hist2d = cv2.normalize(hist2d, None, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist2d.flatten().astype(np.float32)

def _ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

def _open_video_writer(path: str, fps: float, size_wh: tuple[int, int]) -> cv2.VideoWriter:
    """複数コーデックを試して VideoWriter を開く。開けない場合は isOpened() が False のまま返る。"""
    W, H = size_wh
    tried = []
    for fourcc_str in ("mp4v", "avc1", "XVID"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        vw = cv2.VideoWriter(path, fourcc, float(fps), (W, H))
        tried.append(fourcc_str)
        if vw.isOpened():
            logging.info(f"VideoWriter opened with FOURCC='{fourcc_str}' at {fps} fps, size=({W}x{H}).")
            return vw
        else:
            vw.release()
    logging.error(f"VideoWriterを開けませんでした。試行したコーデック: {tried} / パス: {path}")
    return cv2.VideoWriter()  # not opened

def _probe_fps_from_file(filename: str, default_fps: float = 30.0) -> float:
    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not isinstance(fps, (int, float)) or not np.isfinite(fps) or fps <= 0:
        logging.warning(f"入力動画から有効なfpsを取得できませんでした（{fps}）。{default_fps} を使用します。")
        return float(default_fps)
    return float(fps)

def _bbox_iou_xyxy(a, b):
    """
    a, b: [x1, y1, x2, y2]
    IoU を返す。
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)

def single_tracking(
    filename,
    track_model,       # 検出＋追跡（BoT-SORT + ReID）用モデル（例: yolo11n.pt）
    seg_model,         # セグメンテーション用モデル（例: yolo11n-seg.pt）
    output_video_file,
    conf_limit,
    output_json_file,
    tracker_yaml
):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # 出力先フォルダを事前に用意
    _ensure_parent_dir(output_video_file)
    _ensure_parent_dir(output_json_file)

    track_history = defaultdict(list)
    out_video = None
    W = H = None

    # 実fpsを取得（不正なら30fpsへフォールバック）
    fps = _probe_fps_from_file(filename, default_fps=30.0)

    try:
        # 連続ストリームで追跡（BoT-SORT + ReID は track_model 側に任せる）
        frame_idx = -1
        for result in track_model.track(
            source=filename,
            stream=True,          # ストリームで回す
            persist=True,
            tracker=tracker_yaml
        ):
            frame_idx += 1

            # 元フレーム（BGR）
            frame = getattr(result, "orig_img", None)
            if frame is None:
                logging.warning(f"orig_img が None：frame_idx={frame_idx}")
                if out_video is not None and out_video.isOpened():
                    out_video.write(result.plot())
                continue

            if out_video is None:
                H, W = frame.shape[:2]
                out_video = _open_video_writer(output_video_file, fps=fps, size_wh=(W, H))
                if not out_video.isOpened():
                    logging.error("出力VideoWriterを開けなかったため、処理を中断します。")
                    return track_history  # JSONは書かずに終了

            # 1フレーム1回だけ HSV に変換
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            boxes = result.boxes
            if boxes is None or len(boxes) == 0 or boxes.xyxy is None or boxes.conf is None or boxes.cls is None:
                # 可視化フレームは plot() から取得
                out_video.write(result.plot())
                continue

            # 追跡側の検出結果（GPU→CPU）
            boxes_xyxy = boxes.xyxy.cpu().tolist()
            confs      = boxes.conf.cpu().tolist()
            class_ids  = boxes.cls.int().cpu().tolist()
            track_ids  = None if boxes.id is None else boxes.id.int().cpu().tolist()

            # このフレームに人物が1つでもいるか（cls==0）
            has_person = any(int(c) == 0 for c in class_ids)

            # ===== セグメンテーションを人物検出があるフレームだけ実行 =====
            seg_masks_full_255 = []
            seg_bboxes = []
            seg_classes = []

            if has_person and seg_model is not None:
                seg_results_list = seg_model.predict(source=frame, verbose=False)
                if len(seg_results_list) > 0:
                    seg_res = seg_results_list[0]
                    if seg_res.masks is not None and seg_res.masks.data is not None:
                        m_small = seg_res.masks.data.cpu().numpy()  # [N_seg, h, w]
                        seg_masks_full_255 = [
                            cv2.resize(
                                (m > 0.5).astype(np.uint8) * 255,
                                (W, H),
                                interpolation=cv2.INTER_NEAREST
                            )
                            for m in m_small
                        ]
                    if seg_res.boxes is not None and seg_res.boxes.xyxy is not None:
                        seg_bboxes = seg_res.boxes.xyxy.cpu().tolist()
                        seg_classes = (
                            seg_res.boxes.cls.int().cpu().tolist()
                            if seg_res.boxes.cls is not None
                            else [0] * len(seg_bboxes)
                        )

            # 各検出（track_model 側の boxes に従う）
            for i, (xyxy, conf, cls_id) in enumerate(zip(boxes_xyxy, confs, class_ids)):
                # 人物クラスのみ
                if int(cls_id) != 0:
                    continue
                if track_ids is None or i >= len(track_ids) or track_ids[i] is None:
                    continue

                tid = int(track_ids[i])
                x1, y1, x2, y2 = map(int, map(round, xyxy))
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                x_center = (x1 + x2) / 2.0
                y_bottom = float(y2)

                hs_hist_list = None
                if frame_idx > 29 and float(conf) >= float(conf_limit):
                    # ===== セグ結果の中から、IoU 最大の人物マスクを探す =====
                    person_mask = None
                    if seg_masks_full_255 and seg_bboxes:
                        best_j = -1
                        best_iou = 0.0
                        for j, (s_xyxy, s_cls) in enumerate(zip(seg_bboxes, seg_classes)):
                            if int(s_cls) != 0:
                                continue
                            iou = _bbox_iou_xyxy([x1, y1, x2, y2], list(map(float, s_xyxy)))
                            if iou > best_iou:
                                best_iou = iou
                                best_j = j
                        # IoU がある程度以上なら、そのマスクを採用
                        if best_j >= 0 and best_iou > 0.3:
                            person_mask = seg_masks_full_255[best_j]

                    # マッチするマスクがなければ、最後の手段として矩形マスク
                    if person_mask is None:
                        person_mask = np.zeros((H, W), np.uint8)
                        person_mask[y1:y2, x1:x2] = 255

                    hs_hist = compute_hs_hist_from_hsv(hsv, person_mask)
                    hs_hist_list = hs_hist.tolist()

                record = {
                    "frame_idx": int(frame_idx),
                    "center_bottom": [float(x_center), float(y_bottom)],
                    "conf": float(conf),
                    "left": float(x1), "top": float(y1), "right": float(x2), "bottom": float(y2),
                    "hs_hist": hs_hist_list,
                }
                track_history[tid].append(record)

            # 描画フレーム（可視化は追跡モデルの結果でOK）
            out_video.write(result.plot())

        logging.info("ビデオ処理が完了しました。")

        # ===== JSON 保存 =====
        meta = {
            "source_path": filename,
            "width": W, "height": H, "fps": fps,
            "hist_bins": {"h": HBINS, "s": SBINS},
            "conf_limit": conf_limit,
        }
        tracks_json = {str(tid): frames for tid, frames in track_history.items()}
        payload = {"meta": meta, "tracks": tracks_json}
        with open(output_json_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logging.info(f"JSON saved: {output_json_file}")

    finally:
        try:
            if out_video is not None and out_video.isOpened():
                out_video.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
