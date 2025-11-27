from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

from src.config_loader import load_scene_from_yaml
from src.io_tracks import load_multi_cams_from_json
from src.models import SceneConfig, TrackSummary
from src.summarize import build_summaries_after_roi_filter
from src.matching import (CandidateList, MatchScore, build_all_candidates, compute_all_match_scores)

def _read_data_paths_from_yaml(yaml_path: str) -> Dict[str, str]:
    """
    scene.yaml の data_paths セクションだけ取得する。
    SceneConfig には保持しない設計なので、デモ内で直接読む。
    """
    import yaml

    with open(yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}
    dp = y.get("data_paths", {}) or {}
    # 文字列化（YAML で数値等になってしまう可能性を避ける）
    return {str(k): str(v) for k, v in dp.items()}


def _build_all_summaries(scene: SceneConfig, tracks_by_cam: Dict[str, Dict[int, List]]) -> Dict[str, List[TrackSummary]]:
    """
    各カメラごとに TrackSummary の配列を作る。
    - ROI 除外は scene.yaml の cam.roi_boxes に従う
    - entry/exit の判定には cam.entry_rules / cam.exit_rules（矩形 + 角度）が使われる
    """
    summaries_by_cam: Dict[str, List[TrackSummary]] = {}
    for cam_name, cam_cfg in scene.cameras.items():
        raw_tracks = tracks_by_cam.get(cam_name, {})
        summaries = build_summaries_after_roi_filter(raw_tracks, cam_cfg)
        summaries_by_cam[cam_name] = summaries
    return summaries_by_cam


def _print_candidates_preview(scene: SceneConfig, summaries_by_cam: Dict[str, List[TrackSummary]], max_rows: int = 30) -> None:
    """
    可視化・検証用に、入口方向で抽出した候補の一覧を軽く表示。
    """
    cands: List[CandidateList] = build_all_candidates(scene, summaries_by_cam)
    print("\n[Preview] Entry-direction based candidates (first {} rows):".format(max_rows))
    print(" cam  | track_id | neighbors          -> candidates(neighbor:tid, ...)")
    print("------+----------+---------------------------------------------------")
    for i, c in enumerate(cands[:max_rows]):
        cand_str = ", ".join([f"{nb}:{tid}" for (nb, tid) in c.candidates]) if c.candidates else "-"
        neigh_str = ",".join(c.neighbors) if c.neighbors else "-"
        print(f" {c.cam:4s} | {c.track_id:8d} | {neigh_str:18s} -> {cand_str}")


def _print_match_scores_topk(all_scores: Dict[Tuple[str, int], List[MatchScore]], topk: int = 3) -> None:
    """
    融合スコアで降順ソートされたマッチ結果を (src_cam,src_id) ごとに上位 K 件表示。
    all_scores は compute_all_match_scores の結果をそのまま渡す。
    """
    print("\n[Results] Top-{} matches per (src_cam, src_id) ordered by Fusion Score:".format(topk))
    print(" src_cam src_id | tgt_cam tgt_id | pred_frame ΔF | Position  Appearance  Fusion")
    print("---------------+----------------+----------------+--------------------------------")
    for (src_cam, src_id), lst in all_scores.items():
        if not lst:
            print(f" {src_cam:7s} {src_id:6d} | (no candidates)")
            continue

        for rec in lst[:topk]:
            pf = "-" if rec.predicted_appearance_frame is None else f"{rec.predicted_appearance_frame:d}"
            df = "-" if rec.delta_frames is None else f"{rec.delta_frames:+d}"
            ps = "-" if rec.position_score is None else f"{rec.position_score:7.4f}"
            aps = "-" if rec.appearance_score is None else f"{rec.appearance_score:7.4f}"
            fs = "-" if rec.fusion_score is None else f"{rec.fusion_score:7.4f}"
            print(
                f" {src_cam:7s} {src_id:6d} | {rec.tgt_cam:7s} {rec.tgt_id:6d} |"
                f" {pf:10s} {df:>3s} | {ps:>8s}  {aps:>10s}  {fs:>6s}"
            )


def _save_match_scores_all(all_scores: Dict[Tuple[str, int], List[MatchScore]], out_path: str) -> None:
    """
    全てのマッチング結果を JSON に保存する。

    追加で次の値も保存する：
      - src_start_frame: ソーストラックの開始フレーム
      - tgt_end_frame:   ターゲットトラックの終了フレーム
      - tgt_speed_mps:   ターゲットトラックの速度[m/s]
    """
    serializable: List[dict] = []

    for (src_cam, src_id), lst in all_scores.items():
        for rec in lst:
            serializable.append({
                "src_cam": rec.src_cam,
                "src_id": rec.src_id,
                "src_start_frame": rec.src_start_frame,
                "tgt_cam": rec.tgt_cam,
                "tgt_id": rec.tgt_id,
                "tgt_end_frame": rec.tgt_end_frame,
                "tgt_speed_mps": rec.tgt_speed_mps,
                "predicted_appearance_frame": rec.predicted_appearance_frame,
                "delta_frames": rec.delta_frames,
                "position_score": rec.position_score,
                "appearance_score": rec.appearance_score,
                "fusion_score": rec.fusion_score,
            })

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    print(f"[Info] Saved all match scores to: {out_path}")


def _save_summaries_as_json(summaries_by_cam: Dict[str, List[TrackSummary]], out_path: str) -> None:
    """
    TrackSummary を JSON で保存。
    """
    
    serializable: Dict[str, List[dict]] = {}
    for cam_name, lst in summaries_by_cam.items():
        serializable[cam_name] = []
        for s in lst:
            hs = s.hs_hist_avg.tolist() if s.hs_hist_avg is not None else None
            serializable[cam_name].append({
                "cam": s.cam,
                "track_id": s.track_id,
                "start_frame": s.start_frame,
                "end_frame": s.end_frame,
                "start_xy": list(s.start_xy),
                "end_xy": list(s.end_xy),
                "entry_dir_deg": s.entry_dir_deg,
                "exit_dir_deg": s.exit_dir_deg,
                "entry_dir_cams": list(s.entry_dir_cams or []),
                "exit_dir_cams": list(s.exit_dir_cams or []),
                "speed_mps": s.speed_mps
            })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"[Info] Saved TrackSummary JSON to: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Multi-camera ID linking demo (ROI filter -> summaries -> candidate search -> scoring)."
    )
    ap.add_argument(
        "--yaml", "-y",
        default="configs/scene.yaml",
        help="Path to scene.yaml (default: configs/scene.yaml)",
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Top-K matches to print per source track (default: 3)",
    )
    ap.add_argument(
        "--preview-candidates",
        action="store_true",
        help="Print entry-direction based candidates preview before scoring.",
    )
    ap.add_argument(
        "--no-print",
        action="store_true",
        help="Do not print Top-K match scores to stdout.",
    )
    ap.add_argument(
        "--save-matches",
        type=str,
        default=None,
        help="Path to save all match scores as JSON (e.g., out/matches.json).",
    )
    ap.add_argument(
        "--save-summaries",
        type=str,
        default=None,
        help="Path to save TrackSummary as JSON (e.g., out/summaries.json).",
    )
    args = ap.parse_args()

    yaml_path = args.yaml
    if not os.path.isfile(yaml_path):
        print(f"[Error] YAML not found: {yaml_path}")
        sys.exit(1)

    # 1) Scene 読み込み（ホモグラフィ/各種閾値/融合重みなど）
    scene = load_scene_from_yaml(yaml_path, cache_dir=".cache_h")

    # 2) data_paths 読み込み（JSON のある場所）
    cam_to_json = _read_data_paths_from_yaml(yaml_path)
    if not cam_to_json:
        print("[Error] 'data_paths' section is empty in YAML.")
        sys.exit(1)

    # 3) JSON 読み込み
    tracks_by_cam, meta_by_cam = load_multi_cams_from_json(cam_to_json)

    # 4) TrackSummary 構築（ROI 除外→トラック全体→entry/exit 方向→速度→HS平均）
    summaries_by_cam = _build_all_summaries(scene, tracks_by_cam)

    # （任意）候補プレビュー
    if args.preview_candidates:
        _print_candidates_preview(scene, summaries_by_cam, max_rows=30)

    # 5) マッチ・スコア計算（位置 / 外観 / 融合）
    all_scores = compute_all_match_scores(scene, summaries_by_cam, sort_desc=True)

    # 標準出力への表示
    if not args.no_print:
        _print_match_scores_topk(all_scores, topk=args.topk)

    # JSON 保存（マッチ結果）
    if args.save_matches:
        _save_match_scores_all(all_scores, out_path=args.save_matches)

    # JSON 保存（TrackSummary）
    if args.save_summaries:
        _save_summaries_as_json(summaries_by_cam, out_path=args.save_summaries)


if __name__ == "__main__":
    main()