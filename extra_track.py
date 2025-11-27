import json
import os
import sys


def extract_track(json_path: str, track_id: int) -> str:
    """
    JSONファイル内の tracks から、指定された track_id のトラックだけを抜き出して
    新しい JSON ファイルとして保存する。

    Parameters
    ----------
    json_path : str
        元の JSON ファイルのパス
    track_id : int
        抜き出したいトラックID（例: 1）

    Returns
    -------
    str
        出力した JSON ファイルのパス
    """
    
    # JSON ファイルを読み込み
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # "tracks" キーが存在するか確認
    if "tracks" not in data:
        raise KeyError(f'"tracks" キーが JSON 内に見つかりません: {json_path}')

    tracks = data["tracks"]

    # JSON 内のキーは文字列になっている想定なので、TrackID を文字列に変換
    key = str(track_id)

    if key not in tracks:
        raise KeyError(f'TrackID {key} が "tracks" に存在しません')

    # 元の data をベースに、新しい dict を作成
    # 他のメタデータがあればそのまま残し、tracks だけ絞り込む
    new_data = dict(data)
    new_data["tracks"] = {key: tracks[key]}

    # 出力ファイル名: 元ファイル名_track<ID>.json
    base, ext = os.path.splitext(json_path)
    if ext == "":
        ext = ".json"
    out_path = f"{base}_track{key}{ext}"

    # 新しい JSON ファイルとして保存
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    return out_path


def main():
    # コマンドライン引数:
    #   1: JSONファイルパス
    #   2: TrackID（整数）
    if len(sys.argv) != 3:
        print("Usage: python extract_track.py <json_path> <track_id>")
        sys.exit(1)

    json_path = sys.argv[1]

    try:
        track_id = int(sys.argv[2])
    except ValueError:
        print(f"TrackID は整数で指定してください: {sys.argv[2]}")
        sys.exit(1)

    try:
        out_path = extract_track(json_path, track_id)
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        sys.exit(1)

    print(f"TrackID {track_id} のみを含むファイルを出力しました: {out_path}")


if __name__ == "__main__":
    main()
