# import os
# import glob
# import json
# import tarfile
# import io

# def find_episode_logs():
#     """実際のログファイル構造に基づいてエピソードログを検索"""
#     # ルートログディレクトリ
#     logs_root = "./logs"
    
#     # ステップ1: 実験ディレクトリを見つける
#     experiment_dirs = sorted([os.path.join(logs_root, d) for d in os.listdir(logs_root) 
#                              if d.startswith("experiment_")])
    
#     if not experiment_dirs:
#         print("実験ディレクトリが見つかりません")
#         return None
        
#     latest_experiment = experiment_dirs[-1]
#     print(f"最新の実験ディレクトリ: {latest_experiment}")
    
#     # ステップ2: episode_logsサブディレクトリをチェック
#     episode_logs_dir = os.path.join(latest_experiment, "episode_logs")
#     if os.path.exists(episode_logs_dir):
#         # episode_logsディレクトリ内を検索
#         log_pattern = os.path.join(episode_logs_dir, "episode_*")
#         log_files = glob.glob(log_pattern)
#     else:
#         # 直接実験ディレクトリ内を検索
#         log_pattern = os.path.join(latest_experiment, "episode_*")
#         log_files = glob.glob(log_pattern)
    
#     if not log_files:
#         print(f"エピソードログが見つかりません。検索パターン: {log_pattern}")
#         return None
        
#     # 最新のログファイルを取得
#     latest_log = sorted(log_files)[-1]
#     print(f"最新のログファイル: {latest_log}")
    
#     # ファイル拡張子を確認
#     if latest_log.endswith(".json"):
#         # 通常のJSONファイル
#         return latest_log
#     elif latest_log.endswith(".json.tz") or latest_log.endswith(".tz"):
#         # 圧縮ファイル - 一時的に解凍する必要がある
#         print(f"圧縮されたログファイルを検出: {latest_log}")
#         try:
#             # tarファイルとして開く試み
#             with tarfile.open(latest_log, "r:*") as tar:
#                 json_file = None
#                 for member in tar.getmembers():
#                     if member.name.endswith(".json"):
#                         json_file = member
#                         break
                
#                 if json_file:
#                     # JSONファイルを一時ディレクトリに抽出
#                     temp_dir = os.path.join(latest_experiment, "temp")
#                     os.makedirs(temp_dir, exist_ok=True)
#                     tar.extract(json_file, temp_dir)
#                     return os.path.join(temp_dir, json_file.name)
#                 else:
#                     print("圧縮ファイル内にJSONファイルが見つかりません")
#         except Exception as e:
#             print(f"圧縮ファイルの処理中にエラーが発生: {e}")
            
#             # 別の方法: カスタム圧縮形式の可能性
#             try:
#                 # 単純にファイルを開いてJSONとして解析してみる
#                 with open(latest_log, 'r') as f:
#                     content = f.read()
#                     # JSONとして解析できるかテスト
#                     json.loads(content)
#                     return latest_log  # 成功すれば、そのまま返す
#             except:
#                 print("ファイルをJSONとして解析できません")
    
#     print("適切なログファイルが見つかりませんでした")
#     return None

# if __name__ == "__main__":
#     log_file = find_episode_logs()
#     if log_file:
#         print(f"\n使用可能なログファイル: {log_file}")
        
#         # ファイル内容の確認（最初の数行だけ）
#         try:
#             with open(log_file, 'r') as f:
#                 content = f.read(500)  # 最初の500バイトだけ
#                 print("\nログファイルの内容プレビュー:")
#                 print(content[:200] + "...")  # 最初の200文字だけ
#         except Exception as e:
#             print(f"ファイル読み込みエラー: {e}")
#     else:
#         print("\nログファイルが見つかりませんでした。別の場所を確認してください。")
        
#         # 別の場所も探してみる
#         print("\nすべてのログファイルを検索中...")
#         all_logs = []
#         for root, dirs, files in os.walk("./logs"):
#             for file in files:
#                 if "episode" in file.lower() and (".json" in file.lower() or ".tz" in file.lower()):
#                     all_logs.append(os.path.join(root, file))
        
#         if all_logs:
#             print("\n見つかったすべてのログファイル:")
#             for log in all_logs:
#                 print(f"- {log}")
#         else:
#             print("ログファイルが見つかりません。")