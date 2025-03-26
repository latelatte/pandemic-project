from visualization.game_viewer import PandemicViewer
import os
import sys

# コマンドラインから指定されたファイルを使用するか、最新のログを検索
if len(sys.argv) > 1:
    log_file = sys.argv[1]
    print(f"指定されたログファイル {log_file} を使用します")
else:
    # 最新のログを探す
    log_dirs = sorted([d for d in os.listdir("./logs") if d.startswith("experiment_")])
    if not log_dirs:
        print("ログディレクトリが見つかりません")
        sys.exit(1)
        
    latest_dir = os.path.join("./logs", log_dirs[-1])
    episode_files = sorted([f for f in os.listdir(latest_dir) if f.startswith("episode_") and f.endswith(".json")])
    
    if not episode_files:
        print(f"{latest_dir}にエピソードファイルが見つかりません")
        sys.exit(1)
        
    log_file = os.path.join(latest_dir, episode_files[-1])
    print(f"最新のログファイル {log_file} を使用します")

# PyGameビューワーを起動
viewer = PandemicViewer()
viewer.run(log_file)