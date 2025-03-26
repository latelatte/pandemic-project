from visualization.interactive_map import visualize_interactive_map
import os

# 最新の実験ログディレクトリを見つける
def find_latest_log():
    log_dirs = sorted([d for d in os.listdir("./logs") if d.startswith("experiment_")])
    if not log_dirs:
        return None
    return os.path.join("./logs", log_dirs[-1])

# 最新の実験ログを取得
latest_log_dir = find_latest_log()
if latest_log_dir:
    # 最新のエピソードログを取得
    episode_logs = sorted([f for f in os.listdir(latest_log_dir) if f.startswith("episode_") and f.endswith(".json")])
    if episode_logs:
        latest_episode = os.path.join(latest_log_dir, episode_logs[-1])
        print(f"最新のエピソードログを可視化: {latest_episode}")
        
        # インタラクティブマップを生成
        output_file = os.path.join(latest_log_dir, "pandemic_map.html")
        visualize_interactive_map(latest_episode, output_file)
        print(f"可視化が完了しました。ブラウザで開いてください: {output_file}")
    else:
        print("エピソードログが見つかりません")
else:
    print("実験ログディレクトリが見つかりません")