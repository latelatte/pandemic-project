from visualization.city_network import visualize_game_state
import os

# 最新の実験結果を見つける
def find_latest_log_file():
    log_dirs = sorted([d for d in os.listdir("./logs") if d.startswith("experiment_")])
    if not log_dirs:
        print("ログディレクトリが見つかりません")
        return None
        
    latest_dir = os.path.join("./logs", log_dirs[-1])
    episode_files = sorted([f for f in os.listdir(latest_dir) if f.startswith("episode_") and f.endswith(".json")])
    
    if not episode_files:
        print(f"{latest_dir}にエピソードファイルが見つかりません")
        return None
        
    return os.path.join(latest_dir, episode_files[-1])

# 最新のログを使って可視化
log_file = find_latest_log_file()
if log_file:
    print(f"ログファイル {log_file} を使って可視化します")
    
    # JSONログを読み込んで可視化
    with open(log_file, 'r') as f:
        import json
        game_log = json.load(f)
    
    # 出力ファイル名
    output_dir = os.path.dirname(log_file)
    output_file = os.path.join(output_dir, "city_network.png")
    
    # 可視化実行
    visualize_game_state(game_log, output_file)