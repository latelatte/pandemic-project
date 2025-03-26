from visualization.interactive_map import visualize_interactive_map
from visualization.city_network import visualize_game_state
from visualization.game_viewer import PandemicViewer
import os
import json
import sys

# ログファイルを探す（find_logs.pyから関数をインポート）
from find_logs import find_episode_logs

# ログファイルを取得
log_file = find_episode_logs()
print(f"ログファイル {log_file} を使用して可視化を実行します")


with open(log_file, 'r') as f:
    game_log = json.load(f)
    
# 出力ディレクトリ
output_dir = os.path.dirname(log_file)
    
# 1. インタラクティブマップ生成
map_output = os.path.join(output_dir, "pandemic_map.html")
visualize_interactive_map(log_file, map_output)
print(f"インタラクティブマップを生成しました: {map_output}")
# 2. ゲームビューアの起動
viewer = PandemicViewer()
viewer.run(log_file) 
# 3. 静的ネットワーク図の生成
network_output = os.path.join(output_dir, "city_network.png")
visualize_game_state(game_log, network_output)
    
print("\n全ての可視化が完了しました。")