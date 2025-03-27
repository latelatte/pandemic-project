from visualization.interactive_map import visualize_interactive_map
from visualization.city_network import visualize_game_state
from visualization.game_viewer import PandemicViewer
import os
import json
import sys

from find_logs import find_episode_logs

log_file = find_episode_logs()
print(f"ログファイル {log_file} を使用して可視化を実行します")


with open(log_file, 'r') as f:
    game_log = json.load(f)
    

output_dir = os.path.dirname(log_file)
    
# create interactive map
map_output = os.path.join(output_dir, "pandemic_map.html")
visualize_interactive_map(log_file, map_output)
print(f"インタラクティブマップを生成しました: {map_output}")
# open the game viewer
viewer = PandemicViewer()
viewer.run(log_file) 
# create city network
network_output = os.path.join(output_dir, "city_network.png")
visualize_game_state(game_log, network_output)
    
print("\n全ての可視化が完了しました。")