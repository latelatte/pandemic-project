# import os
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# import glob
# import pandas as pd
# from collections import defaultdict

# def create_learning_curves(logs_dir, output_dir=None):
#     """create learning curves from the logs directory"""
#     if output_dir is None:
#         output_dir = os.path.join(logs_dir, "plots")
#     os.makedirs(output_dir, exist_ok=True)
    
#     # use tensorboard logs if available or fallback to json
#     metrics_file = os.path.join(logs_dir, "metrics.json")
#     if not os.path.exists(metrics_file):
#         print(f"WARN: metrics file {metrics_file} not found")
#         return False

#     episode_logs = sorted(glob.glob(os.path.join(logs_dir, "episode_logs", "episode_*_log.json")))
#     if not episode_logs:
#         print("WARN: No episode logs found")
#         return False
    
#     # collect data by each episode
#     episode_data = []
#     for log_file in episode_logs:
#         try:
#             with open(log_file, 'r') as f:
#                 log = json.load(f)
#                 episode_num = log.get('episode', -1)
#                 win = log.get('win', False)
#                 turns = log.get('turns', 0)
                
#                 agent_actions = defaultdict(int)
#                 for turn in log.get('turns_log', []):
#                     if 'action' in turn and 'player' in turn:
#                         agent = turn.get('player', {}).get('strategy', 'unknown')
#                         agent_actions[agent] += 1
                
#                 episode_data.append({
#                     'episode': episode_num,
#                     'win': 1 if win else 0,
#                     'turns': turns,
#                     'agent_actions': dict(agent_actions)
#                 })
#         except Exception as e:
#             print(f"error: {log_file}: {e}")
    
#     if not episode_data:
#         print("WARN: No valid episode data found")
#         return False
    
#     df = pd.DataFrame(episode_data)
#     df = df.sort_values('episode')

#     window_size = min(10, len(df))
#     df['rolling_win_rate'] = df['win'].rolling(window=window_size).mean()

#     df['cumulative_win_rate'] = df['win'].expanding().mean()
    
#     plt.figure(figsize=(12, 6))
#     plt.plot(df['episode'], df['rolling_win_rate'], label=f'Rolling Win Rate (window={window_size})')
#     plt.plot(df['episode'], df['cumulative_win_rate'], label='Cumulative Win Rate')
#     plt.title('Learning Curve - Win Rate Progression', fontsize=16)
#     plt.xlabel('Episode', fontsize=12)
#     plt.ylabel('Win Rate', fontsize=12)
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, "learning_curve_win_rate.png"), dpi=300)
#     plt.close()

#     plt.figure(figsize=(12, 6))
#     plt.plot(df['episode'], df['turns'])
#     plt.title('Learning Curve - Average Turns per Episode', fontsize=16)
#     plt.xlabel('Episode', fontsize=12)
#     plt.ylabel('Turns', fontsize=12)
#     plt.grid(True, linestyle='--', alpha=0.7)
    
#     # enphasize wins
#     win_episodes = df[df['win'] == 1]
#     plt.scatter(win_episodes['episode'], win_episodes['turns'], color='green', 
#                 marker='o', label='Win', zorder=5)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, "learning_curve_turns.png"), dpi=300)
#     plt.close()
    
#     print(f"learning curves saved to {output_dir}.")
#     return True

# if __name__ == "__main__":
#     logs_dir = "./evaluation"
#     output_dir = "./plots"
#     create_learning_curves(logs_dir, output_dir)