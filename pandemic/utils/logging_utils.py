import json
import numpy as np
import gzip
import os
from torch.utils.tensorboard import SummaryWriter

class SimulationLogger:
    """シミュレーションのログとTensorBoard出力を管理するクラス"""
    
    def __init__(self, log_dir="./logs"):
        self.writer = SummaryWriter(log_dir)
        self.log_dir = log_dir
        
    def log_episode(self, simulation, episode, win=False, metrics=None):
        """エピソードごとのデータをTensorBoardとJSONに記録"""
        # TensorBoardへの記録
        self.writer.add_scalar('Metrics/Turns', simulation.turn_count, episode)
        self.writer.add_scalar('Metrics/Outbreaks', simulation.outbreak_count, episode)
        self.writer.add_scalar('Metrics/AvgInfection', 
                              sum(c.infection_level for c in simulation.cities) / 
                              len(simulation.cities), episode)
        
        # エージェント別のメトリクス
        if metrics and 'time_per_move' in metrics:
            for name, times in metrics['time_per_move'].items():
                if times:  # 空でなければ
                    self.writer.add_scalar(f'Agent/{name}/MoveTimes', 
                                         np.mean(times), episode)
        
        # 勝敗記録
        win_value = 1 if win else 0
        self.writer.add_scalar('GameResult/Win', win_value, episode)
        
    def save_episode_log(self, simulation, episode_num):
        """各エピソードのログを通常のJSONとして保存（圧縮なし）"""
        log_data = simulation.get_game_log()
        log_data['episode'] = episode_num
        
        # エピソード単位で分割して保存（巨大化を防止）
        log_dir = os.path.join(self.log_dir, "episode_logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # 圧縮なしのJSON形式に変更
        log_file = os.path.join(log_dir, f"episode_{episode_num}_log.json")
        
        # 通常のJSON形式で保存
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
            
    def log_experiment_summary(self, win_count, total_episodes, avg_turns):
        """実験全体のサマリーをTensorBoardに記録"""
        self.writer.add_scalar('Summary/WinRate', 100.0 * win_count / total_episodes, 0)
        self.writer.add_scalar('Summary/AvgTurns', avg_turns, 0)
    
    def close(self):
        """TensorBoardのリソースを解放"""
        self.writer.close()