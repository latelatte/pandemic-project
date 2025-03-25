import json
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class SimulationLogger:
    """シミュレーションのログとTensorBoard出力を管理するクラス"""
    
    def __init__(self, log_dir="./logs"):
        self.writer = SummaryWriter(log_dir)
        
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
        """各エピソードのログをJSONファイルに保存"""
        log_data = simulation.get_game_log()
        log_data['episode'] = episode_num
        
        # JSONファイルに保存
        with open(f"episode_{episode_num}_log.json", 'w') as f:
            json.dump(log_data, f, indent=2)
            
    def log_experiment_summary(self, win_count, total_episodes, avg_turns):
        """実験全体のサマリーをTensorBoardに記録"""
        self.writer.add_scalar('Summary/WinRate', 100.0 * win_count / total_episodes, 0)
        self.writer.add_scalar('Summary/AvgTurns', avg_turns, 0)
    
    def close(self):
        """TensorBoardのリソースを解放"""
        self.writer.close()