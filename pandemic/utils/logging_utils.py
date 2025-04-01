import json
import numpy as np
import gzip
import os
from torch.utils.tensorboard import SummaryWriter

class SimulationLogger:
    """class for logs and TensorBoard"""
    
    def __init__(self, log_dir="./logs"):
        self.writer = SummaryWriter(log_dir)
        self.log_dir = log_dir
        
    def log_episode(self, simulation, episode, win=False, metrics=None):
        """save episode logs to TensorBoard"""
        self.writer.add_scalar('Metrics/Turns', simulation.turn_count, episode)
        self.writer.add_scalar('Metrics/Outbreaks', simulation.outbreak_count, episode)
        self.writer.add_scalar('Metrics/AvgInfection', 
                              sum(c.infection_level for c in simulation.cities) / 
                              len(simulation.cities), episode)
        
        if metrics and 'time_per_move' in metrics:
            for name, times in metrics['time_per_move'].items():
                if times:  # not empty
                    self.writer.add_scalar(f'Agent/{name}/MoveTimes', 
                                         np.mean(times), episode)
        

        win_value = 1 if win else 0
        self.writer.add_scalar('GameResult/Win', win_value, episode)
        
        if hasattr(self, '_win_history'):
            self._win_history.append(1 if win else 0)
        else:
            self._win_history = [1 if win else 0]
        
        window_size = min(100, len(self._win_history))
        recent_win_rate = sum(self._win_history[-window_size:]) / window_size
        self.writer.add_scalar('Learning/RecentWinRate', recent_win_rate, episode)
        
        cumulative_win_rate = sum(self._win_history) / len(self._win_history)
        self.writer.add_scalar('Learning/CumulativeWinRate', cumulative_win_rate, episode)
        
        if metrics and 'agent_performance' in metrics:
            for name, perf in metrics['agent_performance'].items():
                if 'win_rates' in perf:
                    self.writer.add_scalar(f'Learning/{name}/WinContribution', 
                                           perf['win_rates'], episode)
        
    def save_episode_log(self, simulation, episode_num):
        """save episode log to file"""
        log_data = simulation.get_game_log()
        log_data['episode'] = episode_num
        
        log_dir = os.path.join(self.log_dir, "episode_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"episode_{episode_num}_log.json")
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
            
    def log_experiment_summary(self, win_count, total_episodes, avg_turns):
        """save experiment summary to TensorBoard"""
        self.writer.add_scalar('Summary/WinRate', 100.0 * win_count / total_episodes, 0)
        self.writer.add_scalar('Summary/AvgTurns', avg_turns, 0)
    
    def close(self):
        self.writer.close()