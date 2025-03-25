import time
import numpy as np

class MetricsCollector:
    """シミュレーションのパフォーマンス指標を収集するクラス"""
    
    def __init__(self, agent_names=None):
        self.metrics = {
            'total_turns': [],
            'outbreak_counts': [],
            'treatment_counts': {},
            'win_rates': {},
            'avg_infection_level': [],
            'time_per_move': {},
            'resource_usage': {}
        }
        
        if agent_names:
            for name in agent_names:
                self.metrics['treatment_counts'][name] = 0
                self.metrics['win_rates'][name] = 0
                self.metrics['time_per_move'][name] = []

    def record_game_metrics(self, simulation, win=False):
        """ゲーム終了時のメトリクスを記録"""
        self.metrics['total_turns'].append(simulation.turn_count)
        self.metrics['outbreak_counts'].append(simulation.outbreak_count)
        self.metrics['avg_infection_level'].append(
            sum(c.infection_level for c in simulation.cities) / len(simulation.cities)
        )
        
        # 勝敗記録
        if win:
            for p in simulation.players:
                self.metrics['win_rates'][p.strategy_name] += 1/len(simulation.players)
    
    def record_action_time(self, agent_name, duration):
        """エージェントの行動時間を記録"""
        self.metrics['time_per_move'][agent_name].append(duration)
    
    def record_treatment(self, agent_name, count=1):
        """治療アクションを記録"""
        self.metrics['treatment_counts'][agent_name] += count
    
    def get_summary(self):
        """メトリクスのサマリーを返す"""
        summary = {
            'avg_turns': np.mean(self.metrics['total_turns']),
            'avg_outbreaks': np.mean(self.metrics['outbreak_counts']),
            'avg_infection': np.mean(self.metrics['avg_infection_level']),
            'agent_performance': {}
        }
        
        for name in self.metrics['time_per_move']:
            summary['agent_performance'][name] = {
                'avg_time_ms': np.mean(self.metrics['time_per_move'][name]) * 1000,
                'win_contribution': self.metrics['win_rates'].get(name, 0),
                'treatments': self.metrics['treatment_counts'].get(name, 0)
            }
        
        return summary

