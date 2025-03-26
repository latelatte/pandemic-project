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
        
        self.agent_stats = {}
        
        if agent_names:
            for name in agent_names:
                self.metrics['treatment_counts'][name] = 0
                self.metrics['win_rates'][name] = 0
                self.metrics['time_per_move'][name] = []
                # 各エージェント用の統計も初期化
                self.agent_stats[name] = {
                    "total_time": 0.0,
                    "calls": 0,
                    "win_contribution": 0,
                    "treatments": 0
                }

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
    
    def record_action_time(self, agent_name, time_taken):
        """エージェントのアクション実行時間を記録"""
        if agent_name not in self.agent_stats:
            self.agent_stats[agent_name] = {
                "total_time": 0.0,
                "calls": 0,
                "win_contribution": 0,
                "treatments": 0
            }
        # ここでちゃんとcallsをインクリメントしているか確認
        self.agent_stats[agent_name]["total_time"] += time_taken
        self.agent_stats[agent_name]["calls"] += 1  # この行が実行されているか確認
    
    def record_treatment(self, agent_name, count=1):
        """治療アクションを記録"""
        self.metrics['treatment_counts'][agent_name] += count
    
    def get_summary(self):
        """メトリクスのサマリーを返す"""
        # 空の配列に対するエラー回避
        summary = {
            'avg_turns': 0.0,
            'avg_outbreaks': 0.0,
            'win_rate': 0.0,
            'loss_rate': 1.0,
            'agent_performance': {},
            'resource_usage': {}
        }
        
        # 安全な平均計算
        if self.metrics['total_turns']:
            summary['avg_turns'] = float(np.mean(self.metrics['total_turns']))
        if self.metrics['outbreak_counts']:
            summary['avg_outbreaks'] = float(np.mean(self.metrics['outbreak_counts']))
        
        # エージェントのパフォーマンス情報
        for agent_name, stats in self.agent_stats.items():
            # 分母が0の場合は0.0を設定
            calls = max(1, stats.get("calls", 0))  # 0で割らないようにする
            avg_time = stats.get("total_time", 0) / calls
            
            summary['agent_performance'][agent_name] = {
                "avg_time_ms": float(avg_time * 1000.0),  # msに変換、floatに明示的に変換
                "win_contribution": stats.get("win_contribution", 0),
                "treatments": stats.get("treatments", 0)
            }
        
        return summary

