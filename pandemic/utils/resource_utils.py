import time
import psutil
import os
import numpy as np  # 平均値計算に使用

class ResourceMonitor:
    """エージェントのリソース使用量を計測するクラス"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.measurements = {}
        
    def start_measurement(self, agent_name):
        """計測開始前の状態を記録"""
        if agent_name not in self.measurements:
            self.measurements[agent_name] = {
                'memory': [],
                'cpu_percent': [],
                'timestamps': []
            }
        
        # 現在のベースライン状態を記録
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.start_time = time.time()
        
    def end_measurement(self, agent_name):
        """エージェント実行後のリソース状態を記録"""
        memory_usage = self.process.memory_info().rss / 1024 / 1024 - self.baseline_memory
        cpu_percent = self.process.cpu_percent()
        elapsed = time.time() - self.start_time
        
        self.measurements[agent_name]['memory'].append(memory_usage)
        self.measurements[agent_name]['cpu_percent'].append(cpu_percent)
        self.measurements[agent_name]['timestamps'].append(elapsed)
        
        return {
            'memory_mb': memory_usage,
            'cpu_percent': cpu_percent,
            'time_sec': elapsed
        }
    
    def get_agent_summary(self, agent_name):
        """特定のエージェントのリソース使用量サマリーを取得"""
        if agent_name not in self.measurements:
            return None
            
        data = self.measurements[agent_name]
        
        if not data['memory'] or not data['cpu_percent'] or not data['timestamps']:
            return None
            
        return {
            "avg_memory_mb": sum(data['memory']) / len(data['memory']),
            "avg_cpu_percent": sum(data['cpu_percent']) / len(data['cpu_percent']),
            "avg_time_sec": sum(data['timestamps']) / len(data['timestamps'])
        }
    
    def get_summary(self):
        """全エージェントのリソース使用量サマリーを取得"""
        summary = {}  # ここが修正ポイント：新しい辞書を作成
        
        for agent_name, data in self.measurements.items():
            # データが存在するか確認
            if data['memory'] and data['cpu_percent'] and data['timestamps']:
                summary[agent_name] = {
                    "avg_memory_mb": sum(data['memory']) / len(data['memory']),
                    "avg_cpu_percent": sum(data['cpu_percent']) / len(data['cpu_percent']),
                    "avg_time_sec": sum(data['timestamps']) / len(data['timestamps'])
                }
            else:
                # データがない場合のデフォルト値
                summary[agent_name] = {
                    "avg_memory_mb": 0.0,
                    "avg_cpu_percent": 0.0,
                    "avg_time_sec": 0.0
                }
        
        return summary
        
    def print_stats(self):
        """コンソールにリソース使用状況を表示（デバッグ用）"""
        print("\n== リソース使用状況 ==")
        summary = self.get_summary()
        
        for agent_name, stats in summary.items():
            print(f"エージェント: {agent_name}")
            print(f"  平均メモリ使用: {stats['avg_memory_mb']:.2f} MB")
            print(f"  平均CPU使用率: {stats['avg_cpu_percent']:.2f}%")
            print(f"  平均実行時間: {stats['avg_time_sec'] * 1000:.2f} ms")